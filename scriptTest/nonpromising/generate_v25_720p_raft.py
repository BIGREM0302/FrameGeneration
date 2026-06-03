import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from yuvProc import getOneFrame

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FUSION_MODEL_PATH = "joint_fusion_net_v25_720p_best.pth"
REFINE_MODEL_PATH = "flow_refine_net_v25_720p_best.pth"
PRECOMPUTED_DIR = "./lightweight_precomputed_flow_raft_720p"

# =========================================================================
# 1. 架構 (需與訓練腳本一致)
# =========================================================================
class LiteFlowRefineNet(nn.Module):
    def __init__(self):
        super(LiteFlowRefineNet, self).__init__()
        self.refiner = nn.Sequential(
            nn.Conv2d(5, 32, kernel_size=3, padding=1), nn.InstanceNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.InstanceNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 2, kernel_size=3, padding=1)
        )
    def forward(self, t_up_small, t_target_small, flow_small):
        h, w = t_up_small.shape[2], t_up_small.shape[3]
        mesh_y, mesh_x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
        grid_x = mesh_x.float() + flow_small[:, 0, :, :]
        grid_y = mesh_y.float() + flow_small[:, 1, :, :]
        norm_x = 2.0 * grid_x / (w - 1) - 1.0
        norm_y = 2.0 * grid_y / (h - 1) - 1.0
        warped_target = F.grid_sample(t_target_small, torch.stack((norm_x, norm_y), dim=-1), mode='bicubic', padding_mode='reflection', align_corners=True)
        diff = torch.abs(t_up_small - warped_target)
        x = torch.cat([t_up_small, warped_target, diff, flow_small], dim=1)
        return flow_small + self.refiner(x)

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SEBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False), nn.GELU(),
            nn.Linear(channels // reduction, channels, bias=False), nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.shape
        return x * self.fc(x).view(b, c, 1, 1)

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, channels, dilation=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation, groups=channels)
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1)
    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class DWResidualBlock(nn.Module):
    def __init__(self, channels):
        super(DWResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            DepthwiseSeparableConv(channels), nn.InstanceNorm2d(channels), nn.GELU(),
            DepthwiseSeparableConv(channels), nn.InstanceNorm2d(channels), nn.GELU(),
            DepthwiseSeparableConv(channels) 
        )
        self.gelu = nn.GELU()
    def forward(self, x):
        return self.gelu(x + self.conv(x))

class LiteMicroFusionNetV26(nn.Module):
    def __init__(self):
        super(LiteMicroFusionNetV26, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(7, 48, kernel_size=3, padding=1), nn.InstanceNorm2d(48))
        self.se1 = SEBlock(48)
        self.conv2_d1 = nn.Sequential(nn.Conv2d(48, 20, kernel_size=3, padding=1, dilation=1), DWResidualBlock(20))
        self.conv2_d3 = nn.Sequential(nn.Conv2d(48, 20, kernel_size=3, padding=3, dilation=3), DWResidualBlock(20))
        self.conv2_d5 = nn.Sequential(nn.Conv2d(48, 20, kernel_size=3, padding=5, dilation=5), DWResidualBlock(20))
        self.conv2_d7 = nn.Sequential(nn.Conv2d(48, 20, kernel_size=3, padding=7, dilation=7), DWResidualBlock(20))
        self.se2 = SEBlock(80)
        self.conv3 = nn.Sequential(nn.Conv2d(80, 32, kernel_size=3, padding=1), nn.InstanceNorm2d(32))
        self.conv4 = nn.Conv2d(32, 4, kernel_size=3, padding=1)
        self.gelu = nn.GELU()
        self.temp = nn.Parameter(torch.tensor(10.0))

    def forward(self, t_up, warped_p, warped_n, laplacian_guide, epi_p, epi_n):
        diff_p = torch.abs(warped_p - t_up)
        diff_n = torch.abs(warped_n - t_up)
        diff_pn = torch.abs(warped_p - warped_n)
        x = torch.cat([diff_p, diff_n, diff_pn, t_up, laplacian_guide, epi_p, epi_n], dim=1)
        feat1 = self.se1(self.gelu(self.conv1(x)))
        f1 = self.conv2_d1(feat1)
        f3 = self.conv2_d3(feat1)
        f5 = self.conv2_d5(feat1)
        f7 = self.conv2_d7(feat1)
        feat2 = self.se2(torch.cat([f1, f3, f5, f7], dim=1))
        feat3 = self.gelu(self.conv3(feat2))
        out = self.conv4(feat3)
        raw_logits = out[:, :3, :, :]
        delta_res = torch.tanh(out[:, 3:4, :, :]) * 0.05
        weights = F.softmax(raw_logits * self.temp, dim=1)
        fused_base = (weights[:, 0:1] * t_up) + (weights[:, 1:2] * warped_p) + (weights[:, 2:3] * warped_n)
        return torch.clamp(fused_base + delta_res, 0.0, 1.0), weights

# =========================================================================
# 模型載入與幾何輔助函數
# =========================================================================
refine_model = LiteFlowRefineNet().to(device)
ml_model = LiteMicroFusionNetV26().to(device)
if os.path.exists(FUSION_MODEL_PATH) and os.path.exists(REFINE_MODEL_PATH):
    ml_model.load_state_dict(torch.load(FUSION_MODEL_PATH, map_location=device, weights_only=True))
    refine_model.load_state_dict(torch.load(REFINE_MODEL_PATH, map_location=device, weights_only=True))
refine_model.eval()
ml_model.eval()

W8_H = torch.tensor([-1, 4, -11, 40, 40, -11, 4, -1], dtype=torch.float32, device=device).view(1, 1, 1, 8) / 64.0
W8_V = torch.tensor([-1, 4, -11, 40, 40, -11, 4, -1], dtype=torch.float32, device=device).view(1, 1, 8, 1) / 64.0
W4_H = torch.tensor([-0.0625, 0.5625, 0.5625, -0.0625], dtype=torch.float32, device=device).view(1, 1, 1, 4)
W4_V = torch.tensor([-0.0625, 0.5625, 0.5625, -0.0625], dtype=torch.float32, device=device).view(1, 1, 4, 1)

def gpu_8tap_upscale_y(img_tensor):
    b, c, h, w = img_tensor.shape
    inter = torch.zeros((1, 1, h, w * 2), device=device)
    inter[:, :, :, 0::2] = img_tensor
    inter[:, :, :, 1::2] = F.conv2d(F.pad(img_tensor, (3, 4, 0, 0), mode='reflect'), W8_H)
    out = torch.zeros((1, 1, h * 2, w * 2), device=device)
    out[:, :, 0::2, :] = inter
    out[:, :, 1::2, :] = F.conv2d(F.pad(inter, (0, 0, 3, 4), mode='reflect'), W8_V)
    return out

def gpu_4tap_upscale_uv(img_tensor):
    b, c, h, w = img_tensor.shape
    inter = torch.zeros((1, 1, h, w * 2), device=device)
    inter[:, :, :, 0::2] = img_tensor
    inter[:, :, :, 1::2] = F.conv2d(F.pad(img_tensor, (1, 2, 0, 0), mode='reflect'), W4_H)
    out = torch.zeros((1, 1, h * 2, w * 2), device=device)
    out[:, :, 0::2, :] = inter
    out[:, :, 1::2, :] = F.conv2d(F.pad(inter, (0, 0, 1, 2), mode='reflect'), W4_V)
    return out

def warp_tensor_by_grid(t_img, flow_large):
    h, w = t_img.shape[2], t_img.shape[3]
    mesh_y, mesh_x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
    grid_x = mesh_x.float() + flow_large[:, 0, :, :]
    grid_y = mesh_y.float() + flow_large[:, 1, :, :]
    norm_x = 2.0 * grid_x / (w - 1) - 1.0
    norm_y = 2.0 * grid_y / (h - 1) - 1.0
    return F.grid_sample(t_img, torch.stack((norm_x, norm_y), dim=-1), mode='bicubic', padding_mode='reflection', align_corners=True)

LAPLACIAN_KERNEL = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
def get_laplacian_map(tensor_img): return F.conv2d(F.pad(tensor_img, (1, 1, 1, 1), mode='reflect'), LAPLACIAN_KERNEL)

def ml_fuse_channel_pure_gpu(t_up, t_p, t_n, flow_p_refined, flow_n_refined, epi_p, epi_n):
    h, w = t_up.shape[2], t_up.shape[3]
    scale_h, scale_w = float(h) / 720.0, float(w) / 1280.0
    
    flow_p_large = F.interpolate(flow_p_refined, size=(h, w), mode='bilinear', align_corners=False)
    flow_p_large[:, 0, :, :] *= scale_w
    flow_p_large[:, 1, :, :] *= scale_h
    
    flow_n_large = F.interpolate(flow_n_refined, size=(h, w), mode='bilinear', align_corners=False)
    flow_n_large[:, 0, :, :] *= scale_w
    flow_n_large[:, 1, :, :] *= scale_h
    
    epi_p_guide = F.interpolate(epi_p, size=(h, w), mode='bilinear', align_corners=False)
    epi_n_guide = F.interpolate(epi_n, size=(h, w), mode='bilinear', align_corners=False)
    
    warped_p = warp_tensor_by_grid(t_p, flow_p_large)
    warped_n = warp_tensor_by_grid(t_n, flow_n_large)
    lap_guide = get_laplacian_map(t_up)
    
    fused, _ = ml_model(t_up, warped_p, warped_n, lap_guide, epi_p_guide, epi_n_guide)
    return fused

def main():
    width_base, height_base = 1920, 1080
    width_el, height_el = 3840, 2160
    bytesPerPel = 2
    byteN_base = int((width_base * height_base * 1.5) * bytesPerPel)
    
    video_configs = {
        'AMS05': {'qps': [27, 32, 37, 42], 'base': '../bitstream/base/odd_H2_H3_AMS05_{qp}_0_5.layer0.yuv', 'out': '../bitstream/generated/odd_H2_H3_AMS05_{qp}_0_5_gen.layer0.yuv'},
        'WalkInPark': {'qps': [27, 32, 37, 42], 'base': '../bitstream/base/odd_H2_WalkInPark_{qp}_0_4.layer0.yuv', 'out': '../bitstream/generated/odd_H2_WalkInPark_{qp}_0_4_gen.layer0.yuv'},
        'Procession': {'qps': [25, 30, 35, 40], 'base': '../bitstream/base/odd_Procession_{qp}_0_4.layer0.yuv', 'out': '../bitstream/generated/odd_Procession_{qp}_0_4_gen.layer0.yuv'},
        'Zombie': {'qps': [27, 32, 37, 42], 'base': '../bitstream/base/odd_ZombieClimbing2_{qp}_0_4.layer0.yuv', 'out': '../bitstream/generated/odd_ZombieClimbing2_{qp}_0_4_gen.layer0.yuv'}
    }
    
    os.makedirs('../bitstream/generated', exist_ok=True)
    
    with torch.no_grad():
        for video_name, config in video_configs.items():
            for qp in config['qps']:
                base_file = config['base'].format(qp=qp)
                out_file = config['out'].format(qp=qp)
                if not os.path.exists(base_file): continue
                
                frameCount = (os.path.getsize(base_file) // byteN_base) - 1
                task_flow_dir = os.path.join(PRECOMPUTED_DIR, f"{video_name}_qp{qp}")
                
                prev_file_path = base_file.replace('base', 'enhance').replace('odd_', 'even_').replace('.layer0.yuv', '.layer1.yuv')
                
                with open(out_file, 'wb') as f_out:
                    for t in tqdm(range(frameCount), desc=f"Video: {video_name} QP: {qp}", unit="frame"):
                        f_base = getOneFrame(base_file, width_base, height_base, bytesPerPel, t)
                        f_prev = getOneFrame(prev_file_path, width_el, height_el, bytesPerPel, t)
                        f_next = getOneFrame(prev_file_path, width_el, height_el, bytesPerPel, t+1)
                        
                        t_base_y = torch.tensor(f_base['y'].reshape(1080, 1920), dtype=torch.float32, device=device).view(1,1,1080,1920) / 1023.0
                        t_base_u = torch.tensor(f_base['u'].reshape(540, 960), dtype=torch.float32, device=device).view(1,1,540,960) / 1023.0
                        t_base_v = torch.tensor(f_base['v'].reshape(540, 960), dtype=torch.float32, device=device).view(1,1,540,960) / 1023.0
                        t_p_y = torch.tensor(f_prev['y'].reshape(2160, 3840), dtype=torch.float32, device=device).view(1,1,2160,3840) / 1023.0
                        t_p_u = torch.tensor(f_prev['u'].reshape(1080, 1920), dtype=torch.float32, device=device).view(1,1,1080,1920) / 1023.0
                        t_p_v = torch.tensor(f_prev['v'].reshape(1080, 1920), dtype=torch.float32, device=device).view(1,1,1080,1920) / 1023.0
                        t_n_y = torch.tensor(f_next['y'].reshape(2160, 3840), dtype=torch.float32, device=device).view(1,1,2160,3840) / 1023.0
                        t_n_u = torch.tensor(f_next['u'].reshape(1080, 1920), dtype=torch.float32, device=device).view(1,1,1080,1920) / 1023.0
                        t_n_v = torch.tensor(f_next['v'].reshape(1080, 1920), dtype=torch.float32, device=device).view(1,1,1080,1920) / 1023.0

                        t_up_y = gpu_8tap_upscale_y(t_base_y)
                        t_up_u = gpu_4tap_upscale_uv(t_base_u)
                        t_up_v = gpu_4tap_upscale_uv(t_base_v)
                        
                        flow_data_pt = torch.load(os.path.join(task_flow_dir, f"frame_{t}.pt"), map_location=device, weights_only=True)
                        
                        # 🌟 720p 尺寸
                        t_up_y_small = F.interpolate(t_up_y, size=(720, 1280), mode='bilinear', align_corners=False)
                        t_p_y_small = F.interpolate(t_p_y, size=(720, 1280), mode='bilinear', align_corners=False)
                        t_n_y_small = F.interpolate(t_n_y, size=(720, 1280), mode='bilinear', align_corners=False)
                        
                        flow_p_cv = flow_data_pt['fp'].float().to(device)
                        flow_n_cv = flow_data_pt['fn'].float().to(device)
                        epi_p_cv_t = flow_data_pt['ep'].float().to(device)
                        epi_n_cv_t = flow_data_pt['en'].float().to(device)

                        flow_p_t_refined = refine_model(t_up_y_small, t_p_y_small, flow_p_cv)
                        flow_n_t_refined = refine_model(t_up_y_small, t_n_y_small, flow_n_cv)

                        fused_y = ml_fuse_channel_pure_gpu(t_up_y, t_p_y, t_n_y, flow_p_t_refined, flow_n_t_refined, epi_p_cv_t, epi_n_cv_t)
                        fused_u = ml_fuse_channel_pure_gpu(t_up_u, t_p_u, t_n_u, flow_p_t_refined, flow_n_t_refined, epi_p_cv_t, epi_n_cv_t)
                        fused_v = ml_fuse_channel_pure_gpu(t_up_v, t_p_v, t_n_v, flow_p_t_refined, flow_n_t_refined, epi_p_cv_t, epi_n_cv_t)

                        f_out.write((fused_y.squeeze().cpu().numpy() * 1023.0).clip(0, 1023).astype('<u2').tobytes())
                        f_out.write((fused_u.squeeze().cpu().numpy() * 1023.0).clip(0, 1023).astype('<u2').tobytes())
                        f_out.write((fused_v.squeeze().cpu().numpy() * 1023.0).clip(0, 1023).astype('<u2').tobytes())
            torch.cuda.empty_cache()

if __name__ == '__main__':
    main()