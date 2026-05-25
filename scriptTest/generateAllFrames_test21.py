import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from yuvProc import getOneFrame

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"推論使用設備: {device}")

FUSION_MODEL_PATH = "joint_fusion_net_test21.pth"
REFINE_MODEL_PATH = "flow_refine_net_test21.pth"
PRECOMPUTED_DIR = "./precomputed_flow_pt"

# =========================================================================
# 網路架構 (與訓練端 test22 嚴格對齊)
# =========================================================================
class FlowRefineNet(nn.Module):
    def __init__(self):
        super(FlowRefineNet, self).__init__()
        self.refiner = nn.Sequential(
            nn.Conv2d(4, 12, kernel_size=3, padding=1), nn.GELU(), nn.Conv2d(12, 2, kernel_size=3, padding=1)
        )
    def forward(self, t_up_small, t_target_small, flow_small):
        x = torch.cat([t_up_small, t_target_small, flow_small], dim=1)
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

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )
        self.gelu = nn.GELU()
    def forward(self, x):
        return self.gelu(x + self.conv(x))

class MicroFusionNet(nn.Module):
    def __init__(self):
        super(MicroFusionNet, self).__init__()
        self.conv1 = nn.Conv2d(8, 32, kernel_size=3, padding=1)
        self.se1 = SEBlock(32)
        self.conv2_d1 = nn.Sequential(nn.Conv2d(32, 8, kernel_size=3, padding=1, dilation=1), ResidualBlock(8))
        self.conv2_d5 = nn.Sequential(nn.Conv2d(32, 8, kernel_size=3, padding=5, dilation=5), ResidualBlock(8))
        self.conv2_d9 = nn.Sequential(nn.Conv2d(32, 8, kernel_size=3, padding=9, dilation=9), ResidualBlock(8))
        self.se2 = SEBlock(24)
        self.conv3 = nn.Conv2d(24, 16, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(16, 4, kernel_size=3, padding=1)
        self.gelu = nn.GELU()
        self.temp = nn.Parameter(torch.tensor(10.0))

    def forward(self, t_up, warped_p, warped_n, flow_consistency, laplacian_guide, epi_p, epi_n):
        diff_p = torch.abs(warped_p - t_up)
        diff_n = torch.abs(warped_n - t_up)
        diff_pn = torch.abs(warped_p - warped_n)
        x = torch.cat([diff_p, diff_n, diff_pn, t_up, flow_consistency, laplacian_guide, epi_p, epi_n], dim=1)
        feat1 = self.se1(self.gelu(self.conv1(x)))
        f1 = self.conv2_d1(feat1)
        f5 = self.conv2_d5(feat1)
        f9 = self.conv2_d9(feat1)
        feat2 = self.se2(torch.cat([f1, f5, f9], dim=1))
        feat3 = self.gelu(self.conv3(feat2))
        out = self.conv4(feat3)
        delta_res = torch.tanh(out[:, 3:4, :, :]) * 0.05
        weights = F.softmax(out[:, :3, :, :] * self.temp, dim=1)
        fused_base = (weights[:, 0:1] * t_up) + (weights[:, 1:2] * warped_p) + (weights[:, 2:3] * warped_n)
        return torch.clamp(fused_base + delta_res, 0.0, 1.0), weights

refine_model = FlowRefineNet().to(device)
ml_model = MicroFusionNet().to(device)
if os.path.exists(FUSION_MODEL_PATH) and os.path.exists(REFINE_MODEL_PATH):
    ml_model.load_state_dict(torch.load(FUSION_MODEL_PATH, map_location=device, weights_only=True))
    refine_model.load_state_dict(torch.load(REFINE_MODEL_PATH, map_location=device, weights_only=True))
    print("✅ 載入 test22 原生 GPU 完全體推論權重成功！")
refine_model.eval()
ml_model.eval()

# =========================================================================
# 幾何輔助函數 (100% GPU 化)
# =========================================================================
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
    grid_x = mesh_x.float() + flow_large[0, 0, :, :]
    grid_y = mesh_y.float() + flow_large[0, 1, :, :]
    norm_x = 2.0 * grid_x / (w - 1) - 1.0
    norm_y = 2.0 * grid_y / (h - 1) - 1.0
    return F.grid_sample(t_img, torch.stack((norm_x, norm_y), dim=-1).unsqueeze(0), mode='bicubic', padding_mode='reflection', align_corners=True)

LAPLACIAN_KERNEL = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
def get_laplacian_map(tensor_img): return F.conv2d(F.pad(tensor_img, (1, 1, 1, 1), mode='reflect'), LAPLACIAN_KERNEL)

# =========================================================================
# 推理融合函式 (純 GPU .pt 快取資料流)
# =========================================================================
def ml_fuse_channel_pure_gpu(t_up, t_p, t_n, flow_data_pt):
    h, w = t_up.shape[2], t_up.shape[3]
    t_up_small = F.interpolate(t_up, size=(540, 960), mode='bilinear', align_corners=False)
    t_p_small = F.interpolate(t_p, size=(540, 960), mode='bilinear', align_corners=False)
    t_n_small = F.interpolate(t_n, size=(540, 960), mode='bilinear', align_corners=False)
    
    flow_p_cv = flow_data_pt['fp'].float()
    flow_n_cv = flow_data_pt['fn'].float()
    flow_p_back_cv = flow_data_pt['fpb'].float()
    flow_n_back_cv = flow_data_pt['fnb'].float()
    epi_p_cv_t = flow_data_pt['ep'].float()
    epi_n_cv_t = flow_data_pt['en'].float()
    
    flow_p_t_refined = refine_model(t_up_small, t_p_small, flow_p_cv)
    flow_n_t_refined = refine_model(t_up_small, t_n_small, flow_n_cv)
    flow_p_back_refined = refine_model(t_p_small, t_up_small, flow_p_back_cv)
    flow_n_back_refined = refine_model(t_n_small, t_up_small, flow_n_back_cv)
    
    err_p_gpu = torch.norm(flow_p_t_refined + flow_p_back_refined, dim=1, keepdim=True)
    err_n_gpu = torch.norm(flow_n_t_refined + flow_n_back_refined, dim=1, keepdim=True)
    flow_consistency_t = torch.tanh(torch.max(err_p_gpu, err_n_gpu) / 3.0)
    
    flow_consistency_guide = F.interpolate(flow_consistency_t, size=(h, w), mode='bilinear', align_corners=False)
    epi_p_guide = F.interpolate(epi_p_cv_t, size=(h, w), mode='bilinear', align_corners=False)
    epi_n_guide = F.interpolate(epi_n_cv_t, size=(h, w), mode='bilinear', align_corners=False)
    
    scale_h, scale_w = float(h) / 540.0, float(w) / 960.0
    flow_p_large = F.interpolate(flow_p_t_refined, size=(h, w), mode='bilinear', align_corners=False)
    flow_p_large[:, 0, :, :] *= scale_w
    flow_p_large[:, 1, :, :] *= scale_h
    flow_n_large = F.interpolate(flow_n_t_refined, size=(h, w), mode='bilinear', align_corners=False)
    flow_n_large[:, 0, :, :] *= scale_w
    flow_n_large[:, 1, :, :] *= scale_h
    
    warped_p = warp_tensor_by_grid(t_p, flow_p_large)
    warped_n = warp_tensor_by_grid(t_n, flow_n_large)
    lap_guide = get_laplacian_map(t_up)
    
    fused, _ = ml_model(t_up, warped_p, warped_n, flow_consistency_guide, lap_guide, epi_p_guide, epi_n_guide)
    return fused

# =========================================================================
# 主推理生成迴圈
# =========================================================================
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
            print(f"\n==================== 推理影片: {video_name} ====================")
            for qp in config['qps']:
                base_file = config['base'].format(qp=qp)
                out_file = config['out'].format(qp=qp)
                if not os.path.exists(base_file): continue
                
                frameCount = (os.path.getsize(base_file) // byteN_base) - 1
                task_flow_dir = os.path.join(PRECOMPUTED_DIR, f"{video_name}_qp{qp}")
                
                if not os.path.exists(task_flow_dir):
                    print(f"⚠️ 找不到原生快取資料夾 {task_flow_dir}，請先確認 precompute 腳本已完成")
                    continue
                
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

                        fused_y = ml_fuse_channel_pure_gpu(t_up_y, t_p_y, t_n_y, flow_data_pt)
                        fused_u = ml_fuse_channel_pure_gpu(t_up_u, t_p_u, t_n_u, flow_data_pt)
                        fused_v = ml_fuse_channel_pure_gpu(t_up_v, t_p_v, t_n_v, flow_data_pt)

                        f_out.write((fused_y.squeeze().cpu().numpy() * 1023.0).clip(0, 1023).astype('<u2').tobytes())
                        f_out.write((fused_u.squeeze().cpu().numpy() * 1023.0).clip(0, 1023).astype('<u2').tobytes())
                        f_out.write((fused_v.squeeze().cpu().numpy() * 1023.0).clip(0, 1023).astype('<u2').tobytes())
                print(f"    [成功] 4K YUV 影片渲染完成: {out_file}")
            torch.cuda.empty_cache()

if __name__ == '__main__':
    main()