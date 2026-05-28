import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from yuvProc import getOneFrame

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FUSION_MODEL_PATH = "joint_fusion_net_test22.pth"
REFINE_MODEL_PATH = "flow_refine_net_test22.pth"
PRECOMPUTED_DIR = "./precomputed_flow_pt"
VIS_OUT_DIR = "./visualizations"

# =========================================================================
# 1. 架構一：真・殘差光流微調器
# =========================================================================
class FlowRefineNet(nn.Module):
    def __init__(self):
        super(FlowRefineNet, self).__init__()
        self.refiner = nn.Sequential(
            nn.Conv2d(5, 32, kernel_size=3, padding=1),
            nn.InstanceNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.InstanceNorm2d(32), nn.GELU(),
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

# =========================================================================
# 2. 架構二：抗光影 64通道 殘差融合網路 (🌟 修改回傳值以利視覺化)
# =========================================================================
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
            nn.Conv2d(channels, channels, kernel_size=3, padding=1), 
            nn.InstanceNorm2d(channels), nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1), 
            nn.InstanceNorm2d(channels), nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1) 
        )
        self.gelu = nn.GELU()
    def forward(self, x):
        return self.gelu(x + self.conv(x))

class MicroFusionNet(nn.Module):
    def __init__(self):
        super(MicroFusionNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(8, 64, kernel_size=3, padding=1), nn.InstanceNorm2d(64))
        self.se1 = SEBlock(64)
        self.conv2_d1 = nn.Sequential(nn.Conv2d(64, 32, kernel_size=3, padding=1, dilation=1), ResidualBlock(32))
        self.conv2_d5 = nn.Sequential(nn.Conv2d(64, 32, kernel_size=3, padding=5, dilation=5), ResidualBlock(32))
        self.conv2_d9 = nn.Sequential(nn.Conv2d(64, 32, kernel_size=3, padding=9, dilation=9), ResidualBlock(32))
        self.se2 = SEBlock(96)
        self.conv3 = nn.Sequential(nn.Conv2d(96, 48, kernel_size=3, padding=1), nn.InstanceNorm2d(48))
        self.conv4 = nn.Conv2d(48, 4, kernel_size=3, padding=1) 
        self.gelu = nn.GELU()
        self.temp = nn.Parameter(torch.tensor(10.0)) # 假設你已改為 1.0

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
        raw_logits = out[:, :3, :, :]
        delta_res = torch.tanh(out[:, 3:4, :, :]) * 0.05
        
        weights = F.softmax(raw_logits * self.temp, dim=1)
        fused_base = (weights[:, 0:1] * t_up) + (weights[:, 1:2] * warped_p) + (weights[:, 2:3] * warped_n)
        fused_final = torch.clamp(fused_base + delta_res, 0.0, 1.0)
        
        # 🌟 回傳所有需要觀察的中間特徵
        return fused_final, weights, delta_res, fused_base

# =========================================================================
# 工具函數
# =========================================================================
W8_H = torch.tensor([-1, 4, -11, 40, 40, -11, 4, -1], dtype=torch.float32, device=device).view(1, 1, 1, 8) / 64.0
W8_V = torch.tensor([-1, 4, -11, 40, 40, -11, 4, -1], dtype=torch.float32, device=device).view(1, 1, 8, 1) / 64.0
LAPLACIAN_KERNEL = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=device).view(1, 1, 3, 3)

def gpu_8tap_upscale_y(img_tensor):
    b, c, h, w = img_tensor.shape
    inter = torch.zeros((1, 1, h, w * 2), device=device)
    inter[:, :, :, 0::2] = img_tensor
    inter[:, :, :, 1::2] = F.conv2d(F.pad(img_tensor, (3, 4, 0, 0), mode='reflect'), W8_H)
    out = torch.zeros((1, 1, h * 2, w * 2), device=device)
    out[:, :, 0::2, :] = inter
    out[:, :, 1::2, :] = F.conv2d(F.pad(inter, (0, 0, 3, 4), mode='reflect'), W8_V)
    return out

def warp_tensor_by_grid(t_img, flow_large):
    h, w = t_img.shape[2], t_img.shape[3]
    mesh_y, mesh_x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
    grid_x = mesh_x.float() + flow_large[:, 0, :, :]
    grid_y = mesh_y.float() + flow_large[:, 1, :, :]
    norm_x = 2.0 * grid_x / (w - 1) - 1.0
    norm_y = 2.0 * grid_y / (h - 1) - 1.0
    return F.grid_sample(t_img, torch.stack((norm_x, norm_y), dim=-1), mode='bicubic', padding_mode='reflection', align_corners=True)

def warp_flow(flow_to_warp, flow_guide):
    h, w = flow_to_warp.shape[2], flow_to_warp.shape[3]
    mesh_y, mesh_x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
    grid_x = mesh_x.float() + flow_guide[:, 0, :, :]
    grid_y = mesh_y.float() + flow_guide[:, 1, :, :]
    norm_x = 2.0 * grid_x / (w - 1) - 1.0
    norm_y = 2.0 * grid_y / (h - 1) - 1.0
    return F.grid_sample(flow_to_warp, torch.stack((norm_x, norm_y), dim=-1), mode='bilinear', padding_mode='zeros', align_corners=True)

def get_laplacian_map(tensor_img): return F.conv2d(F.pad(tensor_img, (1, 1, 1, 1), mode='reflect'), LAPLACIAN_KERNEL)

# =========================================================================
# 圖像保存與渲染輔助 (利用 OpenCV)
# =========================================================================
def save_tensor_gray(tensor, path):
    """一般影像 (0~1) 轉為灰階圖儲存"""
    arr = torch.clamp(tensor, 0, 1).squeeze().cpu().numpy()
    img = (arr * 255.0).astype(np.uint8)
    cv2.imwrite(path, img)

def save_tensor_heatmap(tensor, path, colormap=cv2.COLORMAP_JET):
    """機率/權重 (0~1) 轉為熱力圖儲存"""
    arr = torch.clamp(tensor, 0, 1).squeeze().cpu().numpy()
    img = (arr * 255.0).astype(np.uint8)
    heatmap = cv2.applyColorMap(img, colormap)
    cv2.imwrite(path, heatmap)

def save_tensor_diverging(tensor, path):
    """殘差 (-0.05 ~ +0.05) 轉為對比圖儲存，0 變成灰色 128"""
    arr = tensor.squeeze().cpu().numpy()
    # 將範圍從 [-0.05, 0.05] 映射到 [0, 255]
    arr_norm = np.clip((arr + 0.05) / 0.1 * 255.0, 0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(arr_norm, cv2.COLORMAP_TWILIGHT_SHIFTED) # 凸顯正負差異
    cv2.imwrite(path, heatmap)

def save_flow_magnitude(flow_tensor, path):
    """光流向量轉為強度熱力圖"""
    u = flow_tensor[0, 0].cpu().numpy()
    v = flow_tensor[0, 1].cpu().numpy()
    mag, _ = cv2.cartToPolar(u, v)
    mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(mag_norm, cv2.COLORMAP_MAGMA)
    cv2.imwrite(path, heatmap)

# =========================================================================
# 主視覺化流程
# =========================================================================
def main():
    os.makedirs(VIS_OUT_DIR, exist_ok=True)
    
    refine_model = FlowRefineNet().to(device)
    ml_model = MicroFusionNet().to(device)
    if os.path.exists(FUSION_MODEL_PATH) and os.path.exists(REFINE_MODEL_PATH):
        ml_model.load_state_dict(torch.load(FUSION_MODEL_PATH, map_location=device, weights_only=True))
        refine_model.load_state_dict(torch.load(REFINE_MODEL_PATH, map_location=device, weights_only=True))
        print("[*] Models loaded successfully.")
    else:
        print("[!] Warning: Weights not found, using initialized networks.")
    
    refine_model.eval()
    ml_model.eval()

    width_base, height_base = 1920, 1080
    width_el, height_el = 3840, 2160
    bytesPerPel = 2
    
    # 🌟 只挑代表性的 Task 與 QP (極端高與極端低)
    video_configs = {
        'Zombie': {'qps': [27, 37], 'base': '../bitstream/base/odd_ZombieClimbing2_{qp}_0_4.layer0.yuv'},
        'Procession': {'qps': [25, 35], 'base': '../bitstream/base/odd_Procession_{qp}_0_4.layer0.yuv'}
    }
    
    # 🌟 只挑連續兩幀觀察動態
    target_frames = [2, 3] 

    with torch.no_grad():
        for video_name, config in video_configs.items():
            for qp in config['qps']:
                base_file = config['base'].format(qp=qp)
                if not os.path.exists(base_file): continue
                task_flow_dir = os.path.join(PRECOMPUTED_DIR, f"{video_name}_qp{qp}")
                prev_file_path = base_file.replace('base', 'enhance').replace('odd_', 'even_').replace('.layer0.yuv', '.layer1.yuv')
                
                print(f"[*] Processing Visualizations for {video_name} (QP={qp})...")
                
                for t in target_frames:
                    flow_file = os.path.join(task_flow_dir, f"frame_{t}.pt")
                    if not os.path.exists(flow_file): continue
                    
                    # 建立專屬資料夾
                    frame_dir = os.path.join(VIS_OUT_DIR, f"{video_name}_QP{qp}_F{t}")
                    os.makedirs(frame_dir, exist_ok=True)
                    
                    # 讀取 Y channel (10-bit)
                    f_base = getOneFrame(base_file, width_base, height_base, bytesPerPel, t)
                    f_prev = getOneFrame(prev_file_path, width_el, height_el, bytesPerPel, t)
                    f_next = getOneFrame(prev_file_path, width_el, height_el, bytesPerPel, t+1)
                    
                    t_base_y = torch.tensor(f_base['y'].reshape(1080, 1920), dtype=torch.float32, device=device).view(1,1,1080,1920) / 1023.0
                    t_p_y = torch.tensor(f_prev['y'].reshape(2160, 3840), dtype=torch.float32, device=device).view(1,1,2160,3840) / 1023.0
                    t_n_y = torch.tensor(f_next['y'].reshape(2160, 3840), dtype=torch.float32, device=device).view(1,1,2160,3840) / 1023.0
                    t_up_y = gpu_8tap_upscale_y(t_base_y)
                    
                    flow_data_pt = torch.load(flow_file, map_location=device, weights_only=True)
                    
                    # --- 光流與 Refine 計算 ---
                    h, w = 2160, 3840
                    t_up_small = F.interpolate(t_up_y, size=(540, 960), mode='bilinear', align_corners=False)
                    t_p_small = F.interpolate(t_p_y, size=(540, 960), mode='bilinear', align_corners=False)
                    t_n_small = F.interpolate(t_n_y, size=(540, 960), mode='bilinear', align_corners=False)
                    
                    flow_p_t_refined = refine_model(t_up_small, t_p_small, flow_data_pt['fp'].float())
                    flow_n_t_refined = refine_model(t_up_small, t_n_small, flow_data_pt['fn'].float())
                    flow_p_back_refined = refine_model(t_p_small, t_up_small, flow_data_pt['fpb'].float())
                    flow_n_back_refined = refine_model(t_n_small, t_up_small, flow_data_pt['fnb'].float())
                    
                    flow_p_back_warped = warp_flow(flow_p_back_refined, flow_p_t_refined)
                    flow_n_back_warped = warp_flow(flow_n_back_refined, flow_n_t_refined)
                    err_p_gpu = torch.norm(flow_p_t_refined + flow_p_back_warped, dim=1, keepdim=True)
                    err_n_gpu = torch.norm(flow_n_t_refined + flow_n_back_warped, dim=1, keepdim=True)
                    
                    flow_consistency_guide = F.interpolate(torch.tanh(torch.max(err_p_gpu, err_n_gpu) / 3.0), size=(h, w), mode='bilinear', align_corners=False)
                    epi_p_guide = F.interpolate(flow_data_pt['ep'].float(), size=(h, w), mode='bilinear', align_corners=False)
                    epi_n_guide = F.interpolate(flow_data_pt['en'].float(), size=(h, w), mode='bilinear', align_corners=False)
                    
                    scale_h, scale_w = float(h) / 540.0, float(w) / 960.0
                    flow_p_large = F.interpolate(flow_p_t_refined, size=(h, w), mode='bilinear', align_corners=False)
                    flow_p_large[:, 0, :, :] *= scale_w
                    flow_p_large[:, 1, :, :] *= scale_h
                    flow_n_large = F.interpolate(flow_n_t_refined, size=(h, w), mode='bilinear', align_corners=False)
                    flow_n_large[:, 0, :, :] *= scale_w
                    flow_n_large[:, 1, :, :] *= scale_h
                    
                    warped_p = warp_tensor_by_grid(t_p_y, flow_p_large)
                    warped_n = warp_tensor_by_grid(t_n_y, flow_n_large)
                    lap_guide = get_laplacian_map(t_up_y)
                    
                    # --- 核心前向傳播 (取得所有特徵) ---
                    fused_final, weights, delta_res, fused_base = ml_model(
                        t_up_y, warped_p, warped_n, flow_consistency_guide, lap_guide, epi_p_guide, epi_n_guide
                    )
                    
                    # 🌟 匯出圖像
                    # 1. 基礎影像
                    save_tensor_gray(t_up_y, os.path.join(frame_dir, "01_t_up.png"))
                    save_tensor_gray(warped_p, os.path.join(frame_dir, "02_warped_prev.png"))
                    save_tensor_gray(warped_n, os.path.join(frame_dir, "03_warped_next.png"))
                    
                    # 2. 融合權重 (紅色代表信任該像素)
                    save_tensor_heatmap(weights[:, 0:1], os.path.join(frame_dir, "04_weight_up.png"))
                    save_tensor_heatmap(weights[:, 1:2], os.path.join(frame_dir, "05_weight_prev.png"))
                    save_tensor_heatmap(weights[:, 2:3], os.path.join(frame_dir, "06_weight_next.png"))
                    
                    # 3. 殘差與最終融合結果
                    save_tensor_diverging(delta_res, os.path.join(frame_dir, "07_residual_term.png"))
                    save_tensor_gray(fused_base, os.path.join(frame_dir, "08_fused_base_no_res.png"))
                    save_tensor_gray(fused_final, os.path.join(frame_dir, "09_fused_final.png"))
                    
                    # 4. 指引特徵
                    save_tensor_heatmap(flow_consistency_guide, os.path.join(frame_dir, "10_flow_consistency.png",), cv2.COLORMAP_VIRIDIS)
                    save_flow_magnitude(flow_p_large, os.path.join(frame_dir, "11_flow_mag_prev.png"))

if __name__ == '__main__':
    print("=== Visualization Script Start ===")
    main()
    print("=== All visualizations saved to ./visualizations ===")