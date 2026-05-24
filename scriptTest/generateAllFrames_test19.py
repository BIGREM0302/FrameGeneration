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

# 定義神經網路權重讀取路徑 (必須與訓練端嚴格對齊)
FUSION_MODEL_PATH = "joint_fusion_net_test19.pth"
REFINE_MODEL_PATH = "flow_refine_net_test19.pth"

# =========================================================================
# 1. 網路架構一：微型 ML 光流微調器 (FlowRefineNet)
# =========================================================================
class FlowRefineNet(nn.Module):
    def __init__(self):
        super(FlowRefineNet, self).__init__()
        # 輸入 4 通道 = t_up_small(1) + t_target_small(1) + flow_small(2)
        self.refiner = nn.Sequential(
            nn.Conv2d(4, 12, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(12, 2, kernel_size=3, padding=1) # 輸出 \Delta dx, \Delta dy 修正向量
        )

    def forward(self, t_up_small, t_target_small, flow_small):
        x = torch.cat([t_up_small, t_target_small, flow_small], dim=1)
        delta_flow = self.refiner(x)
        return flow_small + delta_flow

# =========================================================================
# 2. 網路架構二：4通道金字塔殘差預測網路 (MicroFusionNet)
# =========================================================================
class MicroFusionNet(nn.Module):
    def __init__(self):
        super(MicroFusionNet, self).__init__()
        # 輸入 8 通道 (影像殘差x3, t_up, flow_consistency, laplacian, epi_p, epi_n)
        self.conv1 = nn.Conv2d(8, 32, kernel_size=3, padding=1)
        
        # ASPP-Lite 三路並進
        self.conv2_d1 = nn.Conv2d(32, 8, kernel_size=3, padding=1, dilation=1) 
        self.conv2_d5 = nn.Conv2d(32, 8, kernel_size=3, padding=5, dilation=5) 
        self.conv2_d9 = nn.Conv2d(32, 8, kernel_size=3, padding=9, dilation=9) 
        
        self.conv3 = nn.Conv2d(24, 16, kernel_size=3, padding=1)
        
        # 【SOTA 核心改造】輸出通道擴張至 4 通道
        # 通道 0, 1, 2: 預測 Softmax 融合權重
        # 通道 3: 預測高頻細節殘差量 (Delta Residual)
        self.conv4 = nn.Conv2d(16, 4, kernel_size=3, padding=1)
        self.gelu = nn.GELU()
        self.temp = nn.Parameter(torch.tensor(10.0))

    def forward(self, t_up, warped_p, warped_n, flow_consistency, laplacian_guide, epi_p, epi_n):
        diff_p = torch.abs(warped_p - t_up)
        diff_n = torch.abs(warped_n - t_up)
        diff_pn = torch.abs(warped_p - warped_n)

        x = torch.cat([diff_p, diff_n, diff_pn, t_up, flow_consistency, laplacian_guide, epi_p, epi_n], dim=1)
        
        feat1 = self.gelu(self.conv1(x))
        f1 = self.gelu(self.conv2_d1(feat1))
        f5 = self.gelu(self.conv2_d5(feat1))
        f9 = self.gelu(self.conv2_d9(feat1))
        
        feat2 = torch.cat([f1, f5, f9], dim=1)
        feat3 = self.gelu(self.conv3(feat2))
        
        out = self.conv4(feat3)
        raw_logits = out[:, :3, :, :] 
        
        # 第 4 通道 Tanh 嚴格限幅細節幻想量，自適應銳利化密集人頭與碎石背景
        delta_res = torch.tanh(out[:, 3:4, :, :]) * 0.05
        
        weights = F.softmax(raw_logits * self.temp, dim=1)
        w_up = weights[:, 0:1, :, :]
        w_p = weights[:, 1:2, :, :]
        w_n = weights[:, 2:3, :, :]
        
        fused_base = (w_up * t_up) + (w_p * warped_p) + (w_n * warped_n)
        fused_final = torch.clamp(fused_base + delta_res, 0.0, 1.0)
        return fused_final, weights

# 初始化神經網路雙大腦並載入權重
refine_model = FlowRefineNet().to(device)
ml_model = MicroFusionNet().to(device)

if os.path.exists(FUSION_MODEL_PATH) and os.path.exists(REFINE_MODEL_PATH):
    ml_model.load_state_dict(torch.load(FUSION_MODEL_PATH, map_location=device))
    refine_model.load_state_dict(torch.load(REFINE_MODEL_PATH, map_location=device))
    print(f"✅ 成功無損載入完全體雙大腦權重！\n   1. {FUSION_MODEL_PATH}\n   2. {REFINE_MODEL_PATH}")
else:
    print("⚠️ 找不到對應權重檔，將使用隨機初始權重推論")

ml_model.eval()
refine_model.eval()

# =========================================================================
# 3. 輔助 GPU 升頻器與 Warp 核心邏輯 (完美沿用 test18 黃金地基)
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

def warp_tensor_by_grid(t_img, flow_4k):
    h, w = t_img.shape[2], t_img.shape[3]
    mesh_y, mesh_x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
    grid_x = mesh_x.float() + flow_4k[0, 0, :, :]
    grid_y = mesh_y.float() + flow_4k[0, 1, :, :]
    norm_x = 2.0 * grid_x / (w - 1) - 1.0
    norm_y = 2.0 * grid_y / (h - 1) - 1.0
    grid = torch.stack((norm_x, norm_y), dim=-1).unsqueeze(0)
    return F.grid_sample(t_img, grid, mode='bicubic', padding_mode='reflection', align_corners=True)

def get_small_flow_img(tensor_4k):
    t_small = F.interpolate(tensor_4k, scale_factor=0.25, mode='bilinear', align_corners=False)
    return torch.clamp(t_small.squeeze() * 255.0, 0, 255).byte().cpu().numpy()

# 全域拉普拉斯卷積核
LAPLACIAN_KERNEL = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=device).view(1, 1, 3, 3)

def get_laplacian_map(tensor_img):
    return F.conv2d(F.pad(tensor_img, (1, 1, 1, 1), mode='reflect'), LAPLACIAN_KERNEL)

def compute_epipolar_error_map(flow, shape):
    H, W = shape[0], shape[1]
    step_y, step_x = max(1, H // 32), max(1, W // 32)
    pts1, pts2 = [], []
    for y in range(0, H, step_y):
        for x in range(0, W, step_x):
            pts1.append([x, y])
            pts2.append([x + flow[y, x, 0], y + flow[y, x, 1]])
    pts1 = np.array(pts1, dtype=np.float32)
    pts2 = np.array(pts2, dtype=np.float32)
    
    F_mat, _ = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 1.0, 0.99)
    if F_mat is not None and F_mat.shape == (3, 3):
        y_idx, x_idx = np.indices((H, W))
        pts1_all = np.stack((x_idx, y_idx, np.ones_like(x_idx)), axis=-1).reshape(-1, 3)
        pts2_all = np.stack((x_idx + flow[..., 0], y_idx + flow[..., 1], np.ones_like(x_idx)), axis=-1).reshape(-1, 3)
        
        lines2 = np.dot(pts1_all, F_mat.T)
        num = np.abs(np.sum(pts2_all * lines2, axis=1))
        den = np.sqrt(lines2[:, 0]**2 + lines2[:, 1]**2) + 1e-6
        err_map = (num / den).reshape(H, W)
        return np.tanh(err_map / 4.0)[..., np.newaxis]
    else:
        return np.zeros((H, W, 1), dtype=np.float32)

# =========================================================================
# 5. 完全體核心推理融合函式 (ML光流微調洗禮 + 4通道自適應細節補償)
# =========================================================================
def ml_fuse_channel(t_up, t_p, t_n, flow_p_cv, flow_n_cv, flow_p_back_cv, flow_n_back_cv):
    h, w = t_up.shape[2], t_up.shape[3]
    
    # 準備 540p 的低解析度影像特徵傳給 FlowRefineNet
    t_up_small = F.interpolate(t_up, size=(540, 960), mode='bilinear', align_corners=False)
    t_p_small = F.interpolate(t_p, size=(540, 960), mode='bilinear', align_corners=False)
    t_n_small = F.interpolate(t_n, size=(540, 960), mode='bilinear', align_corners=False)
    
    # 機器學習光流微調大腦發威：即時修正 OpenCV 錯位的密集人流與肢體向量！
    flow_p_t_refined = refine_model(t_up_small, t_p_small, flow_p_cv)
    flow_n_t_refined = refine_model(t_up_small, t_n_small, flow_n_cv)
    flow_p_back_refined = refine_model(t_p_small, t_up_small, flow_p_back_cv)
    flow_n_back_refined = refine_model(t_n_small, t_up_small, flow_n_back_cv)
    
    # 2. 利用被 ML 修正過的光流，動態計算推論端的時域與空間幾何信任雷達
    fp_np = flow_p_t_refined.squeeze().permute(1,2,0).cpu().numpy()
    fn_np = flow_n_t_refined.squeeze().permute(1,2,0).cpu().numpy()
    fp_back_np = flow_p_back_refined.squeeze().permute(1,2,0).cpu().numpy()
    fn_back_np = flow_n_back_refined.squeeze().permute(1,2,0).cpu().numpy()
    
    err_p_small = np.linalg.norm(fp_np + fp_back_np, axis=2, keepdims=True)
    err_n_small = np.linalg.norm(fn_np + fn_back_np, axis=2, keepdims=True)
    err_consistent_small = np.maximum(err_p_small, err_n_small)
    err_consistent_small = np.tanh(err_consistent_small / 3.0)

    # 抽取 540p 解析度下的極線幾何約束
    epi_p_small = compute_epipolar_error_map(fp_np, (540, 960))
    epi_n_small = compute_epipolar_error_map(fn_np, (540, 960))

    # 包裝成 Tensor 並無損金字塔放大到與當前通道匹配的解析度 (Y是 4K, UV是 1080p)
    flow_consistency_t = torch.tensor(err_consistent_small, dtype=torch.float32, device=device).permute(2,0,1).unsqueeze(0)
    epi_p_t = torch.tensor(epi_p_small, dtype=torch.float32, device=device).permute(2,0,1).unsqueeze(0)
    epi_n_t = torch.tensor(epi_n_small, dtype=torch.float32, device=device).permute(2,0,1).unsqueeze(0)
    
    flow_consistency_guide = F.interpolate(flow_consistency_t, size=(h, w), mode='bilinear', align_corners=False)
    epi_p_guide = F.interpolate(epi_p_t, size=(h, w), mode='bilinear', align_corners=False)
    epi_n_guide = F.interpolate(epi_n_t, size=(h, w), mode='bilinear', align_corners=False)
    
    # 放大光流場向量
    scale_h = float(h) / 540.0
    scale_w = float(w) / 960.0
    flow_p_large = F.interpolate(flow_p_t_refined, size=(h, w), mode='bilinear', align_corners=False)
    flow_p_large[:, 0, :, :] *= scale_w
    flow_p_large[:, 1, :, :] *= scale_h
    
    flow_n_large = F.interpolate(flow_n_t_refined, size=(h, w), mode='bilinear', align_corners=False)
    flow_n_large[:, 0, :, :] *= scale_w
    flow_n_large[:, 1, :, :] *= scale_h
    
    # 精準 Warp
    warped_p = warp_tensor_by_grid(t_p, flow_p_large)
    warped_n = warp_tensor_by_grid(t_n, flow_n_large)
    
    # 計算拉普拉斯高頻
    lap_guide = get_laplacian_map(t_up)
    
    # 送入 4通道 MicroFusionNet，自動輸出加上幻想細節殘差的超高品質影格！
    fused, _ = ml_model(t_up, warped_p, warped_n, flow_consistency_guide, lap_guide, epi_p_guide, epi_n_guide)
    return fused

# =========================================================================
# 6. 主推論生成迴圈
# =========================================================================
def main():
    width_base, height_base = 1920, 1080
    width_el, height_el = 3840, 2160
    bytesPerPel = 2
    byteN_base = int((width_base * height_base * 1.5) * bytesPerPel)
    
    video_configs = {
        'AMS05': {'qps': [27, 32, 37, 42], 'base': '../bitstream/base/odd_H2_H3_AMS05_{qp}_0_5.layer0.yuv', 'even': '../bitstream/enhance/even_H2_H3_AMS05_{qp}_0_5.layer1.yuv', 'out': '../bitstream/generated/odd_H2_H3_AMS05_{qp}_0_5_gen.layer0.yuv'},
        'WalkInPark': {'qps': [27, 32, 37, 42], 'base': '../bitstream/base/odd_H2_WalkInPark_{qp}_0_4.layer0.yuv', 'even': '../bitstream/enhance/even_H2_WalkInPark_{qp}_0_4.layer1.yuv', 'out': '../bitstream/generated/odd_H2_WalkInPark_{qp}_0_4_gen.layer0.yuv'},
        'Procession': {'qps': [25, 30, 35, 40], 'base': '../bitstream/base/odd_Procession_{qp}_0_4.layer0.yuv', 'even': '../bitstream/enhance/even_Procession_{qp}_0_4.layer1.yuv', 'out': '../bitstream/generated/odd_Procession_{qp}_0_4_gen.layer0.yuv'},
        'Zombie': {'qps': [27, 32, 37, 42], 'base': '../bitstream/base/odd_ZombieClimbing2_{qp}_0_4.layer0.yuv', 'even': '../bitstream/enhance/even_ZombieClimbing2_{qp}_0_4.layer1.yuv', 'out': '../bitstream/generated/odd_ZombieClimbing2_{qp}_0_4_gen.layer0.yuv'}
    }
    
    os.makedirs('../bitstream/generated', exist_ok=True)
    
    with torch.no_grad():
        for video_name, config in video_configs.items():
            print(f"\n==================== 開始推理影片: {video_name} ====================")
            for qp in config['qps']:
                base_file = config['base'].format(qp=qp)
                even_file = config['even'].format(qp=qp)
                out_file = config['out'].format(qp=qp)
                if not os.path.exists(base_file) or not os.path.exists(even_file): continue
                
                frameCount = (os.path.getsize(base_file) // byteN_base) - 1
                
                with open(out_file, 'wb') as f_out:
                    for t in tqdm(range(frameCount), desc=f"Video: {video_name} QP: {qp}", unit="frame"):
                        f_base = getOneFrame(base_file, width_base, height_base, bytesPerPel, t)
                        f_prev = getOneFrame(even_file, width_el, height_el, bytesPerPel, t)
                        f_next = getOneFrame(even_file, width_el, height_el, bytesPerPel, t+1)
                        
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
                        
                        up_8b = get_small_flow_img(t_up_y)
                        p_8b = get_small_flow_img(t_p_y)
                        n_8b = get_small_flow_img(t_n_y)
                        
                        # 在 540p 空間計算 OpenCV 的粗糙基礎光流矩陣
                        flow_p_small = cv2.calcOpticalFlowFarneback(up_8b, p_8b, None, 0.5, 5, 25, 3, 5, 1.2, 0)
                        flow_n_small = cv2.calcOpticalFlowFarneback(up_8b, n_8b, None, 0.5, 5, 25, 3, 5, 1.2, 0)
                        flow_p_back = cv2.calcOpticalFlowFarneback(p_8b, up_8b, None, 0.5, 5, 25, 3, 5, 1.2, 0)
                        flow_n_back = cv2.calcOpticalFlowFarneback(n_8b, up_8b, None, 0.5, 5, 25, 3, 5, 1.2, 0)
                        
                        flow_p_cv = torch.tensor(flow_p_small, dtype=torch.float32, device=device).permute(2,0,1).unsqueeze(0)
                        flow_n_cv = torch.tensor(flow_n_small, dtype=torch.float32, device=device).permute(2,0,1).unsqueeze(0)
                        flow_p_back_cv = torch.tensor(flow_p_back, dtype=torch.float32, device=device).permute(2,0,1).unsqueeze(0)
                        flow_n_back_cv = torch.tensor(flow_n_back, dtype=torch.float32, device=device).permute(2,0,1).unsqueeze(0)

                        # 交叉融合：將 OpenCV 粗糙光流餵入，在 ml_fuse_channel 內部解鎖完全體雙模型引導
                        fused_y = ml_fuse_channel(t_up_y, t_p_y, t_n_y, flow_p_cv, flow_n_cv, flow_p_back_cv, flow_n_back_cv)
                        fused_u = ml_fuse_channel(t_up_u, t_p_u, t_n_u, flow_p_cv, flow_n_cv, flow_p_back_cv, flow_n_back_cv)
                        fused_v = ml_fuse_channel(t_up_v, t_p_v, t_n_v, flow_p_cv, flow_n_cv, flow_p_back_cv, flow_n_back_cv)

                        # 清洗並寫入 10-bit YUV
                        f_out.write((fused_y.squeeze().cpu().numpy() * 1023.0).clip(0, 1023).astype('<u2').tobytes())
                        f_out.write((fused_u.squeeze().cpu().numpy() * 1023.0).clip(0, 1023).astype('<u2').tobytes())
                        f_out.write((fused_v.squeeze().cpu().numpy() * 1023.0).clip(0, 1023).astype('<u2').tobytes())
                        
                print(f"    [成功] 影格安全儲存至: {out_file}")
            torch.cuda.empty_cache()

if __name__ == '__main__':
    main()