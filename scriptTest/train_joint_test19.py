import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from yuvProc import getOneFrame
from tqdm import tqdm
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"訓練使用設備: {device}")

# 定義模型儲存路徑
FUSION_MODEL_PATH = "joint_fusion_net_test19.pth"
REFINE_MODEL_PATH = "flow_refine_net_test19.pth"

# =========================================================================
# 1. 網路架構一：微型 ML 光流微調器 (FlowRefineNet)
# =========================================================================
class FlowRefineNet(nn.Module):
    """ 
    SOTA 降維打擊精髓：在 540p 低解析度空間下，主動修正 OpenCV Farneback 算錯的密集人頭與殭屍手腳向量
    參數極少（約數百個），在 540p 下跑兩層卷積，分攤到 4K 每個像素的運算量小於 5K MACs，極度硬體友善！
    """
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
        # 最終 ML 精準光流 = OpenCV 粗糙光流 + ML 修正殘差
        return flow_small + delta_flow

# =========================================================================
# 2. 網路架構二：4通道金字塔殘差預測網路 (MicroFusionNet)
# =========================================================================
class MicroFusionNet(nn.Module):
    def __init__(self):
        super(MicroFusionNet, self).__init__()
        # 輸入 8 通道 (影像殘差x3, t_up, flow_consistency, laplacian, epi_p, epi_n)
        self.conv1 = nn.Conv2d(8, 32, kernel_size=3, padding=1)
        
        # ASPP-Lite 三路並進，給予 32 中間通道寬敞的解耦舞台
        self.conv2_d1 = nn.Conv2d(32, 8, kernel_size=3, padding=1, dilation=1) 
        self.conv2_d5 = nn.Conv2d(32, 8, kernel_size=3, padding=5, dilation=5) 
        self.conv2_d9 = nn.Conv2d(32, 8, kernel_size=3, padding=9, dilation=9) 
        
        self.conv3 = nn.Conv2d(24, 16, kernel_size=3, padding=1)
        
        # 【SOTA 核心改造】輸出通道從 3 提升至 4 通道
        # 通道 0, 1, 2: 預測 w_up, w_p, w_n 物理融合權重
        # 通道 3: 預測高頻細節殘差量 (Delta Residual)
        self.conv4 = nn.Conv2d(16, 4, kernel_size=3, padding=1)
        self.gelu = nn.GELU()
        self.temp = nn.Parameter(torch.tensor(10.0))

    def forward(self, t_up, warped_p, warped_n, flow_consistency, laplacian_guide, epi_p, epi_n):
        diff_p = torch.abs(warped_p - t_up)
        diff_n = torch.abs(warped_n - t_up)
        diff_pn = torch.abs(warped_p - warped_n)

        # 完美的 8 通道幾何矩陣拼接
        x = torch.cat([diff_p, diff_n, diff_pn, t_up, flow_consistency, laplacian_guide, epi_p, epi_n], dim=1)
        
        feat1 = self.gelu(self.conv1(x))
        f1 = self.gelu(self.conv2_d1(feat1))
        f5 = self.gelu(self.conv2_d5(feat1))
        f9 = self.gelu(self.conv2_d9(feat1))
        
        feat2 = torch.cat([f1, f5, f9], dim=1)
        feat3 = self.gelu(self.conv3(feat2))
        
        # 提取 4 通道預測結果
        out = self.conv4(feat3)
        raw_logits = out[:, :3, :, :] # 前 3 個通道計算 Softmax 融合權重
        
        # 【精髓所在】第 4 通道透過 Tanh 限制高頻幻想幅度在 [-0.05, 0.05] 之間
        # 這能給予網路自適應畫出草皮與人頭細節的特權，同時嚴格限制 Zombie 遮擋區的數值暴走
        delta_res = torch.tanh(out[:, 3:4, :, :]) * 0.05
        
        weights = F.softmax(raw_logits * self.temp, dim=1)
        w_up = weights[:, 0:1, :, :]
        w_p = weights[:, 1:2, :, :]
        w_n = weights[:, 2:3, :, :]
        
        # 基礎黃金幾何保底融合
        fused_base = (w_up * t_up) + (w_p * warped_p) + (w_n * warped_n)
        
        # 加上大腦幻化出來的高頻細節殘差，並做安全鉗制 (Clamp)
        fused_final = torch.clamp(fused_base + delta_res, 0.0, 1.0)
        return fused_final, weights

# =========================================================================
# 3. 輔助 GPU 函數與幾何函數
# =========================================================================
W8_H = torch.tensor([-1, 4, -11, 40, 40, -11, 4, -1], dtype=torch.float32, device=device).view(1, 1, 1, 8) / 64.0
W8_V = torch.tensor([-1, 4, -11, 40, 40, -11, 4, -1], dtype=torch.float32, device=device).view(1, 1, 8, 1) / 64.0

def gpu_8tap_upscale_y(img_tensor):
    b, c, h, w = img_tensor.shape
    inter = torch.zeros((1, 1, h, w * 2), device=device)
    inter[:, :, :, 0::2] = img_tensor
    inter[:, :, :, 1::2] = F.conv2d(F.pad(img_tensor, (3, 4, 0, 0), mode='reflect'), W8_H)
    out = torch.zeros((1, 1, h * 2, w * 2), device=device)
    out[:, :, 0::2, :] = inter
    out[:, :, 1::2, :] = F.conv2d(F.pad(inter, (0, 0, 3, 4), mode='reflect'), W8_V)
    return out

def get_small_flow_img(tensor_4k):
    t_small = F.interpolate(tensor_4k, scale_factor=0.25, mode='bilinear', align_corners=False)
    return torch.clamp(t_small.squeeze() * 255.0, 0, 255).byte().cpu().numpy()

def warp_tensor(t_img, flow_4k):
    h, w = t_img.shape[2], t_img.shape[3]
    mesh_y, mesh_x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
    grid_x = mesh_x.float() + flow_4k[0, 0, :, :]
    grid_y = mesh_y.float() + flow_4k[0, 1, :, :]
    norm_x = 2.0 * grid_x / (w - 1) - 1.0
    norm_y = 2.0 * grid_y / (h - 1) - 1.0
    grid = torch.stack((norm_x, norm_y), dim=-1).unsqueeze(0)
    return F.grid_sample(t_img, grid, mode='bicubic', padding_mode='reflection', align_corners=True)

def compute_epipolar_error_map(flow, shape):
    """ 利用兩視圖幾何計算全畫面像素與極線的幾何距離誤差 """
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
        err_map = np.tanh(err_map / 4.0) # Tanh 軟飽和穩定數值域
        return err_map[..., np.newaxis]
    else:
        return np.zeros((H, W, 1), dtype=np.float32)

def get_random_crops(t_up, t_p, t_n, t_gt, t_consistent, t_epi_p, t_epi_n, num_crops=64, patch_size=256):
    """ 同步隨機裁切所有特徵圖管道，保持 Batch 尺寸為安全高效的 256x256 """
    _, _, H, W = t_up.shape
    up_crops, p_crops, n_crops, gt_crops, cons_crops, epi_p_crops, epi_n_crops = [], [], [], [], [], [], []
    for _ in range(num_crops):
        h = random.randint(0, H - patch_size)
        w = random.randint(0, W - patch_size)
        up_crops.append(t_up[:, :, h:h+patch_size, w:w+patch_size])
        p_crops.append(t_p[:, :, h:h+patch_size, w:w+patch_size])
        n_crops.append(t_n[:, :, h:h+patch_size, w:w+patch_size])
        gt_crops.append(t_gt[:, :, h:h+patch_size, w:w+patch_size])
        cons_crops.append(t_consistent[:, :, h:h+patch_size, w:w+patch_size])
        epi_p_crops.append(t_epi_p[:, :, h:h+patch_size, w:w+patch_size])
        epi_n_crops.append(t_epi_n[:, :, h:h+patch_size, w:w+patch_size])
    return (torch.cat(up_crops), torch.cat(p_crops), torch.cat(n_crops), 
            torch.cat(gt_crops), torch.cat(cons_crops), torch.cat(epi_p_crops), torch.cat(epi_n_crops))

LAPLACIAN_KERNEL = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=device).view(1, 1, 3, 3)

def get_laplacian_map(tensor_img):
    return F.conv2d(F.pad(tensor_img, (1, 1, 1, 1), mode='reflect'), LAPLACIAN_KERNEL)

# =========================================================================
# 4. 主 8 任務通用大模型聯合訓練迴圈
# =========================================================================
def main():
    train_tasks = [
        {'name': 'Zombie', 'qp': 27, 'base': '../bitstream/base/odd_ZombieClimbing2_27_0_4.layer0.yuv', 'even': '../bitstream/enhance/even_ZombieClimbing2_27_0_4.layer1.yuv', 'gt': '../orgYUV/odd_Zombie-Climbing2_3840x2160_24fps_10bit_420.yuv'},
        {'name': 'Zombie', 'qp': 37, 'base': '../bitstream/base/odd_ZombieClimbing2_37_0_4.layer0.yuv', 'even': '../bitstream/enhance/even_ZombieClimbing2_37_0_4.layer1.yuv', 'gt': '../orgYUV/odd_Zombie-Climbing2_3840x2160_24fps_10bit_420.yuv'},
        {'name': 'AMS05', 'qp': 27, 'base': '../bitstream/base/odd_H2_H3_AMS05_27_0_5.layer0.yuv', 'even': '../bitstream/enhance/even_H2_H3_AMS05_27_0_5.layer1.yuv', 'gt': '../orgYUV/odd_H2_H3_AMS05_3840x2160_10bit_420_HLG.yuv'}, 
        {'name': 'AMS05', 'qp': 37, 'base': '../bitstream/base/odd_H2_H3_AMS05_37_0_5.layer0.yuv', 'even': '../bitstream/enhance/even_H2_H3_AMS05_37_0_5.layer1.yuv', 'gt': '../orgYUV/odd_H2_H3_AMS05_3840x2160_10bit_420_HLG.yuv'}
        # {'name': 'WalkInPark', 'qp': 27, 'base': '../bitstream/base/odd_H2_WalkInPark_27_0_4.layer0.yuv', 'even': '../bitstream/enhance/even_H2_WalkInPark_27_0_4.layer1.yuv', 'gt': '../orgYUV/odd_H2_WalkInPark_3840x2160_10_60fps_HLG.yuv'},
        # {'name': 'WalkInPark', 'qp': 37, 'base': '../bitstream/base/odd_H2_WalkInPark_37_0_4.layer0.yuv', 'even': '../bitstream/enhance/even_H2_WalkInPark_37_0_4.layer1.yuv', 'gt': '../orgYUV/odd_H2_WalkInPark_3840x2160_10_60fps_HLG.yuv'},
        # {'name': 'Procession', 'qp': 25, 'base': '../bitstream/base/odd_Procession_25_0_4.layer0.yuv', 'even': '../bitstream/enhance/even_Procession_25_0_4.layer1.yuv', 'gt': '../orgYUV/odd_Procession_3840x2160_60fps_10bit_420.yuv'},  
        # {'name': 'Procession', 'qp': 35, 'base': '../bitstream/base/odd_Procession_35_0_4.layer0.yuv', 'even': '../bitstream/enhance/even_Procession_35_0_4.layer1.yuv', 'gt': '../orgYUV/odd_Procession_3840x2160_60fps_10bit_420.yuv'}
    ]
    
    width_base, height_base = 1920, 1080
    width_el, height_el = 3840, 2160
    bytesPerPel = 2
    byteN_base = int((width_base * height_base * 1.5) * bytesPerPel)

    # 同步並聯初始化兩個神經網路
    fusion_model = MicroFusionNet().to(device)
    refine_model = FlowRefineNet().to(device)
    
    eps = 1e-6
    # 讓優化器同時監管融合大腦與光流微調大腦的參數更新
    optimizer = optim.AdamW(
        list(fusion_model.parameters()) + list(refine_model.parameters()), 
        lr=1e-3, weight_decay=1e-4
    )
    
    # 鎖定 80 Epochs 長跑，讓加法殘差與機器學習光流完美並行收斂
    epochs = 20
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"🎬 開始【ML光流微調 + 4通道殘差】完全體大模型聯合特訓 (Epochs: {epochs})")
    
    for epoch in range(epochs):
        fusion_model.train()
        refine_model.train()
        random.shuffle(train_tasks) 
        
        for task in train_tasks:
            if not os.path.exists(task['base']) or not os.path.exists(task['gt']):
                continue
                
            frameCount = (os.path.getsize(task['base']) // byteN_base) - 1
            task_loss = 0.0
            
            pbar = tqdm(range(frameCount), desc=f"Epoch {epoch+1}/{epochs} | {task['name']}(QP{task['qp']})", unit="frame")
            for t in pbar:
                f_base = getOneFrame(task['base'], width_base, height_base, bytesPerPel, t)
                f_prev = getOneFrame(task['even'], width_el, height_el, bytesPerPel, t)
                f_next = getOneFrame(task['even'], width_el, height_el, bytesPerPel, t+1)
                f_gt   = getOneFrame(task['gt'], width_el, height_el, bytesPerPel, t)
                
                with torch.no_grad():
                    t_base_y = torch.tensor(f_base['y'].reshape(1080, 1920), dtype=torch.float32, device=device).view(1,1,1080,1920) / 1023.0
                    t_p_y = torch.tensor(f_prev['y'].reshape(2160, 3840), dtype=torch.float32, device=device).view(1,1,2160,3840) / 1023.0
                    t_n_y = torch.tensor(f_next['y'].reshape(2160, 3840), dtype=torch.float32, device=device).view(1,1,2160,3840) / 1023.0
                    t_gt_y = torch.tensor(f_gt['y'].reshape(2160, 3840), dtype=torch.float32, device=device).view(1,1,2160,3840) / 1023.0
                    
                    t_up_y = gpu_8tap_upscale_y(t_base_y)
                    up_8b, p_8b, n_8b = get_small_flow_img(t_up_y), get_small_flow_img(t_p_y), get_small_flow_img(t_n_y)
                    
                    # 1. 傳統 OpenCV Farneback 計算 540p 空間的粗糙光流
                    flow_p_small_cv = cv2.calcOpticalFlowFarneback(up_8b, p_8b, None, 0.5, 5, 25, 3, 5, 1.2, 0)
                    flow_n_small_cv = cv2.calcOpticalFlowFarneback(up_8b, n_8b, None, 0.5, 5, 25, 3, 5, 1.2, 0)
                    flow_p_back_cv = cv2.calcOpticalFlowFarneback(p_8b, up_8b, None, 0.5, 5, 25, 3, 5, 1.2, 0)
                    flow_n_back_cv = cv2.calcOpticalFlowFarneback(n_8b, up_8b, None, 0.5, 5, 25, 3, 5, 1.2, 0)
                    
                    flow_p_t = torch.tensor(flow_p_small_cv, dtype=torch.float32, device=device).permute(2,0,1).unsqueeze(0)
                    flow_n_t = torch.tensor(flow_n_small_cv, dtype=torch.float32, device=device).permute(2,0,1).unsqueeze(0)
                    flow_p_back_t = torch.tensor(flow_p_back_cv, dtype=torch.float32, device=device).permute(2,0,1).unsqueeze(0)
                    flow_n_back_t = torch.tensor(flow_n_back_cv, dtype=torch.float32, device=device).permute(2,0,1).unsqueeze(0)

                # ======== 【解封：ML光流修正階段】有梯度參與優化 ========
                optimizer.zero_grad()
                
                # 準備 540p 影像特徵傳給 FlowRefineNet
                t_up_small = F.interpolate(t_up_y, scale_factor=0.25, mode='bilinear', align_corners=False)
                t_p_small = F.interpolate(t_p_y, scale_factor=0.25, mode='bilinear', align_corners=False)
                t_n_small = F.interpolate(t_n_y, scale_factor=0.25, mode='bilinear', align_corners=False)
                
                # 機器學習大腦發威：修正 OpenCV 算錯的密集人頭、手腳、細碎岩石向量！
                flow_p_t_refined = refine_model(t_up_small, t_p_small, flow_p_t)
                flow_n_t_refined = refine_model(t_up_small, t_n_small, flow_n_t)
                flow_p_back_refined = refine_model(t_p_small, t_up_small, flow_p_back_t)
                flow_n_back_refined = refine_model(t_n_small, t_up_small, flow_n_back_t)
                
                # 2. 利用被 ML 修正過的光流，動態計算高品質的光流一致性誤差與對極約束
                # (轉回 numpy 做幾何演算)
                fp_np = flow_p_t_refined.detach().squeeze().permute(1,2,0).cpu().numpy()
                fn_np = flow_n_t_refined.detach().squeeze().permute(1,2,0).cpu().numpy()
                fp_back_np = flow_p_back_refined.detach().squeeze().permute(1,2,0).cpu().numpy()
                fn_back_np = flow_n_back_refined.detach().squeeze().permute(1,2,0).cpu().numpy()
                
                err_p_small = np.linalg.norm(fp_np + fp_back_np, axis=2, keepdims=True)
                err_n_small = np.linalg.norm(fn_np + fn_back_np, axis=2, keepdims=True)
                err_consistent_small = np.maximum(err_p_small, err_n_small)
                err_consistent_small = np.tanh(err_consistent_small / 3.0)

                epi_p_small = compute_epipolar_error_map(fp_np, up_8b.shape)
                epi_n_small = compute_epipolar_error_map(fn_np, up_8b.shape)

                # 將高品質幾何雷達地圖送上 GPU
                flow_consistency_t = torch.tensor(err_consistent_small, dtype=torch.float32, device=device).permute(2,0,1).unsqueeze(0)
                epi_p_t = torch.tensor(epi_p_small, dtype=torch.float32, device=device).permute(2,0,1).unsqueeze(0)
                epi_n_t = torch.tensor(epi_n_small, dtype=torch.float32, device=device).permute(2,0,1).unsqueeze(0)
                
                # 金字塔無損放大到 4K 空間
                flow_p_4k = F.interpolate(flow_p_t_refined, scale_factor=4.0, mode='bilinear', align_corners=False) * 4.0
                flow_n_4k = F.interpolate(flow_n_t_refined, scale_factor=4.0, mode='bilinear', align_corners=False) * 4.0
                flow_consistency_4k = F.interpolate(flow_consistency_t, scale_factor=4.0, mode='bilinear', align_corners=False)
                epi_p_4k = F.interpolate(epi_p_t, scale_factor=4.0, mode='bilinear', align_corners=False)
                epi_n_4k = F.interpolate(epi_n_t, scale_factor=4.0, mode='bilinear', align_corners=False)
                
                # 執行高品質精準 Warp 扭曲
                warped_p = warp_tensor(t_p_y, flow_p_4k)
                warped_n = warp_tensor(t_n_y, flow_n_4k)
                
                # 進行 8 通道隨機切塊準備訓練
                b_up, b_p, b_n, b_gt, b_cons, b_epi_p, b_epi_n = get_random_crops(
                    t_up_y, warped_p, warped_n, t_gt_y, flow_consistency_4k, epi_p_4k, epi_n_4k, num_crops=64, patch_size=256
                )
                b_lap = get_laplacian_map(b_up)

                # ======== 【解封：融合大腦與加法殘差幻想階段】 ========
                fused_pred, weights = fusion_model(b_up, b_p, b_n, b_cons, b_lap, b_epi_p, b_epi_n)
                
                # 聯合損失計算與反向傳播
                loss = torch.mean(torch.sqrt((fused_pred - b_gt)**2 + eps)) # Charbonnier Loss
                loss.backward()
                optimizer.step()
                
                task_loss += loss.item()
                pbar.set_postfix({'Charbonnier': f"{loss.item():.4f}"})
                
            print(f"   └─ 平均 Loss: {task_loss/frameCount:.5f}")
            torch.cuda.empty_cache()

        scheduler.step()
        print(f"📉 學習率調整為: {optimizer.param_groups[0]['lr']:.6f}")

        # 同步儲存兩組完全體網路權重
        torch.save(fusion_model.state_dict(), FUSION_MODEL_PATH)
        torch.save(refine_model.state_dict(), REFINE_MODEL_PATH)
        print(f"💾 Epoch {epoch+1} 模型已安全儲存至本機硬碟。\n")

if __name__ == '__main__':
    main()