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

# =========================================================================
# 1. 網路架構 (沿用 test8 最強的 ASPP-Lite 結構，僅將輸入第 4 通道換為光流一致性)
# =========================================================================
class MicroFusionNet(nn.Module):
    def __init__(self):
        super(MicroFusionNet, self).__init__()
        # 輸入 4 通道，擴張到 24 通道
        self.conv1 = nn.Conv2d(6, 24, kernel_size=3, padding=1)
        
        # ASPP-Lite 三路並進，沿用 test8 最穩定的 [1, 5, 9] 廣角视野
        self.conv2_d1 = nn.Conv2d(24, 8, kernel_size=3, padding=1, dilation=1) 
        self.conv2_d5 = nn.Conv2d(24, 8, kernel_size=3, padding=5, dilation=5) 
        self.conv2_d9 = nn.Conv2d(24, 8, kernel_size=3, padding=9, dilation=9) 
        
        self.conv3 = nn.Conv2d(24, 16, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(16, 3, kernel_size=3, padding=1)
        self.gelu = nn.GELU()
        self.temp = nn.Parameter(torch.tensor(10.0))

    def forward(self, t_up, warped_p, warped_n, flow_consistency, laplacian_guide):
        diff_p = torch.abs(warped_p - t_up)
        diff_n = torch.abs(warped_n - t_up)
        diff_pn = torch.abs(warped_p - warped_n)

        # 【核心突破】x 拼裝：前向誤差、後向誤差、光流時域雙向一致性、升頻底圖
        x = torch.cat([diff_p, diff_n, diff_pn, t_up, flow_consistency, laplacian_guide], dim=1)
        
        feat1 = self.gelu(self.conv1(x))
        
        f1 = self.gelu(self.conv2_d1(feat1))
        f5 = self.gelu(self.conv2_d5(feat1))
        f9 = self.gelu(self.conv2_d9(feat1))
        
        feat2 = torch.cat([f1, f5, f9], dim=1)
        feat3 = self.gelu(self.conv3(feat2))
        logits = self.conv4(feat3) * self.temp 
        
        weights = F.softmax(logits, dim=1)
        w_up = weights[:, 0:1, :, :]
        w_p = weights[:, 1:2, :, :]
        w_n = weights[:, 2:3, :, :]
        
        fused = (w_up * t_up) + (w_p * warped_p) + (w_n * warped_n)
        return fused, weights

# =========================================================================
# 2. 輔助 GPU 函數
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

def get_random_crops(t_up, t_p, t_n, t_gt, t_consistent, num_crops=64, patch_size=256):
    """ 同步隨機裁切所有通道，保持 Batch 尺寸為安全高效的 256x256 """
    _, _, H, W = t_up.shape
    up_crops, p_crops, n_crops, gt_crops, cons_crops = [], [], [], [], []
    for _ in range(num_crops):
        h = random.randint(0, H - patch_size)
        w = random.randint(0, W - patch_size)
        up_crops.append(t_up[:, :, h:h+patch_size, w:w+patch_size])
        p_crops.append(t_p[:, :, h:h+patch_size, w:w+patch_size])
        n_crops.append(t_n[:, :, h:h+patch_size, w:w+patch_size])
        gt_crops.append(t_gt[:, :, h:h+patch_size, w:w+patch_size])
        cons_crops.append(t_consistent[:, :, h:h+patch_size, w:w+patch_size])
    return torch.cat(up_crops), torch.cat(p_crops), torch.cat(n_crops), torch.cat(gt_crops), torch.cat(cons_crops)

# 全域定義拉普拉斯卷積核
LAPLACIAN_KERNEL = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=device).view(1, 1, 3, 3)

def get_laplacian_map(tensor_img):
    """ 在 GPU 端極速提取純粹的高頻邊緣紋理 """
    return F.conv2d(F.pad(tensor_img, (1, 1, 1, 1), mode='reflect'), LAPLACIAN_KERNEL)

# =========================================================================
# 3. 主聯合訓練迴圈
# =========================================================================
def main():
    train_tasks = [
        {'name': 'Zombie', 'qp': 27, 'base': '../bitstream/base/odd_ZombieClimbing2_27_0_4.layer0.yuv', 'even': '../bitstream/enhance/even_ZombieClimbing2_27_0_4.layer1.yuv', 'gt': '../orgYUV/odd_Zombie-Climbing2_3840x2160_24fps_10bit_420.yuv'},
        {'name': 'Zombie', 'qp': 37, 'base': '../bitstream/base/odd_ZombieClimbing2_37_0_4.layer0.yuv', 'even': '../bitstream/enhance/even_ZombieClimbing2_37_0_4.layer1.yuv', 'gt': '../orgYUV/odd_Zombie-Climbing2_3840x2160_24fps_10bit_420.yuv'},
        {'name': 'AMS05', 'qp': 27, 'base': '../bitstream/base/odd_H2_H3_AMS05_27_0_5.layer0.yuv', 'even': '../bitstream/enhance/even_H2_H3_AMS05_27_0_5.layer1.yuv', 'gt': '../orgYUV/odd_H2_H3_AMS05_3840x2160_10bit_420_HLG.yuv'}, 
        {'name': 'AMS05', 'qp': 37, 'base': '../bitstream/base/odd_H2_H3_AMS05_37_0_5.layer0.yuv', 'even': '../bitstream/enhance/even_H2_H3_AMS05_37_0_5.layer1.yuv', 'gt': '../orgYUV/odd_H2_H3_AMS05_3840x2160_10bit_420_HLG.yuv'}
    ]
    
    width_base, height_base = 1920, 1080
    width_el, height_el = 3840, 2160
    bytesPerPel = 2
    byteN_base = int((width_base * height_base * 1.5) * bytesPerPel)

    model = MicroFusionNet().to(device)
    eps = 1e-6
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    epochs = 40
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"🎬 開始聯合訓練 (總任務數: {len(train_tasks)}, Epochs: {epochs})")
    
    for epoch in range(epochs):
        model.train()
        random.shuffle(train_tasks) 
        
        for task in train_tasks:
            if not os.path.exists(task['base']) or not os.path.exists(task['gt']):
                print(f"⚠️ 找不到檔案，跳過: {task['name']} (QP {task['qp']})")
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
                    
                    # 【核心修改：雙向光流計算】
                    # 1. 前向光流 (Base -> Prev, Base -> Next)
                    flow_p_small = cv2.calcOpticalFlowFarneback(up_8b, p_8b, None, 0.5, 5, 25, 3, 5, 1.2, 0)
                    flow_n_small = cv2.calcOpticalFlowFarneback(up_8b, n_8b, None, 0.5, 5, 25, 3, 5, 1.2, 0)
                    
                    # 2. 反向光流 (Prev -> Base, Next -> Base)
                    flow_p_back = cv2.calcOpticalFlowFarneback(p_8b, up_8b, None, 0.5, 5, 25, 3, 5, 1.2, 0)
                    flow_n_back = cv2.calcOpticalFlowFarneback(n_8b, up_8b, None, 0.5, 5, 25, 3, 5, 1.2, 0)
                    
                    # 3. 在 540p 空間計算一致性誤差地圖 |Flow_Forward + Flow_Backward|
                    err_p_small = np.linalg.norm(flow_p_small + flow_p_back, axis=2, keepdims=True)
                    err_n_small = np.linalg.norm(flow_n_small + flow_n_back, axis=2, keepdims=True)
                    # 整合兩邊的一致性誤差 (取最大值或平均值)
                    err_consistent_small = np.maximum(err_p_small, err_n_small)

                    # 4. 把光流與一致性地圖送上 GPU 並做放大
                    flow_p_t = torch.tensor(flow_p_small, dtype=torch.float32, device=device).permute(2,0,1).unsqueeze(0)
                    flow_n_t = torch.tensor(flow_n_small, dtype=torch.float32, device=device).permute(2,0,1).unsqueeze(0)
                    flow_consistency_t = torch.tensor(err_consistent_small, dtype=torch.float32, device=device).permute(2,0,1).unsqueeze(0)
                    
                    flow_p_4k = F.interpolate(flow_p_t, scale_factor=4.0, mode='bilinear', align_corners=False) * 4.0
                    flow_n_4k = F.interpolate(flow_n_t, scale_factor=4.0, mode='bilinear', align_corners=False) * 4.0
                    # 一致性誤差放大到 4K (不乘倍率，因為它是數值誤差，不是位移像素)
                    flow_consistency_4k = F.interpolate(flow_consistency_t, scale_factor=4.0, mode='bilinear', align_corners=False)
                    
                    # 扭曲影像
                    warped_p = warp_tensor(t_p_y, flow_p_4k)
                    warped_n = warp_tensor(t_n_y, flow_n_4k)
                    
                    # 切割出 Batch (多切一個光流一致性誤差地圖)
                    b_up, b_p, b_n, b_gt, b_cons = get_random_crops(t_up_y, warped_p, warped_n, t_gt_y, flow_consistency_4k, num_crops=64, patch_size=256)
                    
                    # 【新增】利用當前升頻影格 b_up (B, 1, 256, 256) 計算拉普拉斯高頻地圖
                    b_lap = get_laplacian_map(b_up)

                # ======== 模型訓練 ========
                optimizer.zero_grad()
                fused_pred, weights = model(b_up, b_p, b_n, b_cons, b_lap)
                loss = torch.mean(torch.sqrt((fused_pred - b_gt)**2 + eps)) # Charbonnier Loss
                
                loss.backward()
                optimizer.step()
                
                task_loss += loss.item()
                pbar.set_postfix({'Charbonnier': f"{loss.item():.4f}"})
                
            print(f"   └─ 平均 Loss: {task_loss/frameCount:.5f}")
            torch.cuda.empty_cache()

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"📉 學習率已調整為: {current_lr:.6f}")

        torch.save(model.state_dict(), "joint_fusion_net_test12.pth")
        print(f"💾 Epoch {epoch+1} 模型已儲存為 joint_fusion_net_test12.pth\n")

if __name__ == '__main__':
    main()