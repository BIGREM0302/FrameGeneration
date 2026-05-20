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

# ==========================================
# 1. 網路架構 (MicroFusionNet - 3 Channel 終極版)
# ==========================================
class MicroFusionNet(nn.Module):
    def __init__(self):
        super(MicroFusionNet, self).__init__()
        # 輸入 4 通道，擴張到 24 通道
        self.conv1 = nn.Conv2d(4, 24, kernel_size=3, padding=1)
        
        # 【王牌外掛：ASPP-Lite】三路並進，捕捉近、中、遠尺度特徵
        self.conv2_d1 = nn.Conv2d(24, 8, kernel_size=3, padding=1, dilation=1) # 看局部細節
        self.conv2_d3 = nn.Conv2d(24, 8, kernel_size=3, padding=3, dilation=3) # 看中等紋理
        self.conv2_d7 = nn.Conv2d(24, 8, kernel_size=3, padding=7, dilation=7) # 看大範圍動態
        
        self.conv3 = nn.Conv2d(24, 16, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(16, 3, kernel_size=3, padding=1)
        self.gelu = nn.GELU()
        
        # 【王牌外掛：可學習的溫度參數】讓模型自己決定融合的銳利度
        self.temp = nn.Parameter(torch.tensor(10.0))

    def forward(self, t_up, warped_p, warped_n):
        diff_p = torch.abs(warped_p - t_up)
        diff_n = torch.abs(warped_n - t_up)
        diff_pn = torch.abs(warped_p - warped_n) # 遮擋偵測
        
        x = torch.cat([diff_p, diff_n, diff_pn, t_up], dim=1)
        
        feat1 = self.gelu(self.conv1(x))
        
        # 進入多尺度池化
        f1 = self.gelu(self.conv2_d1(feat1))
        f3 = self.gelu(self.conv2_d3(feat1))
        f7 = self.gelu(self.conv2_d7(feat1))
        
        # 將三種尺度的特徵組合起來 (8+8+8 = 24通道)
        feat2 = torch.cat([f1, f3, f7], dim=1)
        
        feat3 = self.gelu(self.conv3(feat2))
        logits = self.conv4(feat3) * self.temp # 乘上可學習的溫度參數
        
        weights = F.softmax(logits, dim=1)
        w_up = weights[:, 0:1, :, :]
        w_p = weights[:, 1:2, :, :]
        w_n = weights[:, 2:3, :, :]
        
        fused = (w_up * t_up) + (w_p * warped_p) + (w_n * warped_n)
        return fused, weights

# ==========================================
# 2. 輔助 GPU 函數 (VVC 8-tap 升頻與 Warp)
# ==========================================
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

def get_random_crops(t_up, t_p, t_n, t_gt, num_crops=32, patch_size=128):
    """ 從 4K 畫面中隨機切出 N 個小區塊組成 Batch """
    _, _, H, W = t_up.shape
    up_crops, p_crops, n_crops, gt_crops = [], [], [], []
    for _ in range(num_crops):
        h = random.randint(0, H - patch_size)
        w = random.randint(0, W - patch_size)
        up_crops.append(t_up[:, :, h:h+patch_size, w:w+patch_size])
        p_crops.append(t_p[:, :, h:h+patch_size, w:w+patch_size])
        n_crops.append(t_n[:, :, h:h+patch_size, w:w+patch_size])
        gt_crops.append(t_gt[:, :, h:h+patch_size, w:w+patch_size])
    return torch.cat(up_crops), torch.cat(p_crops), torch.cat(n_crops), torch.cat(gt_crops)

# ==========================================
# 3. 主聯合訓練迴圈 (Joint Training)
# ==========================================
def main():
    # 建立聯合訓練的資料集清單 (包含高/低 QP 的 Zombie 與 AMS05)
    train_tasks = [
        {'name': 'Zombie', 'qp': 27, 
         'base': '../bitstream/base/odd_ZombieClimbing2_27_0_4.layer0.yuv', 
         'even': '../bitstream/enhance/even_ZombieClimbing2_27_0_4.layer1.yuv', 
         'gt': '../orgYUV/odd_Zombie-Climbing2_3840x2160_24fps_10bit_420.yuv'},
        {'name': 'Zombie', 'qp': 37, 
         'base': '../bitstream/base/odd_ZombieClimbing2_37_0_4.layer0.yuv', 
         'even': '../bitstream/enhance/even_ZombieClimbing2_37_0_4.layer1.yuv', 
         'gt': '../orgYUV/odd_Zombie-Climbing2_3840x2160_24fps_10bit_420.yuv'},
        {'name': 'AMS05', 'qp': 27, 
         'base': '../bitstream/base/odd_H2_H3_AMS05_27_0_5.layer0.yuv', 
         'even': '../bitstream/enhance/even_H2_H3_AMS05_27_0_5.layer1.yuv', 
         'gt': '../orgYUV/odd_H2_H3_AMS05_3840x2160_10bit_420_HLG.yuv'}, 
        {'name': 'AMS05', 'qp': 37, 
         'base': '../bitstream/base/odd_H2_H3_AMS05_37_0_5.layer0.yuv', 
         'even': '../bitstream/enhance/even_H2_H3_AMS05_37_0_5.layer1.yuv', 
         'gt': '../orgYUV/odd_H2_H3_AMS05_3840x2160_10bit_420_HLG.yuv'}
    ]
    
    width_base, height_base = 1920, 1080
    width_el, height_el = 3840, 2160
    bytesPerPel = 2
    byteN_base = int((width_base * height_base * 1.5) * bytesPerPel)

    # 初始化模型、Loss 與 Optimizer
    model = MicroFusionNet().to(device)
    eps = 1e-6
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    epochs = 20
    # 【新增】Cosine 學習率衰減器，讓模型在最後階段完美收斂
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"🎬 開始聯合訓練 (總任務數: {len(train_tasks)}, Epochs: {epochs})")
    
    for epoch in range(epochs):
        model.train()
        
        # 打亂任務順序，避免模型死背特定影片的特徵
        random.shuffle(train_tasks) 
        
        for task in train_tasks:
            # 檢查檔案是否存在，避免報錯中斷
            if not os.path.exists(task['base']) or not os.path.exists(task['gt']):
                print(f"⚠️ 找不到檔案，跳過: {task['name']} (QP {task['qp']})")
                continue
                
            frameCount = (os.path.getsize(task['base']) // byteN_base) - 1
            task_loss = 0.0
            
            pbar = tqdm(range(frameCount), desc=f"Epoch {epoch+1}/{epochs} | {task['name']}(QP{task['qp']})", unit="frame")
            for t in pbar:
                # 讀取資料
                f_base = getOneFrame(task['base'], width_base, height_base, bytesPerPel, t)
                f_prev = getOneFrame(task['even'], width_el, height_el, bytesPerPel, t)
                f_next = getOneFrame(task['even'], width_el, height_el, bytesPerPel, t+1)
                f_gt   = getOneFrame(task['gt'], width_el, height_el, bytesPerPel, t)
                
                # 轉為 Tensor
                with torch.no_grad():
                    t_base_y = torch.tensor(f_base['y'].reshape(1080, 1920), dtype=torch.float32, device=device).view(1,1,1080,1920) / 1023.0
                    t_p_y = torch.tensor(f_prev['y'].reshape(2160, 3840), dtype=torch.float32, device=device).view(1,1,2160,3840) / 1023.0
                    t_n_y = torch.tensor(f_next['y'].reshape(2160, 3840), dtype=torch.float32, device=device).view(1,1,2160,3840) / 1023.0
                    t_gt_y = torch.tensor(f_gt['y'].reshape(2160, 3840), dtype=torch.float32, device=device).view(1,1,2160,3840) / 1023.0
                    
                    # 升頻與光流
                    t_up_y = gpu_8tap_upscale_y(t_base_y)
                    up_8b, p_8b, n_8b = get_small_flow_img(t_up_y), get_small_flow_img(t_p_y), get_small_flow_img(t_n_y)
                    
                    flow_p_small = cv2.calcOpticalFlowFarneback(up_8b, p_8b, None, 0.5, 5, 25, 3, 5, 1.2, 0)
                    flow_n_small = cv2.calcOpticalFlowFarneback(up_8b, n_8b, None, 0.5, 5, 25, 3, 5, 1.2, 0)
                    
                    flow_p_t = torch.tensor(flow_p_small, dtype=torch.float32, device=device).permute(2,0,1).unsqueeze(0)
                    flow_n_t = torch.tensor(flow_n_small, dtype=torch.float32, device=device).permute(2,0,1).unsqueeze(0)
                    
                    flow_p_4k = F.interpolate(flow_p_t, scale_factor=4.0, mode='bilinear', align_corners=False) * 4.0
                    flow_n_4k = F.interpolate(flow_n_t, scale_factor=4.0, mode='bilinear', align_corners=False) * 4.0
                    
                    # 扭曲影像
                    warped_p = warp_tensor(t_p_y, flow_p_4k)
                    warped_n = warp_tensor(t_n_y, flow_n_4k)
                    
                    # 切割出 Batch
                    b_up, b_p, b_n, b_gt = get_random_crops(t_up_y, warped_p, warped_n, t_gt_y, num_crops=64, patch_size=256)

                # ======== 模型訓練 ========
                optimizer.zero_grad()
                fused_pred, weights = model(b_up, b_p, b_n)
                loss = torch.mean(torch.sqrt((fused_pred - b_gt)**2 + eps))
                
                loss.backward()
                optimizer.step()
                
                task_loss += loss.item()
                pbar.set_postfix({'L1': f"{loss.item():.4f}"})
                
            print(f"   └─ 平均 Loss: {task_loss/frameCount:.5f}")
            # 清理快取避免多部影片切換時記憶體破碎
            torch.cuda.empty_cache()

        # 【新增】每個 Epoch 結束時，降低學習率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"📉 學習率已調整為: {current_lr:.6f}")

        # 每 Epoch 結束存檔為 joint_fusion_net.pth
        torch.save(model.state_dict(), "joint_fusion_net_test7.pth")
        print(f"💾 Epoch {epoch+1} 模型已儲存為 joint_fusion_net_test7.pth\n")

if __name__ == '__main__':
    main()