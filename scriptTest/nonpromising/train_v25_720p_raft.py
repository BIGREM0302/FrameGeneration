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

FUSION_MODEL_PATH = "joint_fusion_net_v25_720p.pth"
REFINE_MODEL_PATH = "flow_refine_net_v25_720p.pth"
FUSION_MODEL_BEST_PATH = "joint_fusion_net_v25_720p_best.pth"
REFINE_MODEL_BEST_PATH = "flow_refine_net_v25_720p_best.pth"
PRECOMPUTED_DIR = "./lightweight_precomputed_flow_raft_720p"
LOSS_PLOT_PATH = "loss_curve_v25_720p.png"

# =========================================================================
# 0. Loss 曲線與 TV Loss
# =========================================================================
def draw_loss_curve_cv2(epochs, losses, save_path):
    if len(losses) < 2: return
    W, H = 800, 600
    margin_x, margin_y = 100, 60
    img = np.ones((H, W, 3), dtype=np.uint8) * 255
    max_loss, min_loss = max(losses), min(losses)
    loss_range = max_loss - min_loss if max_loss > min_loss else 1.0
    cv2.line(img, (margin_x, H - margin_y), (W - 40, H - margin_y), (0, 0, 0), 2)
    cv2.line(img, (margin_x, H - margin_y), (margin_x, margin_y), (0, 0, 0), 2)
    pts = []
    for i, (ep, loss) in enumerate(zip(epochs, losses)):
        x = int(margin_x + (W - margin_x - 60) * (i / (len(epochs) - 1)))
        pad = loss_range * 0.05
        mapped_loss = (loss - (min_loss - pad)) / (loss_range + 2 * pad)
        y = int((H - margin_y) - (H - 2 * margin_y) * mapped_loss)
        pts.append((x, y))
    for i in range(1, len(pts)):
        cv2.line(img, pts[i-1], pts[i], (255, 0, 0), 2)
        cv2.circle(img, pts[i], 4, (0, 0, 255), -1)
    cv2.circle(img, pts[0], 4, (0, 0, 255), -1)
    cv2.imwrite(save_path, img)

def get_tv_loss(weights):
    tv_h = torch.mean(torch.abs(weights[:, :, 1:, :] - weights[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(weights[:, :, :, 1:] - weights[:, :, :, :-1]))
    return tv_h + tv_w

# =========================================================================
# 1. 架構 (720p 優化)
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
        # 🌟 輸入減少了 consistency map，改為 7 通道
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
        
        # 不再串接 flow_consistency
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
# 幾何輔助與 Augmentation
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

def warp_tensor(t_img, flow_4k):
    h, w = t_img.shape[2], t_img.shape[3]
    mesh_y, mesh_x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
    grid_x = mesh_x.float() + flow_4k[:, 0, :, :]
    grid_y = mesh_y.float() + flow_4k[:, 1, :, :]
    norm_x = 2.0 * grid_x / (w - 1) - 1.0
    norm_y = 2.0 * grid_y / (h - 1) - 1.0
    return F.grid_sample(t_img, torch.stack((norm_x, norm_y), dim=-1), mode='bicubic', padding_mode='reflection', align_corners=True)

def get_random_crops(t_up, t_p, t_n, t_gt, t_epi_p, t_epi_n, num_crops=24, patch_size=384):
    _, _, H, W = t_up.shape
    up_crops, p_crops, n_crops, gt_crops, epi_p_crops, epi_n_crops = [], [], [], [], [], []
    for _ in range(num_crops):
        h = random.randint(0, H - patch_size)
        w = random.randint(0, W - patch_size)
        up_crops.append(t_up[:, :, h:h+patch_size, w:w+patch_size])
        p_crops.append(t_p[:, :, h:h+patch_size, w:w+patch_size])
        n_crops.append(t_n[:, :, h:h+patch_size, w:w+patch_size])
        gt_crops.append(t_gt[:, :, h:h+patch_size, w:w+patch_size])
        epi_p_crops.append(t_epi_p[:, :, h:h+patch_size, w:w+patch_size])
        epi_n_crops.append(t_epi_n[:, :, h:h+patch_size, w:w+patch_size])
    return (torch.cat(up_crops), torch.cat(p_crops), torch.cat(n_crops), 
            torch.cat(gt_crops), torch.cat(epi_p_crops), torch.cat(epi_n_crops))

def apply_augmentations(b_up, b_p, b_n, b_gt, b_epi_p, b_epi_n):
    if random.random() < 0.5:
        b_up = torch.flip(b_up, [3])
        b_p, b_n, b_gt = torch.flip(b_p, [3]), torch.flip(b_n, [3]), torch.flip(b_gt, [3])
        b_epi_p, b_epi_n = torch.flip(b_epi_p, [3]), torch.flip(b_epi_n, [3])
    if random.random() < 0.5:
        b_up = torch.flip(b_up, [2])
        b_p, b_n, b_gt = torch.flip(b_p, [2]), torch.flip(b_n, [2]), torch.flip(b_gt, [2])
        b_epi_p, b_epi_n = torch.flip(b_epi_p, [2]), torch.flip(b_epi_n, [2])
    if random.random() < 0.5:
        b_p, b_n = b_n, b_p
        b_epi_p, b_epi_n = b_epi_n, b_epi_p
    return b_up, b_p, b_n, b_gt, b_epi_p, b_epi_n

def get_flow_smoothness_loss(flow, img):
    flow_grad_x = torch.abs(flow[:, :, :, :-1] - flow[:, :, :, 1:])
    flow_grad_y = torch.abs(flow[:, :, :-1, :] - flow[:, :, 1:, :])
    img_grad_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), dim=1, keepdim=True)
    img_grad_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), dim=1, keepdim=True)
    weight_x = torch.exp(-img_grad_x * 10.0)
    weight_y = torch.exp(-img_grad_y * 10.0)
    return torch.mean(flow_grad_x * weight_x) + torch.mean(flow_grad_y * weight_y)

def get_laplacian_map(tensor_img): return F.conv2d(F.pad(tensor_img, (1, 1, 1, 1), mode='reflect'), LAPLACIAN_KERNEL)

# =========================================================================
# 3. 訓練迴圈
# =========================================================================
def main():
    train_tasks = [
        {'name': 'Zombie', 'qp': 27, 'base': '../bitstream/base/odd_ZombieClimbing2_27_0_4.layer0.yuv', 'gt': '../orgYUV/odd_Zombie-Climbing2_3840x2160_24fps_10bit_420.yuv'},
        {'name': 'Zombie', 'qp': 32, 'base': '../bitstream/base/odd_ZombieClimbing2_32_0_4.layer0.yuv', 'gt': '../orgYUV/odd_Zombie-Climbing2_3840x2160_24fps_10bit_420.yuv'},
        {'name': 'Zombie', 'qp': 37, 'base': '../bitstream/base/odd_ZombieClimbing2_37_0_4.layer0.yuv', 'gt': '../orgYUV/odd_Zombie-Climbing2_3840x2160_24fps_10bit_420.yuv'},
        {'name': 'Zombie', 'qp': 42, 'base': '../bitstream/base/odd_ZombieClimbing2_42_0_4.layer0.yuv', 'gt': '../orgYUV/odd_Zombie-Climbing2_3840x2160_24fps_10bit_420.yuv'},
        {'name': 'AMS05', 'qp': 27, 'base': '../bitstream/base/odd_H2_H3_AMS05_27_0_5.layer0.yuv', 'gt': '../orgYUV/odd_H2_H3_AMS05_3840x2160_10bit_420_HLG.yuv'},
        {'name': 'AMS05', 'qp': 37, 'base': '../bitstream/base/odd_H2_H3_AMS05_37_0_5.layer0.yuv', 'gt': '../orgYUV/odd_H2_H3_AMS05_3840x2160_10bit_420_HLG.yuv'},
        {'name': 'WalkInPark', 'qp': 27, 'base': '../bitstream/base/odd_H2_WalkInPark_27_0_4.layer0.yuv', 'gt': '../orgYUV/odd_H2_WalkInPark_3840x2160_10_60fps_HLG.yuv'},
        {'name': 'WalkInPark', 'qp': 37, 'base': '../bitstream/base/odd_H2_WalkInPark_37_0_4.layer0.yuv', 'gt': '../orgYUV/odd_H2_WalkInPark_3840x2160_10_60fps_HLG.yuv'},
        {'name': 'Procession', 'qp': 25, 'base': '../bitstream/base/odd_Procession_25_0_4.layer0.yuv', 'gt': '../orgYUV/odd_Procession_3840x2160_60fps_10bit_420.yuv'},  
        {'name': 'Procession', 'qp': 35, 'base': '../bitstream/base/odd_Procession_35_0_4.layer0.yuv', 'gt': '../orgYUV/odd_Procession_3840x2160_60fps_10bit_420.yuv'},
        {'name': 'Procession', 'qp': 40, 'base': '../bitstream/base/odd_Procession_40_0_4.layer0.yuv', 'gt': '../orgYUV/odd_Procession_3840x2160_60fps_10bit_420.yuv'}
    ]
    
    width_base, height_base = 1920, 1080
    bytesPerPel = 2
    byteN_base = int((width_base * height_base * 1.5) * bytesPerPel)

    fusion_model = LiteMicroFusionNetV26().to(device)
    refine_model = LiteFlowRefineNet().to(device)
    
    eps = 1e-6
    optimizer = optim.AdamW(list(fusion_model.parameters()) + list(refine_model.parameters()), lr=1e-3, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scaler = torch.amp.GradScaler('cuda')

    best_loss = float('inf')
    history_epochs = []
    history_losses = []

    for epoch in range(40):
        fusion_model.train()
        refine_model.train()
        random.shuffle(train_tasks)
        epoch_loss_sum = 0.0
        epoch_batches = 0
        
        for task in train_tasks:
            if not os.path.exists(task['base']): continue
            frameCount = (os.path.getsize(task['base']) // byteN_base) - 1
            task_flow_dir = os.path.join(PRECOMPUTED_DIR, f"{task['name']}_qp{task['qp']}")
            
            pbar = tqdm(range(frameCount), desc=f"Epoch {epoch+1}/40 | {task['name']}(QP{task['qp']})", unit="frame")
            for t in pbar:
                prev_file_path = task['base'].replace('base', 'enhance').replace('odd_', 'even_').replace('.layer0.yuv', '.layer1.yuv')
                f_base = getOneFrame(task['base'], width_base, height_base, bytesPerPel, t)
                f_prev = getOneFrame(prev_file_path, 3840, 2160, bytesPerPel, t)
                f_next = getOneFrame(prev_file_path, 3840, 2160, bytesPerPel, t+1)
                f_gt   = getOneFrame(task['gt'], 3840, 2160, bytesPerPel, t)
                
                with torch.no_grad():
                    t_base_y = torch.tensor(f_base['y'].reshape(1080, 1920), dtype=torch.float32, device=device).view(1,1,1080,1920) / 1023.0
                    t_p_y = torch.tensor(f_prev['y'].reshape(2160, 3840), dtype=torch.float32, device=device).view(1,1,2160,3840) / 1023.0
                    t_n_y = torch.tensor(f_next['y'].reshape(2160, 3840), dtype=torch.float32, device=device).view(1,1,2160,3840) / 1023.0
                    t_gt_y = torch.tensor(f_gt['y'].reshape(2160, 3840), dtype=torch.float32, device=device).view(1,1,2160,3840) / 1023.0
                    t_up_y = gpu_8tap_upscale_y(t_base_y)
                    
                    flow_data = torch.load(os.path.join(task_flow_dir, f"frame_{t}.pt"), map_location=device, weights_only=True)
                    flow_p_t = flow_data['fp'].float().to(device)
                    flow_n_t = flow_data['fn'].float().to(device)
                    epi_p_cv_t = flow_data['ep'].float().to(device)
                    epi_n_cv_t = flow_data['en'].float().to(device)

                optimizer.zero_grad()
                
                with torch.amp.autocast('cuda'):
                    # 🌟 輸入對齊 1/3 解析度 (720x1280)
                    t_up_small = F.interpolate(t_up_y, size=(720, 1280), mode='bilinear', align_corners=False)
                    t_p_small = F.interpolate(t_p_y, size=(720, 1280), mode='bilinear', align_corners=False)
                    t_n_small = F.interpolate(t_n_y, size=(720, 1280), mode='bilinear', align_corners=False)
                    
                    flow_p_t_refined = refine_model(t_up_small, t_p_small, flow_p_t)
                    flow_n_t_refined = refine_model(t_up_small, t_n_small, flow_n_t)
                    
                    # 取消了 back flow 和 consistency 的計算
                    epi_p_4k = F.interpolate(epi_p_cv_t, size=(2160, 3840), mode='bilinear', align_corners=False)
                    epi_n_4k = F.interpolate(epi_n_cv_t, size=(2160, 3840), mode='bilinear', align_corners=False)
                    
                    # 放大光流至 4K，注意尺度為 3 倍
                    flow_p_4k = F.interpolate(flow_p_t_refined, size=(2160, 3840), mode='bilinear', align_corners=False) * 3.0
                    flow_n_4k = F.interpolate(flow_n_t_refined, size=(2160, 3840), mode='bilinear', align_corners=False) * 3.0
                    
                    warped_p = warp_tensor(t_p_y, flow_p_4k)
                    warped_n = warp_tensor(t_n_y, flow_n_4k)
                    
                    b_up, b_p, b_n, b_gt, b_epi_p, b_epi_n = get_random_crops(
                        t_up_y, warped_p, warped_n, t_gt_y, epi_p_4k, epi_n_4k, num_crops=24, patch_size=384
                    )
                    
                    b_up, b_p, b_n, b_gt, b_epi_p, b_epi_n = apply_augmentations(b_up, b_p, b_n, b_gt, b_epi_p, b_epi_n)
                    
                    noise = torch.randn_like(b_up) * 0.005
                    b_up_noisy = torch.clamp(b_up + noise, 0.0, 1.0)
                    b_lap = get_laplacian_map(b_up_noisy)

                    # 🌟 餵給 FusionNet (7個輸入，少了 cons)
                    fused_pred, weights = fusion_model(b_up_noisy, b_p, b_n, b_lap, b_epi_p, b_epi_n)
                    
                    pred_lap_map = get_laplacian_map(fused_pred)
                    gt_lap_map = get_laplacian_map(b_gt)
                    loss_edge = torch.mean(torch.sqrt((pred_lap_map - gt_lap_map)**2 + eps))
                    
                    loss_warp_direct = torch.mean(torch.sqrt((b_p - b_gt)**2 + eps)) + torch.mean(torch.sqrt((b_n - b_gt)**2 + eps))
                    loss_pixel = torch.mean(torch.sqrt((fused_pred - b_gt)**2 + eps))
                    loss_smooth = get_flow_smoothness_loss(flow_p_t_refined, t_up_small) + get_flow_smoothness_loss(flow_n_t_refined, t_up_small)
                    loss_tv_mask = get_tv_loss(weights)
                    
                    loss = loss_pixel + 0.05 * loss_warp_direct + 0.05 * loss_edge + 0.02 * loss_smooth + 0.01 * loss_tv_mask

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(fusion_model.parameters(), max_norm=2.0)
                torch.nn.utils.clip_grad_norm_(refine_model.parameters(), max_norm=2.0)
                scaler.step(optimizer)
                scaler.update()
                
                epoch_loss_sum += loss.item()
                epoch_batches += 1
                pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'Pix': f"{loss_pixel.item():.4f}"})
                
            torch.cuda.empty_cache()

        scheduler.step()
        avg_loss = epoch_loss_sum / max(1, epoch_batches)
        history_epochs.append(epoch + 1)
        history_losses.append(avg_loss)
        
        print(f"\n[Epoch {epoch+1}/40] Average Loss: {avg_loss:.6f} | Best Loss: {best_loss:.6f}")
        draw_loss_curve_cv2(history_epochs, history_losses, LOSS_PLOT_PATH)

        torch.save(fusion_model.state_dict(), FUSION_MODEL_PATH)
        torch.save(refine_model.state_dict(), REFINE_MODEL_PATH)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(fusion_model.state_dict(), FUSION_MODEL_BEST_PATH)
            torch.save(refine_model.state_dict(), REFINE_MODEL_BEST_PATH)

if __name__ == '__main__':
    main()