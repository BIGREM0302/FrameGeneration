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

FUSION_MODEL_PATH = "luma_fusion_net_test28.pth"
REFINE_MODEL_PATH = "flow_refine_net_test28.pth"
FUSION_MODEL_BEST_PATH = "luma_fusion_net_best28.pth"
REFINE_MODEL_BEST_PATH = "flow_refine_net_best28.pth"
PRECOMPUTED_DIR = "./lightweight_precomputed_flow_pt"
LOSS_PLOT_PATH = "loss_curve28.png"

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

class FlowRefineNet(nn.Module):
    def __init__(self):
        super(FlowRefineNet, self).__init__()
        self.refiner = nn.Sequential(
            nn.Conv2d(5, 64, kernel_size=3, padding=1), nn.InstanceNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.InstanceNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 2, kernel_size=3, padding=1)
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

class HeavyResidualBlock(nn.Module):
    def __init__(self, channels):
        super(HeavyResidualBlock, self).__init__()
        # 🌟 革命性升級：更深的 3 層 ResBlock，只處理 Y 通道
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1), nn.InstanceNorm2d(channels), nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1), nn.InstanceNorm2d(channels), nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1), nn.InstanceNorm2d(channels), nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1) 
        )
        self.gelu = nn.GELU()
    def forward(self, x): return self.gelu(x + self.conv(x))

class LumaHeavyFusionNet(nn.Module):
    def __init__(self):
        super(LumaHeavyFusionNet, self).__init__()
        # 🌟 算力解放：通道數提升至 64 -> 32 -> 96 -> 48
        self.conv1 = nn.Sequential(nn.Conv2d(8, 64, kernel_size=3, padding=1), nn.InstanceNorm2d(64))
        self.se1 = SEBlock(64)
        self.conv2_d1 = nn.Sequential(nn.Conv2d(64, 32, kernel_size=3, padding=1, dilation=1), HeavyResidualBlock(32))
        self.conv2_d5 = nn.Sequential(nn.Conv2d(64, 32, kernel_size=3, padding=5, dilation=5), HeavyResidualBlock(32))
        self.conv2_d9 = nn.Sequential(nn.Conv2d(64, 32, kernel_size=3, padding=9, dilation=9), HeavyResidualBlock(32))
        self.se2 = SEBlock(96)
        self.conv3 = nn.Sequential(nn.Conv2d(96, 48, kernel_size=3, padding=1), nn.InstanceNorm2d(48))
        self.conv4 = nn.Conv2d(48, 4, kernel_size=3, padding=1)
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
        
        raw_logits = out[:, :3, :, :]
        delta_res = torch.tanh(out[:, 3:4, :, :]) * 0.05
        weights = F.softmax(raw_logits * self.temp, dim=1)
        fused_base = (weights[:, 0:1] * t_up) + (weights[:, 1:2] * warped_p) + (weights[:, 2:3] * warped_n)
        
        # 🌟 回傳 fused_base 供 Loss 計算，回傳 weights 供 Chroma 奴役使用
        return torch.clamp(fused_base + delta_res, 0.0, 1.0), weights

W8_H = torch.tensor([-1, 4, -11, 40, 40, -11, 4, -1], dtype=torch.float32, device=device).view(1, 1, 1, 8) / 64.0
W8_V = torch.tensor([-1, 4, -11, 40, 40, -11, 4, -1], dtype=torch.float32, device=device).view(1, 1, 8, 1) / 64.0
LAPLACIAN_KERNEL = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=device).view(1, 1, 3, 3)

def compute_fft_loss(pred, gt):
    # 🌟 頻域感知損失，專注修復高頻紋理
    pred_fft = torch.fft.rfft2(pred, norm="backward")
    gt_fft = torch.fft.rfft2(gt, norm="backward")
    return F.l1_loss(torch.abs(pred_fft), torch.abs(gt_fft))

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

def warp_flow(flow_to_warp, flow_guide):
    h, w = flow_to_warp.shape[2], flow_to_warp.shape[3]
    mesh_y, mesh_x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
    grid_x = mesh_x.float() + flow_guide[:, 0, :, :]
    grid_y = mesh_y.float() + flow_guide[:, 1, :, :]
    norm_x = 2.0 * grid_x / (w - 1) - 1.0
    norm_y = 2.0 * grid_y / (h - 1) - 1.0
    return F.grid_sample(flow_to_warp, torch.stack((norm_x, norm_y), dim=-1), mode='bilinear', padding_mode='zeros', align_corners=True)

def get_random_crops_clean(t_up, t_p, t_n, t_gt, t_consistent, t_epi_p, t_epi_n, num_crops=32, patch_size=384): # A4500可以扛32
    _, _, H, W = t_up.shape
    crops = {k: [] for k in ['up', 'p', 'n', 'gt', 'cons', 'ep', 'en']}
    for _ in range(num_crops):
        h = random.randint(0, H - patch_size)
        w = random.randint(0, W - patch_size)
        up, p, n, gt, cons, ep, en = (t_up[:, :, h:h+patch_size, w:w+patch_size], t_p[:, :, h:h+patch_size, w:w+patch_size], 
                                      t_n[:, :, h:h+patch_size, w:w+patch_size], t_gt[:, :, h:h+patch_size, w:w+patch_size], 
                                      t_consistent[:, :, h:h+patch_size, w:w+patch_size], t_epi_p[:, :, h:h+patch_size, w:w+patch_size], 
                                      t_epi_n[:, :, h:h+patch_size, w:w+patch_size])
        if random.random() > 0.5:
            up, p, n, gt, cons, ep, en = [torch.flip(t, [3]) for t in [up, p, n, gt, cons, ep, en]]
        if random.random() > 0.5:
            up, p, n, gt, cons, ep, en = [torch.flip(t, [2]) for t in [up, p, n, gt, cons, ep, en]]
        crops['up'].append(up); crops['p'].append(p); crops['n'].append(n); crops['gt'].append(gt)
        crops['cons'].append(cons); crops['ep'].append(ep); crops['en'].append(en)
    return (torch.cat(crops['up']), torch.cat(crops['p']), torch.cat(crops['n']), 
            torch.cat(crops['gt']), torch.cat(crops['cons']), torch.cat(crops['ep']), torch.cat(crops['en']))

def get_laplacian_map(tensor_img): return F.conv2d(F.pad(tensor_img, (1, 1, 1, 1), mode='reflect'), LAPLACIAN_KERNEL)

def main():
    #train_tasks = [
    #    {'name': 'Zombie', 'qp': 27, 'base': '../bitstream/base/odd_ZombieClimbing2_27_0_4.layer0.yuv', 'gt': '../orgYUV/odd_Zombie-Climbing2_3840x2160_24fps_10bit_420.yuv'},
    #    {'name': 'Zombie', 'qp': 37, 'base': '../bitstream/base/odd_ZombieClimbing2_37_0_4.layer0.yuv', 'gt': '../orgYUV/odd_Zombie-Climbing2_3840x2160_24fps_10bit_420.yuv'},
    #    {'name': 'AMS05', 'qp': 27, 'base': '../bitstream/base/odd_H2_H3_AMS05_27_0_5.layer0.yuv', 'gt': '../orgYUV/odd_H2_H3_AMS05_3840x2160_10bit_420_HLG.yuv'},
    #    {'name': 'AMS05', 'qp': 37, 'base': '../bitstream/base/odd_H2_H3_AMS05_37_0_5.layer0.yuv', 'gt': '../orgYUV/odd_H2_H3_AMS05_3840x2160_10bit_420_HLG.yuv'},
    #    {'name': 'WalkInPark', 'qp': 27, 'base': '../bitstream/base/odd_H2_WalkInPark_27_0_4.layer0.yuv', 'gt': '../orgYUV/odd_H2_WalkInPark_3840x2160_10_60fps_HLG.yuv'},
    #    {'name': 'Procession', 'qp': 25, 'base': '../bitstream/base/odd_Procession_25_0_4.layer0.yuv', 'gt': '../orgYUV/odd_Procession_3840x2160_60fps_10bit_420.yuv'},  
    #]
    
    train_tasks = [
        {'name': 'Zombie', 'qp': 27, 'base': '../bitstream/base/odd_ZombieClimbing2_27_0_4.layer0.yuv', 'gt': '../orgYUV/odd_Zombie-Climbing2_3840x2160_24fps_10bit_420.yuv'},
        #{'name': 'Zombie', 'qp': 32, 'base': '../bitstream/base/odd_ZombieClimbing2_32_0_4.layer0.yuv', 'gt': '../orgYUV/odd_Zombie-Climbing2_3840x2160_24fps_10bit_420.yuv'},
        {'name': 'Zombie', 'qp': 37, 'base': '../bitstream/base/odd_ZombieClimbing2_37_0_4.layer0.yuv', 'gt': '../orgYUV/odd_Zombie-Climbing2_3840x2160_24fps_10bit_420.yuv'},
        {'name': 'Zombie', 'qp': 42, 'base': '../bitstream/base/odd_ZombieClimbing2_42_0_4.layer0.yuv', 'gt': '../orgYUV/odd_Zombie-Climbing2_3840x2160_24fps_10bit_420.yuv'},
        {'name': 'AMS05', 'qp': 27, 'base': '../bitstream/base/odd_H2_H3_AMS05_27_0_5.layer0.yuv', 'gt': '../orgYUV/odd_H2_H3_AMS05_3840x2160_10bit_420_HLG.yuv'},
        {'name': 'AMS05', 'qp': 37, 'base': '../bitstream/base/odd_H2_H3_AMS05_37_0_5.layer0.yuv', 'gt': '../orgYUV/odd_H2_H3_AMS05_3840x2160_10bit_420_HLG.yuv'},
        #{'name': 'AMS05', 'qp': 42, 'base': '../bitstream/base/odd_H2_H3_AMS05_42_0_5.layer0.yuv', 'gt': '../orgYUV/odd_H2_H3_AMS05_3840x2160_10bit_420_HLG.yuv'},
        {'name': 'WalkInPark', 'qp': 27, 'base': '../bitstream/base/odd_H2_WalkInPark_27_0_4.layer0.yuv', 'gt': '../orgYUV/odd_H2_WalkInPark_3840x2160_10_60fps_HLG.yuv'},
        {'name': 'WalkInPark', 'qp': 37, 'base': '../bitstream/base/odd_H2_WalkInPark_37_0_4.layer0.yuv', 'gt': '../orgYUV/odd_H2_WalkInPark_3840x2160_10_60fps_HLG.yuv'},
        {'name': 'Procession', 'qp': 25, 'base': '../bitstream/base/odd_Procession_25_0_4.layer0.yuv', 'gt': '../orgYUV/odd_Procession_3840x2160_60fps_10bit_420.yuv'},  
        #{'name': 'Procession', 'qp': 35, 'base': '../bitstream/base/odd_Procession_35_0_4.layer0.yuv', 'gt': '../orgYUV/odd_Procession_3840x2160_60fps_10bit_420.yuv'},
        {'name': 'Procession', 'qp': 40, 'base': '../bitstream/base/odd_Procession_40_0_4.layer0.yuv', 'gt': '../orgYUV/odd_Procession_3840x2160_60fps_10bit_420.yuv'}
    ]

    width_base, height_base = 1920, 1080
    bytesPerPel = 2
    byteN_base = int((width_base * height_base * 1.5) * bytesPerPel)

    fusion_model = LumaHeavyFusionNet().to(device)
    refine_model = FlowRefineNet().to(device)
    
    # 🌟 A4500 的底層加速黑魔法：開啟 torch.compile
    if hasattr(torch, 'compile'):
        try:
            print("🚀 A4500 啟用 torch.compile C++ 底層加速...")
            fusion_model = torch.compile(fusion_model)
            refine_model = torch.compile(refine_model)
        except Exception as e:
            print(f"Compile skipped: {e}")

    eps = 1e-6
    # 🌟 開啟 Fused 優化器，減少 GPU/CPU IO 延遲
    optimizer = optim.AdamW(list(fusion_model.parameters()) + list(refine_model.parameters()), lr=1e-3, weight_decay=1e-4, fused=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)
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
        use_mse_phase = (epoch >= 20)
        
        for task in train_tasks:
            if not os.path.exists(task['base']): continue
            frameCount = (os.path.getsize(task['base']) // byteN_base) - 1
            task_flow_dir = os.path.join(PRECOMPUTED_DIR, f"{task['name']}_qp{task['qp']}")
            
            phase_name = "MSE+FFT" if use_mse_phase else "L1+FFT"
            pbar = tqdm(range(frameCount), desc=f"Ep {epoch+1}/40 [{phase_name}] | {task['name']}(QP{task['qp']})", unit="frame")
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
                    flow_p_t = flow_data['fp'].float()
                    flow_n_t = flow_data['fn'].float()
                    flow_p_back_t = flow_data['fpb'].float()
                    flow_n_back_t = flow_data['fnb'].float()
                    epi_p_cv_t = flow_data['ep'].float()
                    epi_n_cv_t = flow_data['en'].float()

                optimizer.zero_grad()
                
                with torch.amp.autocast('cuda'):
                    t_up_small = F.interpolate(t_up_y, scale_factor=0.25, mode='bilinear', align_corners=False)
                    t_p_small = F.interpolate(t_p_y, scale_factor=0.25, mode='bilinear', align_corners=False)
                    t_n_small = F.interpolate(t_n_y, scale_factor=0.25, mode='bilinear', align_corners=False)
                    
                    flow_p_t_refined = refine_model(t_up_small, t_p_small, flow_p_t)
                    flow_n_t_refined = refine_model(t_up_small, t_n_small, flow_n_t)
                    flow_p_back_refined = refine_model(t_p_small, t_up_small, flow_p_back_t)
                    flow_n_back_refined = refine_model(t_n_small, t_up_small, flow_n_back_t)
                    
                    flow_p_back_warped = warp_flow(flow_p_back_refined, flow_p_t_refined)
                    flow_n_back_warped = warp_flow(flow_n_back_refined, flow_n_t_refined)
                    
                    err_p_gpu = torch.norm(flow_p_t_refined + flow_p_back_warped, dim=1, keepdim=True)
                    err_n_gpu = torch.norm(flow_n_t_refined + flow_n_back_warped, dim=1, keepdim=True)
                    flow_consistency_t = torch.tanh(torch.max(err_p_gpu, err_n_gpu) / 3.0)
                    
                    flow_consistency_4k = F.interpolate(flow_consistency_t, scale_factor=4.0, mode='bilinear', align_corners=False)
                    epi_p_4k = F.interpolate(epi_p_cv_t, scale_factor=4.0, mode='bilinear', align_corners=False)
                    epi_n_4k = F.interpolate(epi_n_cv_t, scale_factor=4.0, mode='bilinear', align_corners=False)
                    
                    flow_p_4k = F.interpolate(flow_p_t_refined, scale_factor=4.0, mode='bilinear', align_corners=False) * 4.0
                    flow_n_4k = F.interpolate(flow_n_t_refined, scale_factor=4.0, mode='bilinear', align_corners=False) * 4.0
                    
                    warped_p = warp_tensor(t_p_y, flow_p_4k)
                    warped_n = warp_tensor(t_n_y, flow_n_4k)
                    
                    # 🌟 現在我們只訓練 Y 通道，最大化 Y-PSNR
                    b_up, b_p, b_n, b_gt, b_cons, b_epi_p, b_epi_n = get_random_crops_clean(
                        t_up_y, warped_p, warped_n, t_gt_y, flow_consistency_4k, epi_p_4k, epi_n_4k, num_crops=32, patch_size=384
                    )
                    b_lap = get_laplacian_map(b_up)

                    fused_pred, _ = fusion_model(b_up, b_p, b_n, b_cons, b_lap, b_epi_p, b_epi_n)
                    
                    if use_mse_phase:
                        loss_pixel = F.mse_loss(fused_pred, b_gt) 
                    else:
                        loss_pixel = torch.mean(torch.sqrt((fused_pred - b_gt)**2 + eps))
                    
                    # 🌟 FFT 頻域損失，強化紋理對齊
                    loss_fft = compute_fft_loss(fused_pred, b_gt)
                    
                    loss = loss_pixel + 0.005 * loss_fft

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(fusion_model.parameters(), max_norm=2.0)
                torch.nn.utils.clip_grad_norm_(refine_model.parameters(), max_norm=2.0)
                scaler.step(optimizer)
                scaler.update()
                
                epoch_loss_sum += loss.item()
                epoch_batches += 1
                pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'FFT': f"{loss_fft.item():.4f}"})
                
            torch.cuda.empty_cache()

        scheduler.step()
        avg_loss = epoch_loss_sum / max(1, epoch_batches)
        history_epochs.append(epoch + 1)
        history_losses.append(avg_loss)
        
        print(f"\n[Epoch {epoch+1}/40] Average Loss: {avg_loss:.6f} | Best Loss: {best_loss:.6f}")
        draw_loss_curve_cv2(history_epochs, history_losses, LOSS_PLOT_PATH)

        # 這裡的 state_dict 如果被 compile 過會帶有前綴，需處理一下
        uncompiled_fusion = fusion_model._orig_mod if hasattr(fusion_model, '_orig_mod') else fusion_model
        uncompiled_refine = refine_model._orig_mod if hasattr(refine_model, '_orig_mod') else refine_model

        torch.save(uncompiled_fusion.state_dict(), FUSION_MODEL_PATH)
        torch.save(uncompiled_refine.state_dict(), REFINE_MODEL_PATH)

        if avg_loss < best_loss:
            best_loss = avg_loss
            print(f"🌟 新的 Best Checkpoint 產生！")
            torch.save(uncompiled_fusion.state_dict(), FUSION_MODEL_BEST_PATH)
            torch.save(uncompiled_refine.state_dict(), REFINE_MODEL_BEST_PATH)

if __name__ == '__main__':
    main()