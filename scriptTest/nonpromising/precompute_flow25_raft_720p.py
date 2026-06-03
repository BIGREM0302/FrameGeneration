import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from yuvProc import getOneFrame
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading RAFT Flow Model (720p Optimization)...")
raft_model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=True).to(device).eval()

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

def get_raft_flow(img1_small, img2_small):
    # 對齊 RAFT 需要的尺寸 (1280x720 本身就是 8 的倍數，不需 padding)
    i1_rgb = img1_small.repeat(1, 3, 1, 1) * 2.0 - 1.0
    i2_rgb = img2_small.repeat(1, 3, 1, 1) * 2.0 - 1.0
    with torch.no_grad():
        flow = raft_model(i1_rgb, i2_rgb)[-1]
    return flow

def compute_epipolar_cpu(flow, H=720, W=1280):
    step_y, step_x = max(1, H // 32), max(1, W // 32)
    pts1, pts2 = [], []
    for y in range(0, H, step_y):
        for x in range(0, W, step_x):
            pts1.append([x, y])
            pts2.append([x + flow[y, x, 0], y + flow[y, x, 1]])
    F_mat, _ = cv2.findFundamentalMat(np.array(pts1, dtype=np.float32), np.array(pts2, dtype=np.float32), cv2.FM_RANSAC, 1.0, 0.99)
    if F_mat is not None and F_mat.shape == (3, 3):
        y_idx, x_idx = np.indices((H, W))
        pts1_all = np.stack((x_idx, y_idx, np.ones_like(x_idx)), axis=-1).reshape(-1, 3)
        pts2_all = np.stack((x_idx + flow[..., 0], y_idx + flow[..., 1], np.ones_like(x_idx)), axis=-1).reshape(-1, 3)
        lines2 = np.dot(pts1_all, F_mat.T)
        num = np.abs(np.sum(pts2_all * lines2, axis=1))
        den = np.sqrt(lines2[:, 0]**2 + lines2[:, 1]**2) + 1e-6
        return np.tanh((num / den).reshape(H, W) / 4.0)[..., np.newaxis]
    return np.zeros((H, W, 1), dtype=np.float32)

def main():
    train_tasks = [
        {'name': 'Zombie', 'qp': 27, 'base': '../bitstream/base/odd_ZombieClimbing2_27_0_4.layer0.yuv', 'even': '../bitstream/enhance/even_ZombieClimbing2_27_0_4.layer1.yuv'},
        {'name': 'Zombie', 'qp': 32, 'base': '../bitstream/base/odd_ZombieClimbing2_32_0_4.layer0.yuv', 'even': '../bitstream/enhance/even_ZombieClimbing2_32_0_4.layer1.yuv'},
        {'name': 'Zombie', 'qp': 37, 'base': '../bitstream/base/odd_ZombieClimbing2_37_0_4.layer0.yuv', 'even': '../bitstream/enhance/even_ZombieClimbing2_37_0_4.layer1.yuv'},
        {'name': 'Zombie', 'qp': 42, 'base': '../bitstream/base/odd_ZombieClimbing2_42_0_4.layer0.yuv', 'even': '../bitstream/enhance/even_ZombieClimbing2_42_0_4.layer1.yuv'},
        {'name': 'AMS05', 'qp': 27, 'base': '../bitstream/base/odd_H2_H3_AMS05_27_0_5.layer0.yuv', 'even': '../bitstream/enhance/even_H2_H3_AMS05_27_0_5.layer1.yuv'},
        {'name': 'AMS05', 'qp': 32, 'base': '../bitstream/base/odd_H2_H3_AMS05_32_0_5.layer0.yuv', 'even': '../bitstream/enhance/even_H2_H3_AMS05_32_0_5.layer1.yuv'}, 
        {'name': 'AMS05', 'qp': 37, 'base': '../bitstream/base/odd_H2_H3_AMS05_37_0_5.layer0.yuv', 'even': '../bitstream/enhance/even_H2_H3_AMS05_37_0_5.layer1.yuv'},
        {'name': 'AMS05', 'qp': 42, 'base': '../bitstream/base/odd_H2_H3_AMS05_42_0_5.layer0.yuv', 'even': '../bitstream/enhance/even_H2_H3_AMS05_42_0_5.layer1.yuv'},
        {'name': 'WalkInPark', 'qp': 27, 'base': '../bitstream/base/odd_H2_WalkInPark_27_0_4.layer0.yuv', 'even': '../bitstream/enhance/even_H2_WalkInPark_27_0_4.layer1.yuv'},
        {'name': 'WalkInPark', 'qp': 32, 'base': '../bitstream/base/odd_H2_WalkInPark_32_0_4.layer0.yuv', 'even': '../bitstream/enhance/even_H2_WalkInPark_32_0_4.layer1.yuv'},
        {'name': 'WalkInPark', 'qp': 37, 'base': '../bitstream/base/odd_H2_WalkInPark_37_0_4.layer0.yuv', 'even': '../bitstream/enhance/even_H2_WalkInPark_37_0_4.layer1.yuv'},
        {'name': 'WalkInPark', 'qp': 42, 'base': '../bitstream/base/odd_H2_WalkInPark_42_0_4.layer0.yuv', 'even': '../bitstream/enhance/even_H2_WalkInPark_42_0_4.layer1.yuv'},
        {'name': 'Procession', 'qp': 25, 'base': '../bitstream/base/odd_Procession_25_0_4.layer0.yuv', 'even': '../bitstream/enhance/even_Procession_25_0_4.layer1.yuv'},
        {'name': 'Procession', 'qp': 30, 'base': '../bitstream/base/odd_Procession_30_0_4.layer0.yuv', 'even': '../bitstream/enhance/even_Procession_30_0_4.layer1.yuv'},  
        {'name': 'Procession', 'qp': 35, 'base': '../bitstream/base/odd_Procession_35_0_4.layer0.yuv', 'even': '../bitstream/enhance/even_Procession_35_0_4.layer1.yuv'},
        {'name': 'Procession', 'qp': 40, 'base': '../bitstream/base/odd_Procession_40_0_4.layer0.yuv', 'even': '../bitstream/enhance/even_Procession_40_0_4.layer1.yuv'},
    ]
    out_dir = "./lightweight_precomputed_flow_raft_720p"
    os.makedirs(out_dir, exist_ok=True)
    
    for task in train_tasks:
        if not os.path.exists(task['base']): continue
        frameCount = (os.path.getsize(task['base']) // 6220800) - 1
        task_folder = os.path.join(out_dir, f"{task['name']}_qp{task['qp']}")
        os.makedirs(task_folder, exist_ok=True)
        
        pbar = tqdm(range(frameCount), desc=f"720p RAFT Flow {task['name']}_QP{task['qp']}")
        for t in pbar:
            f_base = getOneFrame(task['base'], 1920, 1080, 2, t)
            f_prev = getOneFrame(task['even'], 3840, 2160, 2, t)
            f_next = getOneFrame(task['even'], 3840, 2160, 2, t+1)
            
            with torch.no_grad():
                t_base_y = torch.tensor(f_base['y'].reshape(1080, 1920), dtype=torch.float32, device=device).view(1,1,1080,1920) / 1023.0
                t_p_y = torch.tensor(f_prev['y'].reshape(2160, 3840), dtype=torch.float32, device=device).view(1,1,2160,3840) / 1023.0
                t_n_y = torch.tensor(f_next['y'].reshape(2160, 3840), dtype=torch.float32, device=device).view(1,1,2160,3840) / 1023.0
                
                t_up_y = gpu_8tap_upscale_y(t_base_y)
                
                # 🌟 1/3 解析度縮放，大小變為 720x1280
                t_up_small = F.interpolate(t_up_y, size=(720, 1280), mode='bilinear', align_corners=False)
                t_p_small = F.interpolate(t_p_y, size=(720, 1280), mode='bilinear', align_corners=False)
                t_n_small = F.interpolate(t_n_y, size=(720, 1280), mode='bilinear', align_corners=False)
                
                # 🌟 僅計算單向光流
                flow_p_t = get_raft_flow(t_up_small, t_p_small)
                flow_n_t = get_raft_flow(t_up_small, t_n_small)
                
                flow_p_np = flow_p_t.squeeze().permute(1, 2, 0).cpu().numpy()
                flow_n_np = flow_n_t.squeeze().permute(1, 2, 0).cpu().numpy()
                
            epi_p = compute_epipolar_cpu(flow_p_np)
            epi_n = compute_epipolar_cpu(flow_n_np)
            
            flow_dict = {
                'fp': flow_p_t.cpu(), 
                'fn': flow_n_t.cpu(),
                'ep': torch.from_numpy(epi_p).permute(2, 0, 1).unsqueeze(0),
                'en': torch.from_numpy(epi_n).permute(2, 0, 1).unsqueeze(0)
            }
            torch.save(flow_dict, os.path.join(task_folder, f"frame_{t}.pt"))

if __name__ == '__main__':
    main()