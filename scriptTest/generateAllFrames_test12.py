import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from yuvProc import getOneFrame

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# ==========================================
# 1. 網路架構 (完美對齊訓練端)
# ==========================================
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

ml_model = MicroFusionNet().to(device)
if os.path.exists("joint_fusion_net_test12.pth"):
    ml_model.load_state_dict(torch.load("joint_fusion_net_test12.pth", map_location=device))
    print("✅ 成功載入神經網路權重: joint_fusion_net_test12.pth")
else:
    print("⚠️ 找不到權重檔，將使用隨機權重測試")
ml_model.eval()

# ==========================================
# 輔助功能與升頻器
# ==========================================
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

def ml_fuse_channel(t_up, t_p, t_n, flow_p, flow_n, flow_consistency, laplacian_guide):
    h, w = t_up.shape[2], t_up.shape[3]
    mesh_y, mesh_x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
    
    # Warp
    grid_x_p = mesh_x.float() + flow_p[0, 0, :, :]
    grid_y_p = mesh_y.float() + flow_p[0, 1, :, :]
    norm_x_p = 2.0 * grid_x_p / (w - 1) - 1.0
    norm_y_p = 2.0 * grid_y_p / (h - 1) - 1.0
    warped_p = F.grid_sample(t_p, torch.stack((norm_x_p, norm_y_p), dim=-1).unsqueeze(0), 
                             mode='bicubic', padding_mode='reflection', align_corners=True)
    
    grid_x_n = mesh_x.float() + flow_n[0, 0, :, :]
    grid_y_n = mesh_y.float() + flow_n[0, 1, :, :]
    norm_x_n = 2.0 * grid_x_n / (w - 1) - 1.0
    norm_y_n = 2.0 * grid_y_n / (h - 1) - 1.0
    warped_n = F.grid_sample(t_n, torch.stack((norm_x_n, norm_y_n), dim=-1).unsqueeze(0), 
                             mode='bicubic', padding_mode='reflection', align_corners=True)
    
    # 尺寸保護：彩度通道的引導圖需要從 4K 下採樣到 1080p
    if flow_consistency.shape[2] != h:
        flow_consistency = F.interpolate(flow_consistency, size=(h, w), mode='bilinear', align_corners=False)

    # 【新增尺寸保護二】：如果拉普拉斯引導圖與目前融合通道尺寸不同 (例如融合 1080p 的 UV 時)
    # 將其縮放到與當前通道（t_up）完全一致的解析度，防止 cat 的時候炸維度！
    if laplacian_guide.shape[2] != h or laplacian_guide.shape[3] != w:
        laplacian_guide = F.interpolate(laplacian_guide, size=(h, w), mode='bilinear', align_corners=False)

    fused, _ = ml_model(t_up, warped_p, warped_n, flow_consistency, laplacian_guide)
    return fused

# 全域定義拉普拉斯卷積核
LAPLACIAN_KERNEL = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=device).view(1, 1, 3, 3)

def get_laplacian_map(tensor_img):
    """ 在 GPU 端極速提取純粹的高頻邊緣紋理 """
    return F.conv2d(F.pad(tensor_img, (1, 1, 1, 1), mode='reflect'), LAPLACIAN_KERNEL)

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
            print(f"\n==================== 開始處理影片: {video_name} ====================")
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
                        
                        def get_small_flow_img(tensor_4k):
                            t_small = F.interpolate(tensor_4k, scale_factor=0.25, mode='bilinear', align_corners=False)
                            return torch.clamp(t_small.squeeze() * 255.0, 0, 255).byte().cpu().numpy()

                        up_8b = get_small_flow_img(t_up_y)
                        p_8b = get_small_flow_img(t_p_y)
                        n_8b = get_small_flow_img(t_n_y)
                        
                        # 雙向光流
                        flow_p_small = cv2.calcOpticalFlowFarneback(up_8b, p_8b, None, 0.5, 5, 25, 3, 5, 1.2, 0)
                        flow_n_small = cv2.calcOpticalFlowFarneback(up_8b, n_8b, None, 0.5, 5, 25, 3, 5, 1.2, 0)
                        flow_p_back = cv2.calcOpticalFlowFarneback(p_8b, up_8b, None, 0.5, 5, 25, 3, 5, 1.2, 0)
                        flow_n_back = cv2.calcOpticalFlowFarneback(n_8b, up_8b, None, 0.5, 5, 25, 3, 5, 1.2, 0)
                        
                        err_p_small = np.linalg.norm(flow_p_small + flow_p_back, axis=2, keepdims=True)
                        err_n_small = np.linalg.norm(flow_n_small + flow_n_back, axis=2, keepdims=True)
                        err_consistent_small = np.maximum(err_p_small, err_n_small)

                        flow_p_t_540 = torch.tensor(flow_p_small, dtype=torch.float32, device=device).permute(2,0,1).unsqueeze(0)
                        flow_n_t_540 = torch.tensor(flow_n_small, dtype=torch.float32, device=device).permute(2,0,1).unsqueeze(0)
                        flow_consistency_t = torch.tensor(err_consistent_small, dtype=torch.float32, device=device).permute(2,0,1).unsqueeze(0)
                        
                        flow_p_chroma = F.interpolate(flow_p_t_540, scale_factor=2.0, mode='bilinear', align_corners=False) * 2.0
                        flow_n_chroma = F.interpolate(flow_n_t_540, scale_factor=2.0, mode='bilinear', align_corners=False) * 2.0
                        flow_p_4k = F.interpolate(flow_p_t_540, scale_factor=4.0, mode='bilinear', align_corners=False) * 4.0
                        flow_n_4k = F.interpolate(flow_n_t_540, scale_factor=4.0, mode='bilinear', align_corners=False) * 4.0
                        flow_consistency_4k = F.interpolate(flow_consistency_t, scale_factor=4.0, mode='bilinear', align_corners=False)

                        # 【新增】計算 4K 基礎放大影格的拉普拉斯地圖
                        t_lap_4k = get_laplacian_map(t_up_y)

                        # 交叉融合
                        fused_y = ml_fuse_channel(t_up_y, t_p_y, t_n_y, flow_p_4k, flow_n_4k, flow_consistency_4k, t_lap_4k)
                        fused_u = ml_fuse_channel(t_up_u, t_p_u, t_n_u, flow_p_chroma, flow_n_chroma, flow_consistency_4k, t_lap_4k)
                        fused_v = ml_fuse_channel(t_up_v, t_p_v, t_n_v, flow_p_chroma, flow_n_chroma, flow_consistency_4k, t_lap_4k)

                        f_out.write((fused_y.squeeze().cpu().numpy() * 1023.0).clip(0, 1023).astype('<u2').tobytes())
                        f_out.write((fused_u.squeeze().cpu().numpy() * 1023.0).clip(0, 1023).astype('<u2').tobytes())
                        f_out.write((fused_v.squeeze().cpu().numpy() * 1023.0).clip(0, 1023).astype('<u2').tobytes())
                        
                print(f"\n    [成功] 儲存至: {out_file}")
            torch.cuda.empty_cache()

if __name__ == '__main__':
    main()