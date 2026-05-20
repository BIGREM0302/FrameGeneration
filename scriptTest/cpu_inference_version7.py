import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from yuvProc import getOneFrame

# CPU-only version: force PyTorch to run on CPU.
device = torch.device("cpu")
torch.set_num_threads(os.cpu_count() or 1)
try:
    cv2.setNumThreads(os.cpu_count() or 1)
except Exception:
    pass
print(f"Using device: {device}, torch threads: {torch.get_num_threads()}")

# CPU 上 CNN 會很慢；若想改用傳統權重融合，可把這個改成 False。
USE_ML_FUSION = True

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

# 建立模型並載入權重 (設定為推論模式 eval)
ml_model = MicroFusionNet().to(device)
if os.path.exists("joint_fusion_net_test7.pth"):
    ml_model.load_state_dict(torch.load("joint_fusion_net_test7.pth", map_location=device))
    print("✅ 成功載入神經網路權重:joint_fusion_net_test7.pth")
else:
    print("⚠️ 找不到權重檔，將使用未訓練的隨機權重進行測試")
ml_model.eval() # 關閉 Dropout/BatchNorm 等訓練機制

# ==========================================
# 1. GPU 卷積升頻器 (徹底消滅 CPU Numpy 切片)
# ==========================================
# ========== 【優化 1】將濾波器權重宣告為全域變數，常駐 GPU ==========
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

# ==========================================
# 2. 輔助工具：降採樣供 CPU OpenCV 算光流
# ==========================================
def get_8b_small(tensor_0to1):
    """ 將 GPU 的 4K Tensor 縮小一半並轉為 uint8 給 Farneback 使用 """
    small_t = F.interpolate(tensor_0to1, scale_factor=0.5, mode='bilinear', align_corners=False)
    # 直接在 GPU 乘 255 後轉型，再拉回 CPU，速度極快
    return torch.clamp(small_t.squeeze() * 255.0, 0, 255).byte().cpu().numpy()

# CPU-friendly cache: avoid rebuilding full-resolution mesh grids every frame.
_MESH_CACHE = {}

def get_meshgrid(h, w):
    key = (h, w)
    if key not in _MESH_CACHE:
        mesh_y, mesh_x = torch.meshgrid(
            torch.arange(h, device=device),
            torch.arange(w, device=device),
            indexing='ij'
        )
        _MESH_CACHE[key] = (mesh_y, mesh_x)
    return _MESH_CACHE[key]

# ==========================================
# 3. 核心融合：Patch-based 區域加權平均
# ==========================================
def fuse_channel(t_up, t_p, t_n, flow_p, flow_n, sigma_val):
    h, w = t_up.shape[2], t_up.shape[3]
    
    mesh_y, mesh_x = get_meshgrid(h, w)
    
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
    
    # 【退回 L1 Norm】絕對誤差對光流的小瑕疵有極好的包容力
    diff_p = F.avg_pool2d(torch.abs(warped_p - t_up), kernel_size=5, stride=1, padding=2)
    diff_n = F.avg_pool2d(torch.abs(warped_n - t_up), kernel_size=5, stride=1, padding=2)
    
    # 計算權重
    w_p = torch.exp(-diff_p / sigma_val)
    w_n = torch.exp(-diff_n / sigma_val)
    
    # 固定保底權重 0.3，不要讓 Base Layer 喧賓奪主
    w_base = torch.full_like(w_p, 0.3)
    
    fused = (w_p * warped_p + w_n * warped_n + w_base * t_up) / (w_p + w_n + w_base)
    return fused

# ==========================================
# 3.5 ML 核心融合：神經網路決策
# ==========================================
def ml_fuse_channel(t_up, t_p, t_n, flow_p, flow_n):
    h, w = t_up.shape[2], t_up.shape[3]
    
    mesh_y, mesh_x = get_meshgrid(h, w)
    
    # 影像扭曲 (Warp)
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
    
    # 交給神經網路預測權重並融合！
    # 因為我們有 with torch.no_grad()，這裡不會消耗額外的記憶體來存梯度
    fused, _ = ml_model(t_up, warped_p, warped_n)
    
    return fused

def main():
    width_base, height_base = 1920, 1080
    width_el, height_el = 3840, 2160
    bytesPerPel = 2
    byteN_base = int((width_base * height_base * 1.5) * bytesPerPel)
    
    video_configs = {
        'AMS05': {
            'qps': [27, 32, 37, 42],
            'base': '../bitstream/base/odd_H2_H3_AMS05_{qp}_0_5.layer0.yuv',
            'even': '../bitstream/enhance/even_H2_H3_AMS05_{qp}_0_5.layer1.yuv',
            'out': '../bitstream/generated/odd_H2_H3_AMS05_{qp}_0_5_gen.layer0.yuv'
            },
        'WalkInPark': {
            'qps': [27, 32, 37, 42],
            'base': '../bitstream/base/odd_H2_WalkInPark_{qp}_0_4.layer0.yuv',
            'even': '../bitstream/enhance/even_H2_WalkInPark_{qp}_0_4.layer1.yuv',
            'out': '../bitstream/generated/odd_H2_WalkInPark_{qp}_0_4_gen.layer0.yuv'
            },
        'Procession': {
            'qps': [25, 30, 35, 40],
            'base': '../bitstream/base/odd_Procession_{qp}_0_4.layer0.yuv',
            'even': '../bitstream/enhance/even_Procession_{qp}_0_4.layer1.yuv',
            'out': '../bitstream/generated/odd_Procession_{qp}_0_4_gen.layer0.yuv'
            },
        'Zombie': {
            'qps': [27, 32, 37, 42],
            'base': '../bitstream/base/odd_ZombieClimbing2_{qp}_0_4.layer0.yuv',
            'even': '../bitstream/enhance/even_ZombieClimbing2_{qp}_0_4.layer1.yuv',
            'out': '../bitstream/generated/odd_ZombieClimbing2_{qp}_0_4_gen.layer0.yuv'
            }
    }
    
    out_dir = '../bitstream/generated'
    os.makedirs(out_dir, exist_ok=True)
    
    # 啟動 no_grad 節省大量記憶體並提升推論速度
    with torch.inference_mode():
        for video_name, config in video_configs.items():
            print(f"\n==================== 開始處理影片: {video_name} ====================")
            
            for qp in config['qps']:
                base_file = config['base'].format(qp=qp)
                even_file = config['even'].format(qp=qp)
                out_file = config['out'].format(qp=qp)
                
                if not os.path.exists(base_file) or not os.path.exists(even_file): continue
                
                frameCount = (os.path.getsize(base_file) // byteN_base) - 1
                # print(f"--> 正在處理 QP {qp} (預計生成 {frameCount} 幀)...")
                
                with open(out_file, 'wb') as f_out:
                    for t in tqdm(range(frameCount), desc=f"Video: {video_name} QP: {qp}", unit="frame"):
                        # print(f"    進度: {t+1}/{frameCount} 幀", end='\r')
                        
                        f_base = getOneFrame(base_file, width_base, height_base, bytesPerPel, t)
                        f_prev = getOneFrame(even_file, width_el, height_el, bytesPerPel, t)
                        f_next = getOneFrame(even_file, width_el, height_el, bytesPerPel, t+1)
                        
                        # 1. 轉 Tensor (歸一化 0~1) 並送上 GPU
                        t_base_y = torch.from_numpy(f_base['y'].reshape(1080, 1920)).to(device=device, dtype=torch.float32).view(1,1,1080,1920) / 1023.0
                        t_base_u = torch.from_numpy(f_base['u'].reshape(540, 960)).to(device=device, dtype=torch.float32).view(1,1,540,960) / 1023.0
                        t_base_v = torch.from_numpy(f_base['v'].reshape(540, 960)).to(device=device, dtype=torch.float32).view(1,1,540,960) / 1023.0
                        
                        t_p_y = torch.from_numpy(f_prev['y'].reshape(2160, 3840)).to(device=device, dtype=torch.float32).view(1,1,2160,3840) / 1023.0
                        t_p_u = torch.from_numpy(f_prev['u'].reshape(1080, 1920)).to(device=device, dtype=torch.float32).view(1,1,1080,1920) / 1023.0
                        t_p_v = torch.from_numpy(f_prev['v'].reshape(1080, 1920)).to(device=device, dtype=torch.float32).view(1,1,1080,1920) / 1023.0
                        
                        t_n_y = torch.from_numpy(f_next['y'].reshape(2160, 3840)).to(device=device, dtype=torch.float32).view(1,1,2160,3840) / 1023.0
                        t_n_u = torch.from_numpy(f_next['u'].reshape(1080, 1920)).to(device=device, dtype=torch.float32).view(1,1,1080,1920) / 1023.0
                        t_n_v = torch.from_numpy(f_next['v'].reshape(1080, 1920)).to(device=device, dtype=torch.float32).view(1,1,1080,1920) / 1023.0

                        # 2. 全 GPU 極速升頻
                        t_up_y = gpu_8tap_upscale_y(t_base_y)
                        t_up_u = gpu_4tap_upscale_uv(t_base_u)
                        t_up_v = gpu_4tap_upscale_uv(t_base_v)
                        
                        # 3. 光流運算 (540p 降維打擊版)
                        def get_small_flow_img(tensor_4k):
                            t_small = F.interpolate(tensor_4k, scale_factor=0.25, mode='bilinear', align_corners=False)
                            return torch.clamp(t_small.squeeze() * 255.0, 0, 255).byte().cpu().numpy()

                        up_8b = get_small_flow_img(t_up_y)
                        p_8b = get_small_flow_img(t_p_y)
                        n_8b = get_small_flow_img(t_n_y)
                        
                        # 重新啟用 Farneback
                        flow_p_small = cv2.calcOpticalFlowFarneback(up_8b, p_8b, None, 0.5, 5, 25, 3, 5, 1.2, 0)
                        flow_n_small = cv2.calcOpticalFlowFarneback(up_8b, n_8b, None, 0.5, 5, 25, 3, 5, 1.2, 0)
                        
                        flow_p_t_540 = torch.from_numpy(flow_p_small).to(device=device, dtype=torch.float32).permute(2,0,1).unsqueeze(0)
                        flow_n_t_540 = torch.from_numpy(flow_n_small).to(device=device, dtype=torch.float32).permute(2,0,1).unsqueeze(0)
                        
                        # Chroma 放 2 倍
                        flow_p_chroma = F.interpolate(flow_p_t_540, scale_factor=2.0, mode='bilinear', align_corners=False) * 2.0
                        flow_n_chroma = F.interpolate(flow_n_t_540, scale_factor=2.0, mode='bilinear', align_corners=False) * 2.0

                        # Luma 放 4 倍
                        flow_p_4k = F.interpolate(flow_p_t_540, scale_factor=4.0, mode='bilinear', align_corners=False) * 4.0
                        flow_n_4k = F.interpolate(flow_n_t_540, scale_factor=4.0, mode='bilinear', align_corners=False) * 4.0
                        
                        # # 4. GPU Warping 與融合
                        # # 【最後的微調】可以針對不同影片特性微調這裡的 sigma_val
                        # # 數值越小 -> 越嚴格(容易退回模糊 Base)；數值越大 -> 越寬鬆(容易吃進殘影)
                        # fused_y = fuse_channel(t_up_y, t_p_y, t_n_y, flow_p_4k, flow_n_4k, 25.0/1023.0)
                        # fused_u = fuse_channel(t_up_u, t_p_u, t_n_u, flow_p_chroma, flow_n_chroma, 35.0/1023.0)
                        # fused_v = fuse_channel(t_up_v, t_p_v, t_n_v, flow_p_chroma, flow_n_chroma, 35.0/1023.0)

                        # 4. Warping 與融合
                        # CPU 上 USE_ML_FUSION=True 會很慢；改 False 會用傳統權重融合，通常快很多但結果會不同。
                        if USE_ML_FUSION:
                            fused_y = ml_fuse_channel(t_up_y, t_p_y, t_n_y, flow_p_4k, flow_n_4k)
                            fused_u = ml_fuse_channel(t_up_u, t_p_u, t_n_u, flow_p_chroma, flow_n_chroma)
                            fused_v = ml_fuse_channel(t_up_v, t_p_v, t_n_v, flow_p_chroma, flow_n_chroma)
                        else:
                            fused_y = fuse_channel(t_up_y, t_p_y, t_n_y, flow_p_4k, flow_n_4k, 25.0/1023.0)
                            fused_u = fuse_channel(t_up_u, t_p_u, t_n_u, flow_p_chroma, flow_n_chroma, 35.0/1023.0)
                            fused_v = fuse_channel(t_up_v, t_p_v, t_n_v, flow_p_chroma, flow_n_chroma, 35.0/1023.0)
                        
                        # 5. 拉回 CPU 儲存
                        f_out.write((fused_y.squeeze().cpu().numpy() * 1023.0).clip(0, 1023).astype('<u2').tobytes())
                        f_out.write((fused_u.squeeze().cpu().numpy() * 1023.0).clip(0, 1023).astype('<u2').tobytes())
                        f_out.write((fused_v.squeeze().cpu().numpy() * 1023.0).clip(0, 1023).astype('<u2').tobytes())
                        
                print(f"\n    [成功] 儲存至: {out_file}")

if __name__ == '__main__':
    main()