"""
Currently not work very well, may need to refine some parameters
"""

import os
import cv2
import numpy as np
from yuvProc import getOneFrame

def phase_aligned_upscale_2x(img):
    """
    手動實作零相位偏移(Zero-Phase Shift)的雙線性插值 (Linear FIR)
    確保放大後的偶數座標完全對齊原始影像，避免 OpenCV 帶來的亞像素偏移導致 PSNR 暴跌。
    """
    h, w = img.shape
    out = np.zeros((h * 2, w * 2), dtype=np.float32)

    # 1. 偶數點：完全複製原圖 (Phase 0)
    out[0::2, 0::2] = img

    # 2. 奇數點 (水平)：左右相鄰像素平均
    out[0::2, 1:-1:2] = (img[:, :-1] + img[:, 1:]) * 0.5
    out[0::2, -1] = img[:, -1] # 邊界處理

    # 3. 奇數點 (垂直)：上下相鄰像素平均
    out[1:-1:2, 0::2] = (img[:-1, :] + img[1:, :]) * 0.5
    out[-1, 0::2] = img[-1, :] # 邊界處理

    # 4. 對角線中心點：四周像素平均
    out[1:-1:2, 1:-1:2] = (img[:-1, :-1] + img[:-1, 1:] + img[1:, :-1] + img[1:, 1:]) * 0.25
    out[-1, 1:-1:2] = (img[-1, :-1] + img[-1, 1:]) * 0.5
    out[1:-1:2, -1] = (img[:-1, -1] + img[1:, -1]) * 0.5
    out[-1, -1] = img[-1, -1]

    return out

def phase_aligned_4tap_upscale_2x(img):
    """
    修正版：零相位偏移可分離式 4-tap (Bicubic 級別) 放大器
    嚴格確保偶數座標完全對齊原圖，切片長度完美匹配 w 與 h。
    """
    h, w = img.shape
    
    # --- Step 1: 水平方向放大 (H, W) -> (H, W * 2) ---
    inter = np.empty((h, w * 2), dtype=np.float32)
    # 偶數欄：完美複製原圖
    inter[:, 0::2] = img
    
    # 邊界擴展 (左右各補 2 像素)
    img_pad_h = np.pad(img, ((0, 0), (2, 2)), mode='reflect')
    
    # 奇數欄：精準擷取長度為 w (1920) 的 4 個鄰近卷積窗
    p0_h = img_pad_h[:, 1 : w + 1]
    p1_h = img_pad_h[:, 2 : w + 2]
    p2_h = img_pad_h[:, 3 : w + 3]
    p3_h = img_pad_h[:, 4 : w + 4]
    
    # 套用 Catmull-Rom Bicubic 權重 [-1/16, 9/16, 9/16, -1/16]
    inter[:, 1::2] = -0.0625 * p0_h + 0.5625 * p1_h + 0.5625 * p2_h - 0.0625 * p3_h

    # --- Step 2: 垂直方向放大 (H, W * 2) -> (H * 2, W * 2) ---
    out = np.empty((h * 2, w * 2), dtype=np.float32)
    # 偶數列：直接複製水平放大的結果
    out[0::2, :] = inter
    
    # 邊界擴展 (上下各補 2 像素)
    inter_pad_v = np.pad(inter, ((2, 2), (0, 0)), mode='reflect')
    
    # 奇數列：精準擷取長度為 h (1080) 的 4 個鄰近卷積窗
    p0_v = inter_pad_v[1 : h + 1, :]
    p1_v = inter_pad_v[2 : h + 2, :]
    p2_v = inter_pad_v[3 : h + 3, :]
    p3_v = inter_pad_v[4 : h + 4, :]
    
    out[1::2, :] = -0.0625 * p0_v + 0.5625 * p1_v + 0.5625 * p2_v - 0.0625 * p3_v

    return out

def upscale_frame_lanczos(frame_base, width_base=1920, height_base=1080):
    """
    使用 8-tap Lanczos 插值濾波器，逼近 VVC/HEVC 官方標準的 DCT-IF，
    最大程度保留高頻細節，作為後續光流的極佳基底。
    """
    y = frame_base['y'].reshape((height_base, width_base)).astype(np.float32)
    u = frame_base['u'].reshape((height_base // 2, width_base // 2)).astype(np.float32)
    v = frame_base['v'].reshape((height_base // 2, width_base // 2)).astype(np.float32)
    
    #y_up = cv2.resize(y, (width_base * 2, height_base * 2), interpolation=cv2.INTER_LANCZOS4)
    #u_up = cv2.resize(u, (width_base, height_base), interpolation=cv2.INTER_LANCZOS4)
    #v_up = cv2.resize(v, (width_base, height_base), interpolation=cv2.INTER_LANCZOS4)
    # +47% fuck
    
    # 全部換成零相位偏移的 4-tap 高階放大器
    #y_up = phase_aligned_4tap_upscale_2x(y)
    #u_up = phase_aligned_4tap_upscale_2x(u)
    #v_up = phase_aligned_4tap_upscale_2x(v)
    # -21.32%

    y_up = phase_aligned_upscale_2x(y)
    u_up = phase_aligned_upscale_2x(u)
    v_up = phase_aligned_upscale_2x(v)
    # -14%

    return {
        'y': np.clip(y_up, 0, 1023).astype(np.uint16),
        'u': np.clip(u_up, 0, 1023).astype(np.uint16),
        'v': np.clip(v_up, 0, 1023).astype(np.uint16)
    }

def warp_and_fuse_channel(up_c, prev_c, next_c, flow_prev, flow_next, sigma=30.0, threshold=45.0, base_weight=0.5):
    """
    對單一通道進行 Warping 與遮擋處理融合
    """
    h, w = up_c.shape
    
    # 建立網格
    y_grid, x_grid = np.mgrid[0:h, 0:w].reshape(2, -1)
    
    # Remap 前一幀
    map_x_prev = (x_grid + flow_prev[..., 0].flatten()).astype(np.float32).reshape(h, w)
    map_y_prev = (y_grid + flow_prev[..., 1].flatten()).astype(np.float32).reshape(h, w)
    warped_prev = cv2.remap(prev_c.astype(np.float32), map_x_prev, map_y_prev, cv2.INTER_CUBIC)

    # Remap 後一幀
    map_x_next = (x_grid + flow_next[..., 0].flatten()).astype(np.float32).reshape(h, w)
    map_y_next = (y_grid + flow_next[..., 1].flatten()).astype(np.float32).reshape(h, w)
    warped_next = cv2.remap(next_c.astype(np.float32), map_x_next, map_y_next, cv2.INTER_CUBIC)

    # 計算差異
    up_float = up_c.astype(np.float32)
    diff_prev = np.abs(warped_prev - up_float)
    diff_next = np.abs(warped_next - up_float)

    # Hard Threshold Mask: 若殘差過大 (如物體邊緣遮擋、光流算錯)，直接捨棄該參考點
    mask_prev = (diff_prev < threshold).astype(np.float32)
    mask_next = (diff_next < threshold).astype(np.float32)

    # 計算權重 (加入 mask 過濾)
    weight_prev = np.exp(-diff_prev / sigma) * mask_prev
    weight_next = np.exp(-diff_next / sigma) * mask_next
    weight_up = np.full((h, w), base_weight, dtype=np.float32) # Base 幀保底權重

    # 融合
    numerator = (weight_prev * warped_prev) + (weight_next * warped_next) + (weight_up * up_float)
    denominator = weight_prev + weight_next + weight_up
    
    final_c = numerator / denominator
    return np.clip(final_c, 0, 1023).astype(np.uint16)

def temporal_super_resolution_advanced(up_frame, prev_frame, next_frame):
    """
    全通道時間超解析度融合 (採用 DIS Optical Flow)
    """
    h, w = up_frame['y'].shape
    
    # --- 1. 計算 Y 通道光流 ---
    # 將 10-bit (0-1023) 轉為 8-bit (0-255) 供 OpenCV 計算光流
    up_y_8b = np.clip(up_frame['y'] / 4.0, 0, 255).astype(np.uint8)
    prev_y_8b = np.clip(prev_frame['y'] / 4.0, 0, 255).astype(np.uint8)
    next_y_8b = np.clip(next_frame['y'] / 4.0, 0, 255).astype(np.uint8)

    # 使用 DIS 光流 (預設 MEDIUM，兼具極高畫質與合理速度)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    flow_prev = dis.calc(up_y_8b, prev_y_8b, None)
    flow_next = dis.calc(up_y_8b, next_y_8b, None)

    # --- 2. Y 通道融合 ---
    y_gen = warp_and_fuse_channel(
        up_frame['y'], prev_frame['y'], next_frame['y'], 
        flow_prev, flow_next, 
        sigma=25.0, threshold=45.0, base_weight=0.4
    )

    # --- 3. U/V 通道光流調整 ---
    # Chroma 尺寸減半，因此光流場的尺寸與向量大小都要除以 2
    flow_prev_chroma = cv2.resize(flow_prev, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR) * 0.5
    flow_next_chroma = cv2.resize(flow_next, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR) * 0.5

    # --- 4. U 通道融合 ---
    u_gen = warp_and_fuse_channel(
        up_frame['u'], prev_frame['u'], next_frame['u'], 
        flow_prev_chroma, flow_next_chroma, 
        sigma=15.0, threshold=30.0, base_weight=0.6 # 色度較不敏感，稍微提高 Base 權重與收緊閥值
    )

    # --- 5. V 通道融合 ---
    v_gen = warp_and_fuse_channel(
        up_frame['v'], prev_frame['v'], next_frame['v'], 
        flow_prev_chroma, flow_next_chroma, 
        sigma=15.0, threshold=30.0, base_weight=0.6
    )

    return y_gen, u_gen, v_gen

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
    
    out_dir = '../bitstream/generated'
    os.makedirs(out_dir, exist_ok=True)
    
    for video_name, config in video_configs.items():
        print(f"\n==================== 開始處理: {video_name} ====================")
        
        for qp in config['qps']:
            base_file = config['base'].format(qp=qp)
            even_file = config['even'].format(qp=qp)
            out_file = config['out'].format(qp=qp)
            
            if not os.path.exists(base_file) or not os.path.exists(even_file):
                print(f" ⚠️ 找不到輸入檔案，跳過: QP {qp}")
                continue
                
            file_size = os.path.getsize(base_file)
            frameCount = (file_size // byteN_base) - 1
            
            print(f"--> 正在處理 QP {qp} (預計生成 {frameCount} 幀)...")
            
            with open(out_file, 'wb') as f_out:
                for t in range(frameCount):
                    print(f"    進度: {t+1}/{frameCount} 幀", end='\r')
                    
                    # 1. 讀取 Base 幀並放大 (Lanczos)
                    frame_base_raw = getOneFrame(base_file, width_base, height_base, bytesPerPel, t)
                    frame_up = upscale_frame_lanczos(frame_base_raw, width_base, height_base)
                    
                    # 2. 讀取 Enhance 偶數幀 (格式重整為 dict)
                    prev_raw = getOneFrame(even_file, width_el, height_el, bytesPerPel, t)
                    next_raw = getOneFrame(even_file, width_el, height_el, bytesPerPel, t+1)
                    
                    frame_even_prev = {
                        'y': prev_raw['y'].reshape((height_el, width_el)),
                        'u': prev_raw['u'].reshape((height_el // 2, width_el // 2)),
                        'v': prev_raw['v'].reshape((height_el // 2, width_el // 2))
                    }
                    frame_even_next = {
                        'y': next_raw['y'].reshape((height_el, width_el)),
                        'u': next_raw['u'].reshape((height_el // 2, width_el // 2)),
                        'v': next_raw['v'].reshape((height_el // 2, width_el // 2))
                    }
                    
                    # 3. 執行全通道時間融合
                    y_gen, u_gen, v_gen = temporal_super_resolution_advanced(frame_up, frame_even_prev, frame_even_next)
                    
                    # 4. 寫入檔案
                    f_out.write(y_gen.astype('<u2').tobytes())
                    f_out.write(u_gen.astype('<u2').tobytes())
                    f_out.write(v_gen.astype('<u2').tobytes())
                    
            print(f"\n    [成功] 儲存至: {out_file}")

if __name__ == '__main__':
    main()