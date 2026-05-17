import os
import cv2
import numpy as np
from yuvProc import getOneFrame

def temporal_super_resolution(up_y, prev_y, next_y, sigma=30.0):
    """
    對 Y 通道進行光流估計與自適應融合
    """
    h, w = up_y.shape
    
    # 1. 轉為 8-bit (0-255) 以供 OpenCV 計算
    # 假設 10-bit 資料範圍是 0-1023
    up_8b = np.clip(up_y / 4.0, 0, 255).astype(np.uint8)
    prev_8b = np.clip(prev_y / 4.0, 0, 255).astype(np.uint8)
    next_8b = np.clip(next_y / 4.0, 0, 255).astype(np.uint8)

    # 【加速技巧】將 4K 縮小到 1080p 來計算光流，大幅減少 CPU 計算時間
    up_8b_small = cv2.resize(up_8b, (w//2, h//2), interpolation=cv2.INTER_AREA)
    prev_8b_small = cv2.resize(prev_8b, (w//2, h//2), interpolation=cv2.INTER_AREA)
    next_8b_small = cv2.resize(next_8b, (w//2, h//2), interpolation=cv2.INTER_AREA)

    # 計算光流 (縮小版)
    flow_prev_small = cv2.calcOpticalFlowFarneback(up_8b_small, prev_8b_small, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    flow_next_small = cv2.calcOpticalFlowFarneback(up_8b_small, next_8b_small, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # 將光流放大回 4K (向量大小也要乘以 2)
    flow_prev = cv2.resize(flow_prev_small, (w, h), interpolation=cv2.INTER_LINEAR) * 2.0
    flow_next = cv2.resize(flow_next_small, (w, h), interpolation=cv2.INTER_LINEAR) * 2.0

    # 2. 建立網格進行 Warping
    y, x = np.mgrid[0:h, 0:w].reshape(2, -1)
    
    map_x_prev = (x + flow_prev[..., 0].flatten()).astype(np.float32).reshape(h, w)
    map_y_prev = (y + flow_prev[..., 1].flatten()).astype(np.float32).reshape(h, w)
    warped_prev = cv2.remap(prev_y.astype(np.float32), map_x_prev, map_y_prev, cv2.INTER_CUBIC)

    map_x_next = (x + flow_next[..., 0].flatten()).astype(np.float32).reshape(h, w)
    map_y_next = (y + flow_next[..., 1].flatten()).astype(np.float32).reshape(h, w)
    warped_next = cv2.remap(next_y.astype(np.float32), map_x_next, map_y_next, cv2.INTER_CUBIC)

    # 3. 自適應融合 (Adaptive Blending 考慮 PSNR)
    up_float = up_y.astype(np.float32)
    
    # 計算差異 (SAD)
    diff_prev = np.abs(warped_prev - up_float)
    diff_next = np.abs(warped_next - up_float)

    # 計算權重
    weight_prev = np.exp(-diff_prev / sigma)
    weight_next = np.exp(-diff_next / sigma)
    weight_up = np.full((h, w), 0.3, dtype=np.float32) # 保底權重

    # 加權平均
    numerator = (weight_prev * warped_prev) + (weight_next * warped_next) + (weight_up * up_float)
    denominator = weight_prev + weight_next + weight_up
    
    final_frame = numerator / denominator
    
    # 確保數值在 10-bit 範圍內，並轉回 uint16
    return np.clip(final_frame, 0, 1023).astype(np.uint16)

def main():
    width = 3840
    height = 2160
    bytesPerPel = 2
    frameCount = 49  # 針對 AMS05 影片的奇數幀數量
    
    # 品質參數列表
    qps = [27, 32, 37, 42]
    
    # 建立輸出資料夾
    out_dir = '../bitstream/generated'
    os.makedirs(out_dir, exist_ok=True)
    
    for qp in qps:
        print(f"========== 正在處理 QP {qp} ==========")
        up_file = f'../bitstream/upscaled/odd_H2_H3_AMS05_{qp}_0_5_up.layer0.yuv'
        even_file = f'../bitstream/enhance/even_H2_H3_AMS05_{qp}_0_5.layer1.yuv'
        out_file = f'{out_dir}/odd_H2_H3_AMS05_{qp}_0_5_gen.layer0.yuv'
        
        # 開啟輸出檔案準備寫入
        with open(out_file, 'wb') as f_out:
            for t in range(frameCount):
                print(f"  正在處理第 {t}/{frameCount-1} 幀...", end='\r')
                
                # 讀取幀資訊 (getOneFrame 回傳的是 dictionary 包含 y, u, v 的 1D array)
                frame_up = getOneFrame(up_file, width, height, bytesPerPel, t)
                # 奇數幀 t 對應的前後偶數幀分別為 t 與 t+1
                frame_even_prev = getOneFrame(even_file, width, height, bytesPerPel, t)
                frame_even_next = getOneFrame(even_file, width, height, bytesPerPel, t+1)
                
                # 將 1D array reshape 成 2D 影像矩陣 (Y 通道)
                y_up = frame_up['y'].reshape((height, width))
                y_prev = frame_even_prev['y'].reshape((height, width))
                y_next = frame_even_next['y'].reshape((height, width))
                
                # 執行我們的 SR 演算法 (只處理 Y 通道以節省時間並最大化 PSNR 收益)
                y_gen = temporal_super_resolution(y_up, y_prev, y_next, sigma=30.0)
                
                # 寫入 Y 通道 (轉為 little-endian uint16 byte)
                f_out.write(y_gen.astype('<u2').tobytes())
                
                # U, V 通道我們直接使用原本 upscaled 的結果來節省運算資源
                f_out.write(frame_up['u'].astype('<u2').tobytes())
                f_out.write(frame_up['v'].astype('<u2').tobytes())
                
            print(f"\n  完成儲存: {out_file}")

if __name__ == '__main__':
    main()