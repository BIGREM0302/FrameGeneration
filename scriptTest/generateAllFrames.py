import os
import cv2
import numpy as np
from yuvProc import getOneFrame

def temporal_super_resolution(up_y, prev_y, next_y, sigma=30.0):
    """
    對 Y 通道進行光流估計與自適應融合 (含縮小加速版)
    """
    h, w = up_y.shape
    
    # 1. 轉為 8-bit (0-255) 供 OpenCV 計算
    up_8b = np.clip(up_y / 4.0, 0, 255).astype(np.uint8)
    prev_8b = np.clip(prev_y / 4.0, 0, 255).astype(np.uint8)
    next_8b = np.clip(next_y / 4.0, 0, 255).astype(np.uint8)

    # 【CPU 加速】縮小到 1080p 計算光流
    up_8b_small = cv2.resize(up_8b, (w//2, h//2), interpolation=cv2.INTER_AREA)
    prev_8b_small = cv2.resize(prev_8b, (w//2, h//2), interpolation=cv2.INTER_AREA)
    next_8b_small = cv2.resize(next_8b, (w//2, h//2), interpolation=cv2.INTER_AREA)

    flow_prev_small = cv2.calcOpticalFlowFarneback(up_8b_small, prev_8b_small, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    flow_next_small = cv2.calcOpticalFlowFarneback(up_8b_small, next_8b_small, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # 放大回 4K 光流場 (向量值乘 2)
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

    # 3. 自適應融合
    up_float = up_y.astype(np.float32)
    diff_prev = np.abs(warped_prev - up_float)
    diff_next = np.abs(warped_next - up_float)

    weight_prev = np.exp(-diff_prev / sigma)
    weight_next = np.exp(-diff_next / sigma)
    weight_up = np.full((h, w), 0.3, dtype=np.float32) # 保底權重

    numerator = (weight_prev * warped_prev) + (weight_next * warped_next) + (weight_up * up_float)
    denominator = weight_prev + weight_next + weight_up
    
    final_frame = numerator / denominator
    return np.clip(final_frame, 0, 1023).astype(np.uint16)

def main():
    width = 3840
    height = 2160
    bytesPerPel = 2
    
    # 計算單幀 420 10-bit YUV 的位元組大小
    byteN = int((width * height * 1.5) * bytesPerPel)
    
    # 4部影片的配置資料庫
    video_configs = {
        'AMS05': {
            'qps': [27, 32, 37, 42],
            'up': '../bitstream/upscaled/odd_H2_H3_AMS05_{qp}_0_5_up.layer0.yuv',
            'even': '../bitstream/enhance/even_H2_H3_AMS05_{qp}_0_5.layer1.yuv',
            'out': '../bitstream/generated/odd_H2_H3_AMS05_{qp}_0_5_gen.layer0.yuv'
        },
        'WalkInPark': {
            'qps': [27, 32, 37, 42],
            'up': '../bitstream/upscaled/odd_H2_WalkInPark_{qp}_0_4_up.layer0.yuv',
            'even': '../bitstream/enhance/even_H2_WalkInPark_{qp}_0_4.layer1.yuv',
            'out': '../bitstream/generated/odd_H2_WalkInPark_{qp}_0_4_gen.layer0.yuv'
        },
        'Procession': {
            'qps': [25, 30, 35, 40], # Procession 的 QP 規格不同
            'up': '../bitstream/upscaled/odd_Procession_{qp}_0_4_up.layer0.yuv',
            'even': '../bitstream/enhance/even_Procession_{qp}_0_4.layer1.yuv',
            'out': '../bitstream/generated/odd_Procession_{qp}_0_4_gen.layer0.yuv'
        },
        'Zombie': {
            'qps': [27, 32, 37, 42],
            'up': '../bitstream/upscaled/odd_ZombieClimbing2_{qp}_0_4_up.layer0.yuv',
            'even': '../bitstream/enhance/even_ZombieClimbing2_{qp}_0_4.layer1.yuv',
            'out': '../bitstream/generated/odd_ZombieClimbing2_{qp}_0_4_gen.layer0.yuv'
        }
    }
    
    out_dir = '../bitstream/generated'
    os.makedirs(out_dir, exist_ok=True)
    
    for video_name, config in video_configs.items():
        print(f"\n==================== 開始處理影片: {video_name} ====================")
        
        for qp in config['qps']:
            up_file = config['up'].format(qp=qp)
            even_file = config['even'].format(qp=qp)
            out_file = config['out'].format(qp=qp)
            
            if not os.path.exists(up_file) or not os.path.exists(even_file):
                print(f" ⚠️ 找不到檔案，跳過: QP {qp}")
                continue
                
            # 【關鍵】動態計算當前影片的奇數幀總數 (frameCount = 總幀數 - 1)
            file_size = os.path.getsize(up_file)
            total_frames = file_size // byteN
            frameCount = total_frames - 1
            
            print(f"--> 正在處理 QP {qp} (總共 {frameCount} 幀)...")
            
            with open(out_file, 'wb') as f_out:
                for t in range(frameCount):
                    print(f"    進度: {t+1}/{frameCount} 幀", end='\r')
                    
                    frame_up = getOneFrame(up_file, width, height, bytesPerPel, t)
                    frame_even_prev = getOneFrame(even_file, width, height, bytesPerPel, t)
                    frame_even_next = getOneFrame(even_file, width, height, bytesPerPel, t+1)
                    
                    y_up = frame_up['y'].reshape((height, width))
                    y_prev = frame_even_prev['y'].reshape((height, width))
                    y_next = frame_even_next['y'].reshape((height, width))
                    
                    # 執行純 CV 運動補償融合
                    y_gen = temporal_super_resolution(y_up, y_prev, y_next, sigma=30.0)
                    
                    # 寫入 Y 通道
                    f_out.write(y_gen.astype('<u2').tobytes())
                    # 直接複製原 upscaled 的 U, V 通道
                    f_out.write(frame_up['u'].astype('<u2').tobytes())
                    f_out.write(frame_up['v'].astype('<u2').tobytes())
            print(f"\n    [成功] 儲存至: {out_file}")

if __name__ == '__main__':
    main()