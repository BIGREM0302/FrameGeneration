import os
import cv2
import numpy as np
from yuvProc import getOneFrame

def phase_aligned_upscale_2x_v2(img):
    """
    優化版：零相位偏移雙線性插值 (Linear FIR)
    採用邊界鏡像擴展 (Reflect Padding) 與全向量化切片運算，提升畫質與執行速度。
    """
    h, w = img.shape
    
    # 1. 建立 2x 大小的畫布
    out = np.empty((h * 2, w * 2), dtype=np.float32)
    
    # 2. 偶數點 (Phase 0)：直接映射原圖像素
    out[0::2, 0::2] = img

    #為了完美處理邊緣的奇數點，我們對原圖做鏡像擴展 (右方與下方各多補 1 像素)
    # mode='reflect' 會讓邊緣向外延伸： 例如最後幾項為 [A, B, C] -> 擴展為 [A, B, C, B]
    img_pad = np.pad(img, ((0, 1), (0, 1)), mode='reflect')

    # 3. 奇數點 (水平方向)：目前像素與右方像素的平均
    # out[0::2, 1::2] 對應的是 img 偶數列、奇數欄
    out[0::2, 1::2] = (img_pad[:-1, :-1] + img_pad[:-1, 1:]) * 0.5

    # 4. 奇數點 (垂直方向)：目前像素與下方像素的平均
    # out[1::2, 0::2] 對應的是 img 奇數列、偶數欄
    out[1::2, 0::2] = (img_pad[:-1, :-1] + img_pad[1:, :-1]) * 0.5

    # 5. 對角線中心點：周圍四個像素的平均
    # out[1::2, 1::2] 對應的是 img 奇數列、奇數欄
    out[1::2, 1::2] = (img_pad[:-1, :-1] + img_pad[:-1, 1:] + 
                       img_pad[1:, :-1] + img_pad[1:, 1:]) * 0.25

    return out

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

def upscale_frame(frame_base, width_base=1920, height_base=1080):
    """
    將 1080p 的 Base Layer 幀手動 Upscale 到 4K 規格
    這裡先使用高畫質的 Bicubic (雙三次插值) 作為示範，您也可以自行換成特定的 Linear FIR Filter
    """
    # 1. 將 1D 陣列重塑回 2D 影像矩陣 (注意 420 格式下 U/V 寬高各減半)
    y = frame_base['y'].reshape((height_base, width_base)).astype(np.float32)
    u = frame_base['u'].reshape((height_base // 2, width_base // 2)).astype(np.float32)
    v = frame_base['v'].reshape((height_base // 2, width_base // 2)).astype(np.float32)
    
    # 2. 放大 2 倍：Y 變成 3840x2160，U/V 變成 1920x1080
    #y_up = cv2.resize(y, (width_base * 2, height_base * 2), interpolation=cv2.INTER_CUBIC)
    #u_up = cv2.resize(u, (width_base, height_base), interpolation=cv2.INTER_CUBIC)
    #v_up = cv2.resize(v, (width_base, height_base), interpolation=cv2.INTER_CUBIC)
    
    # 2. 替換為我們自定義的零相位偏移放大器
    y_up = phase_aligned_upscale_2x(y)
    u_up = phase_aligned_upscale_2x(u)
    v_up = phase_aligned_upscale_2x(v)

    # 3. 限制 10-bit 數值區間 (0-1023) 並轉回 uint16
    return {
        'y': np.clip(y_up, 0, 1023).astype(np.uint16),
        'u': np.clip(u_up, 0, 1023).astype(np.uint16),
        'v': np.clip(v_up, 0, 1023).astype(np.uint16)
    }

def temporal_super_resolution(up_y, prev_y, next_y, sigma=30.0):
    """
    對 Y 通道進行光流估計與自適應融合 (含縮小加速版)
    """
    h, w = up_y.shape
    
    # 1. 轉為 8-bit (0-255) 供 OpenCV 計算光流
    up_8b = np.clip(up_y / 4.0, 0, 255).astype(np.uint8)
    prev_8b = np.clip(prev_y / 4.0, 0, 255).astype(np.uint8)
    next_8b = np.clip(next_y / 4.0, 0, 255).astype(np.uint8)

    # 【CPU 加速】縮小到 1080p 計算光流
    up_8b_small = cv2.resize(up_8b, (w//2, h//2), interpolation=cv2.INTER_AREA)
    prev_8b_small = cv2.resize(prev_8b, (w//2, h//2), interpolation=cv2.INTER_AREA)
    next_8b_small = cv2.resize(next_8b, (w//2, h//2), interpolation=cv2.INTER_AREA)

    flow_prev_small = cv2.calcOpticalFlowFarneback(up_8b_small, prev_8b_small, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    flow_next_small = cv2.calcOpticalFlowFarneback(up_8b_small, next_8b_small, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # 放大回 4K 光流場 (向量值記得乘 2)
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
    # 基礎定義 (Base=1080p, Enhance/Output=4K)
    width_base = 1920
    height_base = 1080
    width_el = 3840
    height_el = 2160
    bytesPerPel = 2
    
    # 【修正】計算單幀 1080p 420 10-bit YUV 的位元組大小，用來算 base 檔案的幀數
    byteN_base = int((width_base * height_base * 1.5) * bytesPerPel)
    
    # 4部影片的配置資料庫 (全面改為讀取 base 資料夾，若您的實體檔名有出入請微調此處)
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
    
    for video_name, config in video_configs.items():
        print(f"\n==================== 開始處理影片: {video_name} ====================")
        
        for qp in config['qps']:
            base_file = config['base'].format(qp=qp)
            even_file = config['even'].format(qp=qp)
            out_file = config['out'].format(qp=qp)
            
            if not os.path.exists(base_file) or not os.path.exists(even_file):
                print(f" ⚠️ 找不到輸入檔案，跳過: QP {qp}")
                continue
                
            # 【修正】透過 1080p base 檔案大小動態計算奇數幀總數 (frameCount = 總奇數幀數 - 1)
            file_size = os.path.getsize(base_file)
            total_frames = file_size // byteN_base
            frameCount = total_frames - 1
            
            print(f"--> 正在處理 QP {qp} (預計生成 {frameCount} 幀)...")
            
            with open(out_file, 'wb') as f_out:
                for t in range(frameCount):
                    print(f"    進度: {t+1}/{frameCount} 幀", end='\r')
                    
                    # 1. 讀取 1080p 的 Base 幀
                    frame_base = getOneFrame(base_file, width_base, height_base, bytesPerPel, t)
                    
                    # 2. 在程式內部自行將 Base 幀 Upscale 到 4K
                    frame_up = upscale_frame(frame_base, width_base, height_base)
                    
                    # 3. 讀取 前、後 鄰近的 4K Enhance 偶數幀
                    frame_even_prev = getOneFrame(even_file, width_el, height_el, bytesPerPel, t)
                    frame_even_next = getOneFrame(even_file, width_el, height_el, bytesPerPel, t+1)
                    
                    # 4. 提取 Y 通道進行超解析度與光流融合
                    y_up = frame_up['y']  # 已經是 (2160, 3840) 的 2D 矩陣
                    y_prev = frame_even_prev['y'].reshape((height_el, width_el))
                    y_next = frame_even_next['y'].reshape((height_el, width_el))
                    
                    # 執行時間超解析度融合
                    y_gen = temporal_super_resolution(y_up, y_prev, y_next, sigma=30.0)
                    
                    # 5. 寫入最終的 4K YUV 檔案
                    f_out.write(y_gen.astype('<u2').tobytes())                     # 寫入融合後的 4K Y
                    f_out.write(frame_up['u'].astype('<u2').tobytes())            # 寫入自行放大後的 4K U
                    f_out.write(frame_up['v'].astype('<u2').tobytes())            # 寫入自行放大後的 4K V
                    
            print(f"\n    [成功] 儲存至: {out_file}")

if __name__ == '__main__':
    main()