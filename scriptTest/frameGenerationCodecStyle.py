import os
import cv2
import numpy as np
from yuvProc import getOneFrame

# ============================================================
# Codec-style Temporal Mode Decision v2
# ============================================================
# Important fix from v1:
#   DO NOT include pure spatial upsample in LR block decision.
#
# Why:
#   The zero-phase spatial upsample satisfies:
#       up_y[0::2, 0::2] == base_y_lr
#   Therefore, if spatial is a candidate and candidates are scored by
#   downsampling to the LR grid, spatial always wins trivially.
#
# This v2 uses your original full-frame temporal fusion as the safe baseline.
# Block decision is only allowed to replace it with temporal candidates
# when they are clearly better on the LR observation.
#
# Dependencies:
#   numpy
#   opencv-python

# ---------------- Tunables ----------------
FLOW_PROFILE = "original"      # "original" or "stable"

# LR block size: 8, 16, 24, 32.
# Smaller = more local decisions, can help motion boundaries but may be noisy.
LR_BLOCK = 16 #-29.953%, -29.935%
LR_BLOCK = 8 # -29.712%

# Soft OBMC-like mask smoothing. Try 0, 3, 5.
# Default 0 first because blurred wrong decisions can hurt PSNR.
MODE_BLUR = 0

DECISION_METRIC = "sad"       # "sad" or "mse"

# Original temporal fusion parameters, same spirit as your best baseline.
SIGMA_Y = 30.0
WEIGHT_UP_Y = 0.30

# Candidate residual clipping relative to spatial anchor.
# This prevents destructive candidate blocks.
RESIDUAL_CLIP_Y = 160.0

# Only switch away from baseline if the best temporal candidate improves
# the LR score by at least this ratio.
# 0.00 = always choose best candidate.
# 0.03 = candidate must be 3% better than baseline.
SWITCH_MARGIN = 0.015 # -29.953%
SWITCH_MARGIN = 0.00 # -29.935%, -29.712%

# Additional guard: if best candidate LR error is too large, keep baseline.
# This is not compared to spatial; it just blocks catastrophic temporal candidates.
ABS_ERROR_GUARD = 34.0 # -29.953%
ABS_ERROR_GUARD = 50.0 # -29.935%, -29.712%


# Use nearest mode map first. Blending mode masks may average incompatible predictors.
USE_SOFT_MODE = False

# Candidate list control
USE_PREV_NEXT_RAW = True
USE_BI = True
USE_MEDIAN = True
USE_DIRECTIONAL_ORIGINALS = True

# Chroma: keep spatial.
USE_TEMPORAL_CHROMA = False


# ============================================================
# Zero-phase upscalers
# ============================================================
def phase_aligned_8tap_dctif_2x(img):
    img = img.astype(np.float32, copy=False)
    h, w = img.shape

    inter = np.empty((h, w * 2), dtype=np.float32)
    inter[:, 0::2] = img

    img_pad_h = np.pad(img, ((0, 0), (4, 4)), mode='reflect')
    p0_h = img_pad_h[:, 1 : w + 1]
    p1_h = img_pad_h[:, 2 : w + 2]
    p2_h = img_pad_h[:, 3 : w + 3]
    p3_h = img_pad_h[:, 4 : w + 4]
    p4_h = img_pad_h[:, 5 : w + 5]
    p5_h = img_pad_h[:, 6 : w + 6]
    p6_h = img_pad_h[:, 7 : w + 7]
    p7_h = img_pad_h[:, 8 : w + 8]

    inter[:, 1::2] = (
        -1.0 * p0_h + 4.0 * p1_h - 11.0 * p2_h + 40.0 * p3_h +
         40.0 * p4_h - 11.0 * p5_h + 4.0 * p6_h - 1.0 * p7_h
    ) / 64.0

    out = np.empty((h * 2, w * 2), dtype=np.float32)
    out[0::2, :] = inter

    inter_pad_v = np.pad(inter, ((4, 4), (0, 0)), mode='reflect')
    p0_v = inter_pad_v[1 : h + 1, :]
    p1_v = inter_pad_v[2 : h + 2, :]
    p2_v = inter_pad_v[3 : h + 3, :]
    p3_v = inter_pad_v[4 : h + 4, :]
    p4_v = inter_pad_v[5 : h + 5, :]
    p5_v = inter_pad_v[6 : h + 6, :]
    p6_v = inter_pad_v[7 : h + 7, :]
    p7_v = inter_pad_v[8 : h + 8, :]

    out[1::2, :] = (
        -1.0 * p0_v + 4.0 * p1_v - 11.0 * p2_v + 40.0 * p3_v +
         40.0 * p4_v - 11.0 * p5_v + 4.0 * p6_v - 1.0 * p7_v
    ) / 64.0

    return out


def phase_aligned_4tap_upscale_2x(img):
    img = img.astype(np.float32, copy=False)
    h, w = img.shape

    inter = np.empty((h, w * 2), dtype=np.float32)
    inter[:, 0::2] = img

    img_pad_h = np.pad(img, ((0, 0), (2, 2)), mode='reflect')
    p0_h = img_pad_h[:, 1 : w + 1]
    p1_h = img_pad_h[:, 2 : w + 2]
    p2_h = img_pad_h[:, 3 : w + 3]
    p3_h = img_pad_h[:, 4 : w + 4]

    inter[:, 1::2] = -0.0625 * p0_h + 0.5625 * p1_h + 0.5625 * p2_h - 0.0625 * p3_h

    out = np.empty((h * 2, w * 2), dtype=np.float32)
    out[0::2, :] = inter

    inter_pad_v = np.pad(inter, ((2, 2), (0, 0)), mode='reflect')
    p0_v = inter_pad_v[1 : h + 1, :]
    p1_v = inter_pad_v[2 : h + 2, :]
    p2_v = inter_pad_v[3 : h + 3, :]
    p3_v = inter_pad_v[4 : h + 4, :]

    out[1::2, :] = -0.0625 * p0_v + 0.5625 * p1_v + 0.5625 * p2_v - 0.0625 * p3_v
    return out


def upscale_frame(frame_base, width_base=1920, height_base=1080):
    y = frame_base['y'].reshape((height_base, width_base)).astype(np.float32)
    u = frame_base['u'].reshape((height_base // 2, width_base // 2)).astype(np.float32)
    v = frame_base['v'].reshape((height_base // 2, width_base // 2)).astype(np.float32)

    y_up = phase_aligned_8tap_dctif_2x(y)
    u_up = phase_aligned_4tap_upscale_2x(u)
    v_up = phase_aligned_4tap_upscale_2x(v)

    return {
        'y': np.rint(np.clip(y_up, 0, 1023)).astype(np.uint16),
        'u': np.rint(np.clip(u_up, 0, 1023)).astype(np.uint16),
        'v': np.rint(np.clip(v_up, 0, 1023)).astype(np.uint16)
    }


# ============================================================
# Motion compensation
# ============================================================
def _to_8bit(img10):
    return np.clip(img10.astype(np.float32) / 4.0, 0, 255).astype(np.uint8)


def _farneback_params():
    if FLOW_PROFILE == "stable":
        return dict(
            pyr_scale=0.5,
            levels=4,
            winsize=25,
            iterations=4,
            poly_n=7,
            poly_sigma=1.5,
            flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN
        )
    return dict(
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )


def motion_compensate_y(up_y, prev_y, next_y):
    h, w = up_y.shape

    up_8b = _to_8bit(up_y)
    prev_8b = _to_8bit(prev_y)
    next_8b = _to_8bit(next_y)

    up_small = cv2.resize(up_8b, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    prev_small = cv2.resize(prev_8b, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    next_small = cv2.resize(next_8b, (w // 2, h // 2), interpolation=cv2.INTER_AREA)

    params = _farneback_params()
    flow_prev_small = cv2.calcOpticalFlowFarneback(up_small, prev_small, None, **params)
    flow_next_small = cv2.calcOpticalFlowFarneback(up_small, next_small, None, **params)

    flow_prev = cv2.resize(flow_prev_small, (w, h), interpolation=cv2.INTER_LINEAR) * 2.0
    flow_next = cv2.resize(flow_next_small, (w, h), interpolation=cv2.INTER_LINEAR) * 2.0

    grid_x, grid_y = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32)
    )

    warped_prev = cv2.remap(
        prev_y.astype(np.float32),
        grid_x + flow_prev[..., 0],
        grid_y + flow_prev[..., 1],
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    warped_next = cv2.remap(
        next_y.astype(np.float32),
        grid_x + flow_next[..., 0],
        grid_y + flow_next[..., 1],
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    return warped_prev, warped_next


# ============================================================
# Temporal prediction candidates
# ============================================================
def original_temporal_fusion(up_y, warped_prev, warped_next):
    up_float = up_y.astype(np.float32)
    diff_prev = np.abs(warped_prev - up_float)
    diff_next = np.abs(warped_next - up_float)

    weight_prev = np.exp(-diff_prev / SIGMA_Y).astype(np.float32)
    weight_next = np.exp(-diff_next / SIGMA_Y).astype(np.float32)
    weight_up = np.float32(WEIGHT_UP_Y)

    numerator = weight_prev * warped_prev + weight_next * warped_next + weight_up * up_float
    denominator = weight_prev + weight_next + weight_up
    return numerator / denominator


def directional_fusion(up_float, warped, sigma=SIGMA_Y):
    diff = np.abs(warped - up_float)
    w = np.exp(-diff / sigma).astype(np.float32)
    return (w * warped + WEIGHT_UP_Y * up_float) / (w + WEIGHT_UP_Y)


def _clip_to_spatial(cand, up_float):
    return up_float + np.clip(cand - up_float, -RESIDUAL_CLIP_Y, RESIDUAL_CLIP_Y)


def _lr_view(cand4k):
    return cand4k[0::2, 0::2]


def _make_candidates(up_float, warped_prev, warped_next):
    baseline = original_temporal_fusion(up_float, warped_prev, warped_next)

    candidates = [baseline]
    names = ["baseline_original_fusion"]

    if USE_DIRECTIONAL_ORIGINALS:
        candidates.append(directional_fusion(up_float, warped_prev))
        names.append("prev_directional_fusion")
        candidates.append(directional_fusion(up_float, warped_next))
        names.append("next_directional_fusion")

    if USE_PREV_NEXT_RAW:
        candidates.append(_clip_to_spatial(warped_prev, up_float))
        names.append("prev_warp")
        candidates.append(_clip_to_spatial(warped_next, up_float))
        names.append("next_warp")

    if USE_BI:
        candidates.append(_clip_to_spatial(0.5 * (warped_prev + warped_next), up_float))
        names.append("bi_average")

    if USE_MEDIAN:
        med = np.median(np.stack([up_float, warped_prev, warped_next], axis=0), axis=0).astype(np.float32)
        candidates.append(_clip_to_spatial(med, up_float))
        names.append("median_spatial_prev_next")

    return candidates, names


def _compute_block_score(err, block):
    h, w = err.shape
    by = (h + block - 1) // block
    bx = (w + block - 1) // block
    score = np.zeros((by, bx), dtype=np.float32)

    integ = cv2.integral(err.astype(np.float32))
    for iy in range(by):
        y0 = iy * block
        y1 = min((iy + 1) * block, h)
        for ix in range(bx):
            x0 = ix * block
            x1 = min((ix + 1) * block, w)
            s = integ[y1, x1] - integ[y0, x1] - integ[y1, x0] + integ[y0, x0]
            score[iy, ix] = s / ((y1 - y0) * (x1 - x0))
    return score


def _block_scores(candidate_lr_list, base_lr):
    scores = []
    base = base_lr.astype(np.float32)

    for cand_lr in candidate_lr_list:
        diff = cand_lr.astype(np.float32) - base
        if DECISION_METRIC == "mse":
            err = diff * diff
        else:
            err = np.abs(diff)
        scores.append(_compute_block_score(err, LR_BLOCK))

    return np.stack(scores, axis=0)


def _upsample_labels_to_4k(best_idx, num_modes, out_h, out_w):
    by, bx = best_idx.shape

    if not USE_SOFT_MODE:
        # Nearest upsample label map to 4K grid.
        label_small = best_idx.astype(np.float32)
        label_4k = cv2.resize(label_small, (out_w, out_h), interpolation=cv2.INTER_NEAREST).astype(np.int32)
        masks = np.zeros((num_modes, out_h, out_w), dtype=np.float32)
        for k in range(num_modes):
            masks[k] = (label_4k == k).astype(np.float32)
        return masks

    masks_lr = np.zeros((num_modes, by, bx), dtype=np.float32)
    for k in range(num_modes):
        masks_lr[k] = (best_idx == k).astype(np.float32)

    masks_4k = []
    for k in range(num_modes):
        m = cv2.resize(masks_lr[k], (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        if MODE_BLUR > 0:
            ksize = int(MODE_BLUR * 2 + 1)
            ksize = max(3, ksize | 1)
            m = cv2.GaussianBlur(m, (ksize, ksize), 0)
        masks_4k.append(m.astype(np.float32))

    masks = np.stack(masks_4k, axis=0)
    masks /= np.sum(masks, axis=0, keepdims=True) + 1e-6
    return masks


def codec_style_temporal_sr_v2(up_y, base_y_lr, prev_y, next_y):
    up_float = up_y.astype(np.float32)

    warped_prev, warped_next = motion_compensate_y(up_float, prev_y, next_y)
    candidates, names = _make_candidates(up_float, warped_prev, warped_next)

    candidate_lr_list = [_lr_view(c) for c in candidates]
    scores = _block_scores(candidate_lr_list, base_y_lr)

    # Baseline is candidate 0.
    baseline_score = scores[0]
    best_idx = np.argmin(scores, axis=0).astype(np.int32)
    best_score = np.min(scores, axis=0)

    # Conservative mode decision:
    # Keep original temporal fusion unless another temporal mode is clearly better.
    improvement = (baseline_score - best_score) / (baseline_score + 1e-6)
    keep_baseline = improvement < SWITCH_MARGIN

    # Absolute guard on selected temporal candidate.
    keep_baseline |= (best_score > ABS_ERROR_GUARD)

    best_idx[keep_baseline] = 0

    h, w = up_float.shape
    masks = _upsample_labels_to_4k(best_idx, len(candidates), h, w)

    out = np.zeros_like(up_float, dtype=np.float32)
    for k, c in enumerate(candidates):
        out += masks[k] * c

    return np.rint(np.clip(out, 0, 1023)).astype(np.uint16)


# ============================================================
# Main
# ============================================================
def main():
    width_base = 1920
    height_base = 1080
    width_el = 3840
    height_el = 2160
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

    print("============================================================")
    print("Codec-style Temporal Mode Decision v2")
    print(f"FLOW_PROFILE       = {FLOW_PROFILE}")
    print(f"LR_BLOCK           = {LR_BLOCK}")
    print(f"MODE_BLUR          = {MODE_BLUR}")
    print(f"USE_SOFT_MODE      = {USE_SOFT_MODE}")
    print(f"DECISION_METRIC    = {DECISION_METRIC}")
    print(f"SIGMA_Y            = {SIGMA_Y}")
    print(f"WEIGHT_UP_Y        = {WEIGHT_UP_Y}")
    print(f"RESIDUAL_CLIP_Y    = {RESIDUAL_CLIP_Y}")
    print(f"SWITCH_MARGIN      = {SWITCH_MARGIN}")
    print(f"ABS_ERROR_GUARD    = {ABS_ERROR_GUARD}")
    print("============================================================")

    for video_name, config in video_configs.items():
        print(f"\n==================== 開始處理影片: {video_name} ====================")

        for qp in config['qps']:
            base_file = config['base'].format(qp=qp)
            even_file = config['even'].format(qp=qp)
            out_file = config['out'].format(qp=qp)

            if not os.path.exists(base_file) or not os.path.exists(even_file):
                print(f" ⚠️ 找不到輸入檔案，跳過: QP {qp}")
                continue

            file_size = os.path.getsize(base_file)
            total_frames = file_size // byteN_base
            frameCount = total_frames - 1

            print(f"--> 正在處理 QP {qp} (預計生成 {frameCount} 幀)...")

            with open(out_file, 'wb') as f_out:
                for t in range(frameCount):
                    print(f"    進度: {t + 1}/{frameCount} 幀", end='\r')

                    frame_base = getOneFrame(base_file, width_base, height_base, bytesPerPel, t)
                    frame_up = upscale_frame(frame_base, width_base, height_base)

                    base_y_lr = frame_base['y'].reshape((height_base, width_base)).astype(np.float32)

                    frame_even_prev = getOneFrame(even_file, width_el, height_el, bytesPerPel, t)
                    frame_even_next = getOneFrame(even_file, width_el, height_el, bytesPerPel, t + 1)

                    y_up = frame_up['y']
                    y_prev = frame_even_prev['y'].reshape((height_el, width_el))
                    y_next = frame_even_next['y'].reshape((height_el, width_el))

                    y_gen = codec_style_temporal_sr_v2(y_up, base_y_lr, y_prev, y_next)

                    f_out.write(y_gen.astype('<u2').tobytes())
                    f_out.write(frame_up['u'].astype('<u2').tobytes())
                    f_out.write(frame_up['v'].astype('<u2').tobytes())

            print(f"\n    [成功] 儲存至: {out_file}")


if __name__ == '__main__':
    main()
