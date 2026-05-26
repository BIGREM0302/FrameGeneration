import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from yuvProc import getOneFrame

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 8-tap GPU upscale filter for Y channel
# ============================================================
W8_H = torch.tensor(
    [-1, 4, -11, 40, 40, -11, 4, -1],
    dtype=torch.float32,
    device=device
).view(1, 1, 1, 8) / 64.0

W8_V = torch.tensor(
    [-1, 4, -11, 40, 40, -11, 4, -1],
    dtype=torch.float32,
    device=device
).view(1, 1, 8, 1) / 64.0


def gpu_8tap_upscale_y(img_tensor):
    """
    Input : BCHW tensor, usually 1x1x1080x1920, value range 0~1
    Output: BCHW tensor, 2x upscaled, usually 1x1x2160x3840
    """
    b, c, h, w = img_tensor.shape

    # Horizontal 2x upscale
    inter = torch.zeros((b, c, h, w * 2), device=device)
    inter[:, :, :, 0::2] = img_tensor
    inter[:, :, :, 1::2] = F.conv2d(
        F.pad(img_tensor, (3, 4, 0, 0), mode="reflect"),
        W8_H
    )

    # Vertical 2x upscale
    out = torch.zeros((b, c, h * 2, w * 2), device=device)
    out[:, :, 0::2, :] = inter
    out[:, :, 1::2, :] = F.conv2d(
        F.pad(inter, (0, 0, 3, 4), mode="reflect"),
        W8_V
    )

    return out


def get_small_flow_img(tensor_4k):
    """
    Convert 4K tensor image 0~1 to 960x540 uint8 image for Farneback optical flow.
    Input : 1x1x2160x3840 tensor
    Output: 540x960 uint8 numpy array
    """
    t_small = F.interpolate(
        tensor_4k,
        scale_factor=0.25,
        mode="bilinear",
        align_corners=False
    )
    return torch.clamp(t_small.squeeze() * 255.0, 0, 255).byte().cpu().numpy()


def compute_epipolar_cpu(flow, H=540, W=960):
    """
    Compute epipolar error map from optical flow.
    Return:
        epi_map: HxWx1 float32, value roughly 0~1
        F_mat  : 3x3 fundamental matrix, or None
    """
    step_y, step_x = max(1, H // 32), max(1, W // 32)

    pts1, pts2 = [], []
    for y in range(0, H, step_y):
        for x in range(0, W, step_x):
            pts1.append([x, y])
            pts2.append([x + flow[y, x, 0], y + flow[y, x, 1]])

    pts1 = np.array(pts1, dtype=np.float32)
    pts2 = np.array(pts2, dtype=np.float32)

    F_mat, _ = cv2.findFundamentalMat(
        pts1,
        pts2,
        cv2.FM_RANSAC,
        1.0,
        0.99
    )

    if F_mat is not None and F_mat.shape == (3, 3):
        y_idx, x_idx = np.indices((H, W))

        pts1_all = np.stack(
            (x_idx, y_idx, np.ones_like(x_idx)),
            axis=-1
        ).reshape(-1, 3)

        pts2_all = np.stack(
            (x_idx + flow[..., 0], y_idx + flow[..., 1], np.ones_like(x_idx)),
            axis=-1
        ).reshape(-1, 3)

        lines2 = np.dot(pts1_all, F_mat.T)

        num = np.abs(np.sum(pts2_all * lines2, axis=1))
        den = np.sqrt(lines2[:, 0] ** 2 + lines2[:, 1] ** 2) + 1e-6

        epi = np.tanh((num / den).reshape(H, W) / 4.0).astype(np.float32)
        return epi[..., np.newaxis], F_mat

    return np.zeros((H, W, 1), dtype=np.float32), None


def normalize_to_u8(x, percentile=99.0):
    """
    Normalize float image to uint8 using percentile clipping.
    Good for heatmap-like values.
    """
    x = np.asarray(x, dtype=np.float32)
    vmax = np.percentile(x, percentile)
    vmax = max(vmax, 1e-6)
    y = np.clip(x / vmax * 255.0, 0, 255)
    return y.astype(np.uint8)


def apply_colormap_u8(gray_u8, colormap=cv2.COLORMAP_INFERNO):
    """
    Input gray uint8, output BGR color image.
    """
    return cv2.applyColorMap(gray_u8, colormap)


def y10_to_u8(y10):
    """
    Convert 10-bit Y plane 0~1023 to uint8 0~255 for visualization.
    """
    y10 = np.asarray(y10, dtype=np.float32)
    return np.clip(y10 / 1023.0 * 255.0, 0, 255).astype(np.uint8)


def resize_keep_aspect_by_width(img, target_w=960):
    """
    Resize image by target width while keeping aspect ratio.
    Useful for making 4K/1080p inputs easy to inspect.
    """
    h, w = img.shape[:2]
    target_h = int(round(h * target_w / w))
    return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)


def flow_to_hsv_bgr(flow, clip_percentile=99):
    """
    Visualize optical flow as HSV color image.
    Hue        : direction
    Saturation : fixed 255
    Value      : normalized magnitude
    Output     : BGR image for cv2.imwrite
    """
    fx = flow[..., 0]
    fy = flow[..., 1]

    mag, ang = cv2.cartToPolar(fx, fy, angleInDegrees=False)

    vmax = np.percentile(mag, clip_percentile)
    vmax = max(vmax, 1e-6)

    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(mag / vmax * 255.0, 0, 255).astype(np.uint8)

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr, mag


def add_title_bar(img_bgr, title, bar_h=36):
    """
    Add a simple black title bar on top using OpenCV only.
    """
    if img_bgr.ndim == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)

    h, w = img_bgr.shape[:2]
    bar = np.zeros((bar_h, w, 3), dtype=np.uint8)
    cv2.putText(
        bar,
        title,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )
    return np.vstack([bar, img_bgr])


def save_img(path, img, title=None):
    """
    Save grayscale or BGR image.
    """
    out = img
    if title is not None:
        out = add_title_bar(out, title)
    cv2.imwrite(path, out)


def make_grid(items, cols=3, pad=8, bg=20):
    """
    items: list of BGR or grayscale images, already title-barred if wanted.
    Return BGR grid image.
    """
    bgr_items = []
    for img in items:
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        bgr_items.append(img)

    max_h = max(img.shape[0] for img in bgr_items)
    max_w = max(img.shape[1] for img in bgr_items)

    padded = []
    for img in bgr_items:
        h, w = img.shape[:2]
        canvas = np.full((max_h, max_w, 3), bg, dtype=np.uint8)
        canvas[:h, :w] = img
        padded.append(canvas)

    rows = int(np.ceil(len(padded) / cols))
    empty = np.full((max_h, max_w, 3), bg, dtype=np.uint8)

    row_imgs = []
    for r in range(rows):
        row = padded[r * cols:(r + 1) * cols]
        while len(row) < cols:
            row.append(empty.copy())
        row_imgs.append(np.hstack(row))

    grid = np.vstack(row_imgs)

    if pad > 0:
        # Add simple border padding around whole grid.
        grid = cv2.copyMakeBorder(
            grid, pad, pad, pad, pad,
            cv2.BORDER_CONSTANT,
            value=(bg, bg, bg)
        )

    return grid


def draw_flow_quiver(base_gray, flow, stride=32, scale=1.0):
    """
    Draw sparse flow arrows on grayscale image using cv2.arrowedLine.
    """
    if base_gray.ndim == 2:
        vis = cv2.cvtColor(base_gray, cv2.COLOR_GRAY2BGR)
    else:
        vis = base_gray.copy()

    H, W = base_gray.shape[:2]

    for y in range(stride // 2, H, stride):
        for x in range(stride // 2, W, stride):
            u = float(flow[y, x, 0]) * scale
            v = float(flow[y, x, 1]) * scale

            x2 = int(round(x + u))
            y2 = int(round(y + v))

            if 0 <= x2 < W and 0 <= y2 < H:
                cv2.arrowedLine(
                    vis,
                    (x, y),
                    (x2, y2),
                    (0, 255, 0),
                    1,
                    tipLength=0.35
                )

    return vis


def visualize_one_case():
    # ============================================================
    # 你只需要改這裡：選一組 task 和 frame index
    # ============================================================
    task = {
        "name": "AMS05",
        "qp": 32,
        "base": "../bitstream/base/odd_H2_H3_AMS05_32_0_5.layer0.yuv",
        "even": "../bitstream/enhance/even_H2_H3_AMS05_32_0_5.layer1.yuv",
    }

    frame_idx = 0
    out_dir = "./flow_visualize_output"
    os.makedirs(out_dir, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Task: {task['name']} QP{task['qp']}, frame={frame_idx}")

    if not os.path.exists(task["base"]):
        raise FileNotFoundError(f"Base YUV not found: {task['base']}")

    if not os.path.exists(task["even"]):
        raise FileNotFoundError(f"Even YUV not found: {task['even']}")

    # ============================================================
    # 1. Read one base odd frame and two even 4K frames
    # ============================================================
    f_base = getOneFrame(task["base"], 1920, 1080, 2, frame_idx)
    f_prev = getOneFrame(task["even"], 3840, 2160, 2, frame_idx)
    f_next = getOneFrame(task["even"], 3840, 2160, 2, frame_idx + 1)

    # ============================================================
    # Raw input Y images, before tensor conversion / upscale / flow
    # ============================================================
    base_y_raw = f_base["y"].reshape(1080, 1920)
    prev_y_raw = f_prev["y"].reshape(2160, 3840)
    next_y_raw = f_next["y"].reshape(2160, 3840)

    base_y_u8 = y10_to_u8(base_y_raw)
    prev_y_u8 = y10_to_u8(prev_y_raw)
    next_y_u8 = y10_to_u8(next_y_raw)

    # For grid preview, make them all width 960.
    base_y_preview = resize_keep_aspect_by_width(base_y_u8, 960)  # 960x540
    prev_y_preview = resize_keep_aspect_by_width(prev_y_u8, 960)  # 960x540
    next_y_preview = resize_keep_aspect_by_width(next_y_u8, 960)  # 960x540

    # ============================================================
    # 2. Convert Y channel to torch tensor, normalize to 0~1
    # ============================================================
    with torch.no_grad():
        t_base_y = torch.tensor(
            base_y_raw,
            dtype=torch.float32,
            device=device
        ).view(1, 1, 1080, 1920) / 1023.0

        t_p_y = torch.tensor(
            prev_y_raw,
            dtype=torch.float32,
            device=device
        ).view(1, 1, 2160, 3840) / 1023.0

        t_n_y = torch.tensor(
            next_y_raw,
            dtype=torch.float32,
            device=device
        ).view(1, 1, 2160, 3840) / 1023.0

        # 1080p base -> 4K upscaled base
        t_up_y = gpu_8tap_upscale_y(t_base_y)

        # Convert three 4K tensors into 960x540 uint8 images for flow
        up_8b = get_small_flow_img(t_up_y)
        p_8b = get_small_flow_img(t_p_y)
        n_8b = get_small_flow_img(t_n_y)

    # ============================================================
    # 3. Optical flow
    # ============================================================
    print("Computing optical flow...")

    flow_p = cv2.calcOpticalFlowFarneback(
        up_8b, p_8b, None,
        0.5, 5, 25, 3, 5, 1.2, 0
    )

    flow_n = cv2.calcOpticalFlowFarneback(
        up_8b, n_8b, None,
        0.5, 5, 25, 3, 5, 1.2, 0
    )

    flow_pb = cv2.calcOpticalFlowFarneback(
        p_8b, up_8b, None,
        0.5, 5, 25, 3, 5, 1.2, 0
    )

    flow_nb = cv2.calcOpticalFlowFarneback(
        n_8b, up_8b, None,
        0.5, 5, 25, 3, 5, 1.2, 0
    )

    # ============================================================
    # 4. Epipolar error maps
    # ============================================================
    print("Computing epipolar error maps...")

    epi_p, F_p = compute_epipolar_cpu(flow_p)
    epi_n, F_n = compute_epipolar_cpu(flow_n)

    # ============================================================
    # 5. Visualization images
    # ============================================================
    flow_p_bgr, mag_p = flow_to_hsv_bgr(flow_p)
    flow_n_bgr, mag_n = flow_to_hsv_bgr(flow_n)
    flow_pb_bgr, mag_pb = flow_to_hsv_bgr(flow_pb)
    flow_nb_bgr, mag_nb = flow_to_hsv_bgr(flow_nb)

    diff_up_prev = cv2.absdiff(up_8b, p_8b)
    diff_up_next = cv2.absdiff(up_8b, n_8b)

    # Simple diagnostic only. Real FB consistency should sample backward flow at x + forward_flow.
    fb_err_p = np.sqrt(
        (flow_p[..., 0] + flow_pb[..., 0]) ** 2 +
        (flow_p[..., 1] + flow_pb[..., 1]) ** 2
    )

    fb_err_n = np.sqrt(
        (flow_n[..., 0] + flow_nb[..., 0]) ** 2 +
        (flow_n[..., 1] + flow_nb[..., 1]) ** 2
    )

    mag_p_color = apply_colormap_u8(normalize_to_u8(mag_p))
    mag_n_color = apply_colormap_u8(normalize_to_u8(mag_n))
    epi_p_color = apply_colormap_u8(normalize_to_u8(epi_p.squeeze()))
    epi_n_color = apply_colormap_u8(normalize_to_u8(epi_n.squeeze()))
    fb_p_color = apply_colormap_u8(normalize_to_u8(fb_err_p))
    fb_n_color = apply_colormap_u8(normalize_to_u8(fb_err_n))

    quiver_p = draw_flow_quiver(up_8b, flow_p, stride=32, scale=1.0)
    quiver_n = draw_flow_quiver(up_8b, flow_n, stride=32, scale=1.0)

    # ============================================================
    # 6. Save visualized outputs
    # ============================================================
    print(f"Saving visualization results to: {out_dir}")

    # ============================================================
    # 6-1. Save raw input images before any upscale/downsample flow preprocessing
    # ============================================================
    save_img(
        os.path.join(out_dir, "input_00_base_odd_1080p_y.png"),
        base_y_u8,
        "RAW input: base odd Y, 1920x1080"
    )

    save_img(
        os.path.join(out_dir, "input_01_prev_even_4k_y.png"),
        prev_y_u8,
        "RAW input: prev/current even Y, 3840x2160"
    )

    save_img(
        os.path.join(out_dir, "input_02_next_even_4k_y.png"),
        next_y_u8,
        "RAW input: next even Y, 3840x2160"
    )

    raw_input_grid = make_grid(
        [
            add_title_bar(base_y_preview, "RAW base odd Y resized preview"),
            add_title_bar(prev_y_preview, "RAW prev/current even Y resized preview"),
            add_title_bar(next_y_preview, "RAW next even Y resized preview"),
        ],
        cols=3
    )
    cv2.imwrite(os.path.join(out_dir, "input_03_raw_inputs_grid.png"), raw_input_grid)

    # Difference of raw input previews at the same preview size.
    raw_diff_base_prev = cv2.absdiff(base_y_preview, prev_y_preview)
    raw_diff_base_next = cv2.absdiff(base_y_preview, next_y_preview)

    raw_input_compare_grid = make_grid(
        [
            add_title_bar(base_y_preview, "RAW base odd preview"),
            add_title_bar(prev_y_preview, "RAW prev even preview"),
            add_title_bar(next_y_preview, "RAW next even preview"),
            add_title_bar(raw_diff_base_prev, "RAW abs(base-prev) preview"),
            add_title_bar(raw_diff_base_next, "RAW abs(base-next) preview"),
        ],
        cols=3
    )
    cv2.imwrite(os.path.join(out_dir, "input_04_raw_inputs_compare_grid.png"), raw_input_compare_grid)

    save_img(
        os.path.join(out_dir, "01_upscaled_base_small.png"),
        up_8b,
        "Upscaled base odd -> 960x540"
    )

    save_img(
        os.path.join(out_dir, "02_prev_even_small.png"),
        p_8b,
        "Prev/current even -> 960x540"
    )

    save_img(
        os.path.join(out_dir, "03_next_even_small.png"),
        n_8b,
        "Next even -> 960x540"
    )

    save_img(
        os.path.join(out_dir, "04_diff_base_prev.png"),
        diff_up_prev,
        "abs(base - prev)"
    )

    save_img(
        os.path.join(out_dir, "05_diff_base_next.png"),
        diff_up_next,
        "abs(base - next)"
    )

    save_img(
        os.path.join(out_dir, "06_flow_p_base_to_prev_hsv.png"),
        flow_p_bgr,
        "flow_p: base -> prev/current even"
    )

    save_img(
        os.path.join(out_dir, "07_flow_n_base_to_next_hsv.png"),
        flow_n_bgr,
        "flow_n: base -> next even"
    )

    save_img(
        os.path.join(out_dir, "08_flow_pb_prev_to_base_hsv.png"),
        flow_pb_bgr,
        "flow_pb: prev/current even -> base"
    )

    save_img(
        os.path.join(out_dir, "09_flow_nb_next_to_base_hsv.png"),
        flow_nb_bgr,
        "flow_nb: next even -> base"
    )

    save_img(
        os.path.join(out_dir, "10_flow_p_magnitude.png"),
        mag_p_color,
        "Magnitude of flow_p"
    )

    save_img(
        os.path.join(out_dir, "11_flow_n_magnitude.png"),
        mag_n_color,
        "Magnitude of flow_n"
    )

    save_img(
        os.path.join(out_dir, "12_epipolar_p_error.png"),
        epi_p_color,
        "Epipolar error from flow_p"
    )

    save_img(
        os.path.join(out_dir, "13_epipolar_n_error.png"),
        epi_n_color,
        "Epipolar error from flow_n"
    )

    save_img(
        os.path.join(out_dir, "14_forward_backward_error_p.png"),
        fb_p_color,
        "Simple FB error: flow_p + flow_pb"
    )

    save_img(
        os.path.join(out_dir, "15_forward_backward_error_n.png"),
        fb_n_color,
        "Simple FB error: flow_n + flow_nb"
    )

    save_img(
        os.path.join(out_dir, "16_flow_p_quiver.png"),
        quiver_p,
        "Sparse arrows: flow_p over base"
    )

    save_img(
        os.path.join(out_dir, "17_flow_n_quiver.png"),
        quiver_n,
        "Sparse arrows: flow_n over base"
    )

    grid_items = [
        add_title_bar(base_y_preview, "RAW base odd preview"),
        add_title_bar(prev_y_preview, "RAW prev even preview"),
        add_title_bar(next_y_preview, "RAW next even preview"),
        add_title_bar(up_8b, "upscaled base -> small"),
        add_title_bar(p_8b, "prev/current even -> small"),
        add_title_bar(n_8b, "next even -> small"),
        add_title_bar(diff_up_prev, "abs(base-prev)"),
        add_title_bar(diff_up_next, "abs(base-next)"),
        add_title_bar(flow_p_bgr, "flow_p HSV"),
        add_title_bar(flow_n_bgr, "flow_n HSV"),
        add_title_bar(mag_p_color, "mag flow_p"),
        add_title_bar(mag_n_color, "mag flow_n"),
        add_title_bar(epi_p_color, "epipolar p"),
        add_title_bar(epi_n_color, "epipolar n"),
        add_title_bar(quiver_p, "quiver flow_p"),
    ]

    grid = make_grid(grid_items, cols=3)
    cv2.imwrite(os.path.join(out_dir, "00_all_visualizations_grid.png"), grid)

    # Save a small text report
    report_path = os.path.join(out_dir, "summary.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Task: {task['name']} QP{task['qp']}\n")
        f.write(f"Frame index: {frame_idx}\n")
        f.write(f"Device: {device}\n\n")

        f.write("Shapes:\n")
        f.write(f"  raw base_y: {base_y_raw.shape}\n")
        f.write(f"  raw prev_y: {prev_y_raw.shape}\n")
        f.write(f"  raw next_y: {next_y_raw.shape}\n")
        f.write(f"  up_8b: {up_8b.shape}\n")
        f.write(f"  p_8b : {p_8b.shape}\n")
        f.write(f"  n_8b : {n_8b.shape}\n")
        f.write(f"  flow_p : {flow_p.shape}\n")
        f.write(f"  flow_n : {flow_n.shape}\n")
        f.write(f"  flow_pb: {flow_pb.shape}\n")
        f.write(f"  flow_nb: {flow_nb.shape}\n")
        f.write(f"  epi_p  : {epi_p.shape}\n")
        f.write(f"  epi_n  : {epi_n.shape}\n\n")

        f.write("Magnitude statistics:\n")
        for name, mag in [
            ("flow_p", mag_p),
            ("flow_n", mag_n),
            ("flow_pb", mag_pb),
            ("flow_nb", mag_nb),
        ]:
            f.write(
                f"  {name}: min={mag.min():.4f}, "
                f"mean={mag.mean():.4f}, "
                f"max={mag.max():.4f}, "
                f"p99={np.percentile(mag, 99):.4f}\n"
            )

        f.write("\nEpipolar statistics:\n")
        f.write(
            f"  epi_p: min={epi_p.min():.4f}, "
            f"mean={epi_p.mean():.4f}, "
            f"max={epi_p.max():.4f}\n"
        )
        f.write(
            f"  epi_n: min={epi_n.min():.4f}, "
            f"mean={epi_n.mean():.4f}, "
            f"max={epi_n.max():.4f}\n"
        )

        f.write("\nFundamental matrix F_p:\n")
        f.write(str(F_p) + "\n")
        f.write("\nFundamental matrix F_n:\n")
        f.write(str(F_n) + "\n")

    print("Done.")
    print(f"Open this folder to inspect results: {out_dir}")


if __name__ == "__main__":
    visualize_one_case()
