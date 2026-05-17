# Temporal Super-Resolution for 4K YUV Reconstruction

This project implements a Temporal Super-Resolution (TSR) pipeline for reconstructing missing 4K odd frames from:

- a low-resolution 1080p Base Layer
- neighboring 4K Enhance Layer frames

The system combines:

1. Spatial Super-Resolution
2. Motion Estimation
3. Motion Compensation
4. Adaptive Temporal Fusion

to generate high-quality reconstructed 4K frames.

The implementation is inspired by concepts used in:

- HEVC / H.265
- VVC / H.266
- Motion-Compensated Frame Interpolation
- Video Super-Resolution

---

# Overall Pipeline

```text
            Base Layer (1080p Odd Frames)
                         │
                         ▼
              Spatial Upscaling (2×)
        ┌────────────────────────────────┐
        │  8-tap / 4-tap FIR Interp.    │
        │  Phase-aligned interpolation  │
        └────────────────────────────────┘
                         │
                         ▼
                Upscaled 4K Frame
                         │
                         │
        ┌────────────────┴────────────────┐
        ▼                                 ▼
 Previous 4K Even Frame          Next 4K Even Frame
        │                                 │
        └──────── Optical Flow ───────────┘
                         │
                         ▼
                 Motion Compensation
                         │
                         ▼
                 Adaptive Fusion
                         │
                         ▼
              Reconstructed 4K Odd Frame
```

---

# Input / Output Format

## Input

### Base Layer

- Resolution: `1920 × 1080`
- Format: `YUV420 10-bit`
- Contains odd frames only

### Enhance Layer

- Resolution: `3840 × 2160`
- Format: `YUV420 10-bit`
- Contains even frames only

---

## Output

### Generated Sequence

- Resolution: `3840 × 2160`
- Format: `YUV420 10-bit`
- Reconstructed odd frames

---

# Spatial Super-Resolution

The first stage reconstructs a high-resolution 4K image from a 1080p frame.

The problem can be written as:

$$
I_{HR} = \mathcal{U}(I_{LR})
$$

where:

- $I_{LR}$ = low-resolution image
- $I_{HR}$ = reconstructed high-resolution image
- $\mathcal{U}$ = upsampling operator

---

# Why OpenCV Resize Was Not Used

Naive interpolation methods like:

- bilinear
- bicubic
- Lanczos

introduce:

- sub-pixel phase shifts
- interpolation misalignment

This becomes problematic for:

- PSNR
- motion estimation
- temporal fusion

because reconstructed pixels are no longer perfectly aligned with original sampling locations.

---

# Zero-Phase Alignment

The implemented interpolation guarantees:

```text
Upscaled even coordinates == original pixels
```

Meaning:

$$
I_{HR}(2x,2y) = I_{LR}(x,y)
$$

This is critical for:

- codec consistency
- motion compensation stability
- temporal coherence

---

# Linear FIR Interpolation

The simplest interpolation is:

$$
I(x+\frac12)
=
\frac12 I(x)
+
\frac12 I(x+1)
$$

This corresponds to the FIR kernel:

```text
[ 0.5  0.5 ]
```

Implemented in:

```python
phase_aligned_upscale_2x()
```

---

# 4-Tap Bicubic FIR Filter

The project uses a Catmull-Rom style cubic interpolation filter:

$$
\left[
-\frac1{16},
\frac9{16},
\frac9{16},
-\frac1{16}
\right]
$$

Implemented as:

```python
phase_aligned_4tap_upscale_2x()
```

The interpolated pixel is:

$$
p'
=
-\frac1{16}p_0
+
\frac9{16}p_1
+
\frac9{16}p_2
-
\frac1{16}p_3
$$

Compared to bilinear interpolation:

- preserves edges better
- reduces blur
- improves high-frequency reconstruction

because the kernel approximates a cubic spline response.

---

# VVC / HEVC 8-Tap DCT Interpolation Filter

The main luminance upsampler uses the VVC/HEVC-style 8-tap interpolation kernel:

$$
[-1,\ 4,\ -11,\ 40,\ 40,\ -11,\ 4,\ -1]/64
$$

Implemented in:

```python
phase_aligned_8tap_dctif_2x()
```

The interpolated pixel is:

$$
p'
=
\frac{
-1p_0
+4p_1
-11p_2
+40p_3
+40p_4
-11p_5
+4p_6
-1p_7
}{64}
$$

This filter is derived from:

- sinc interpolation approximation
- DCT-domain reconstruction theory

The goal is to approximate an ideal low-pass reconstruction filter.

Advantages:

- sharper edges
- reduced aliasing
- better frequency preservation
- codec-grade interpolation quality

This is similar to interpolation filters used in:

- HEVC fractional motion compensation
- VVC interpolation
- sub-pixel prediction

---

# Separable Filtering

2D interpolation is implemented as:

1. horizontal filtering
2. vertical filtering

This reduces complexity from:

$$
O(N^2)
$$

to:

$$
O(2N)
$$

for each pixel neighborhood.

---

# Boundary Handling

The implementation uses:

```python
np.pad(..., mode='reflect')
```

This mirrors boundary pixels:

```text
ABC|CBA
```

instead of zero padding.

Benefits:

- avoids ringing artifacts
- improves edge continuity
- reduces boundary distortion

---

# Temporal Super-Resolution

Spatial upsampling alone cannot recover all lost details.

Therefore temporal information from neighboring frames is used.

The model assumes:

```text
neighboring frames contain missing spatial information
```

This is the key idea behind:

- video super-resolution
- motion-compensated interpolation
- frame synthesis

---

# Optical Flow Estimation

The implementation uses:

```python
cv2.calcOpticalFlowFarneback()
```

to estimate motion between frames.

---

# Optical Flow Theory

Optical flow estimates motion vectors:

$$
(u,v)
$$

between two frames under the brightness constancy assumption:

$$
I(x,y,t)
=
I(x+u,y+v,t+1)
$$

Using Taylor expansion:

$$
I_x u + I_y v + I_t = 0
$$

where:

- $I_x$ = spatial x-gradient
- $I_y$ = spatial y-gradient
- $I_t$ = temporal gradient

---

# Farneback Optical Flow

Farneback approximates local neighborhoods using quadratic polynomials:

$$
f(x)
=
x^T A x + b^T x + c
$$

Motion is estimated by matching polynomial coefficients.

Advantages:

- dense optical flow
- relatively fast
- robust for smooth motion

---

# Multi-Scale Flow Estimation

To accelerate computation:

```python
cv2.resize(..., INTER_AREA)
```

reduces the frame to 1080p before flow estimation.

The flow field is then upscaled back to 4K.

This significantly reduces CPU cost.

---

# Motion Compensation

Using optical flow:

$$
(x,y)
\rightarrow
(x+u,y+v)
$$

neighboring frames are warped into the current frame coordinate system.

Implemented using:

```python
cv2.remap()
```

This produces:

- `warped_prev`
- `warped_next`

Without motion compensation:

```text
temporal averaging → ghosting
```

Motion alignment ensures:

- moving objects overlap correctly
- details reinforce each other

instead of blurring.

---

# Adaptive Temporal Fusion

The final reconstruction blends:

- spatially upscaled frame
- warped previous frame
- warped next frame

using confidence-based weights.

---

# Weight Computation

Differences are computed as:

$$
d_{prev}
=
|I_{warp-prev} - I_{up}|
$$

$$
d_{next}
=
|I_{warp-next} - I_{up}|
$$

Weights are:

$$
w
=
e^{-d/\sigma}
$$

where:

- smaller difference → larger confidence
- larger difference → lower confidence

This behaves similarly to a bilateral weighting function.

---

# Final Reconstruction Formula

The reconstructed pixel is:

$$
I_{final}
=
\frac{
w_p I_p
+
w_n I_n
+
w_u I_u
}{
w_p + w_n + w_u
}
$$

where:

- $I_p$ = warped previous frame
- $I_n$ = warped next frame
- $I_u$ = upscaled frame

This suppresses:

- optical flow errors
- occlusion artifacts
- motion mismatch

while still exploiting temporal detail.

---

# Chroma Processing

The chroma channels:

- U
- V

are upscaled separately using:

```python
phase_aligned_4tap_upscale_2x()
```

Reason:

- chroma contains lower-frequency information
- 8-tap filtering is less beneficial
- reduces computation

---

# Complexity Analysis

## Spatial Upscaling

For each interpolated pixel:

- 4-tap:
  - 4 multiply-adds
- 8-tap:
  - 8 multiply-adds

Complexity:

$$
O(HW)
$$

---

## Optical Flow

Farneback complexity is approximately:

$$
O(HW \cdot L)
$$

where:

- $L$ = pyramid levels

This is the dominant computational cost.

---

# Why Only the Y Channel Uses TSR

Human vision is more sensitive to luminance.

Therefore:

- Y channel:
  - temporal super-resolution
- U/V channels:
  - spatial interpolation only

This is standard in many codecs and video processing systems.

---

# Experimental Observations

| Method      | Result              |
| ----------- | ------------------- |
| Bilinear    | blurry              |
| Bicubic     | sharper             |
| Lanczos     | ringing artifacts   |
| 4-tap FIR   | stable              |
| 8-tap DCTIF | best spatial detail |

Temporal fusion further improves:

- texture recovery
- edge sharpness
- temporal consistency

---

# Key Design Goals

This implementation prioritizes:

- phase correctness
- codec-style interpolation
- temporal consistency
- PSNR preservation
- computational practicality

---

# Future Improvements

## 1. GPU Optical Flow

Potential replacements:

- CUDA Farneback
- NVIDIA Optical Flow SDK
- RAFT

---

## 2. Deep Learning Video SR

Possible models:

- EDVR
- BasicVSR++
- RVRT
- VRT

---

## 3. Better Motion Models

Current model assumes:

- translational flow only

Future work:

- occlusion handling
- deformable alignment
- confidence maps

---

## 4. Chroma Temporal SR

Apply temporal fusion to:

- U
- V

channels as well.

---

# Dependencies

```bash
pip install numpy opencv-python
```

---

# Running the Program

```bash
python main.py
```

---

# Output Example

```text
odd_ZombieClimbing2_27_0_4_gen.layer0.yuv
```

---

# References

## Video Coding

- HEVC (H.265)
- VVC (H.266)

## Optical Flow

- Farnebäck, G. (2003)
  "Two-Frame Motion Estimation Based on Polynomial Expansion"

## Super-Resolution

- Video Super Resolution literature
- Motion Compensated Frame Interpolation
- Temporal Reconstruction methods
