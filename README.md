# Frame Generation Project

## 🚀 Quick Start Guide

### Prerequisites

- Linux/WSL environment with FFmpeg installed
- Python 3.x with pip

### Installation

#### FFmpeg

```bash
sudo apt update
sudo apt install ffmpeg
```

#### YUView 2.13 (Optional - for YUV video visualization)

YUView is a useful tool for viewing and analyzing YUV video files during development and testing.

**Linux (via Flatpak):**

```bash
flatpak install https://github.com/IENT/YUView/releases/download/v2.13/YUView.flatpak
```

**Manual Installation (All Platforms):**

1. Visit [YUView 2.13 Releases](https://github.com/IENT/YUView/releases/tag/v2.13)
2. Download the appropriate version for your platform:
   - **Linux**: `YUView-Linux.AppImage` or `YUView.flatpak`
   - **Windows**: `YUView-Win.zip` or `YUViewSetup.msi`
   - **macOS**: `YUView-Mac*.zip` (choose your macOS version)
3. Extract or install the application
4. For AppImage on Linux, make it executable:
   ```bash
   chmod +x YUView-Linux.AppImage
   ./YUView-Linux.AppImage
   ```

## 📋 Workflow

### Step 1: Extract Source Videos

Extract the original YUV video files to the `orgYUV/` folder, navigate to project directory and extract source videos:

```bash
unzip orgYUV.zip
```

### Step 2: Decode and Upscale Videos

Navigate to `bitstream/videoBin/` and execute the decoding and upscaling script:

```bash
cd bitstream/videoBin
bash doDec_upscale.sh
```

This generates upscaled (1080p to 4K) frames in `bitstream/upscaled/`.

### Step 3: Decode Base and Enhancement Layers

```bash
bash doDec.sh
```

This generates:

- Even 4K frames in `bitstream/enhance/`
- Odd 1080p frames in `bitstream/base/`

### Step 4: Evaluate Optical Flow-Based Super Resolution

Navigate to the `scriptTest/` folder and set up the Python environment:

```bash
cd scriptTest
python -m venv fgopencv  # or use any virtual environment manager
source fgopencv/bin/activate  # On Windows: fgopencv\Scripts\activate
pip install -r requirements.txt
```

Generate interpolated frames:

```bash
python generateAllFramesFromBase.py
# Alternative: python generateAllFrames.py (requires bitstream/upscaled files)
```

Evaluate the results for each test sequence:

```bash
python testOddFrames{Subject}.py
```

Where `{Subject}` is one of: `AMS05`, `WalkInPark`, `Procession`, or `Zombie`

### Step 5: Obtain BD-Rate Results

Results are recorded in `memo.txt` (negative BD-rate values indicate improved encoding efficiency)

## � Directory Structure

### Root Files

| File/Folder                                        | Description                                                                          |
| -------------------------------------------------- | ------------------------------------------------------------------------------------ |
| **memo.txt**                                       | BD-rate computation results (video encoding efficiency metrics) for 4 test sequences |
| **CV_2026_Final_Project_FrameGeneration.pdf/pptx** | Project documentation and presentation slides                                        |
| **CV final project extra information.pdf/pptx**    | Supplementary project information                                                    |
| **VVCMLinfo.xlsx**                                 | Video encoding parameters (VVC/H.266 related)                                        |
| **orgYUV.zip**                                     | Compressed archive of original YUV video files                                       |

### bitstream/ Folder - Encoded/Decoded Artifacts

### bitstream/ Folder - Encoded/Decoded Artifacts

- **base/** - Base layer bitstream files
- **enhance/** - Enhancement layer bitstream files
- **upscaled/** - Upscaled YUV video files (4K output from decoder)
- **generated/** - Generated/interpolated frame sequences
- **bin/** - Decoder executable (`DecoderAppStatic`)
- **videoBin/** - Video processing scripts
  - `doDec.sh` - Decoding script
  - `doDec_upscale.sh` - Decoding and upscaling script

### scriptTest/ Folder - Python Testing Tools

- **fgopencv/** - Python virtual environment (excluded from version control)
- **requirements.txt** - Python dependencies (numpy, opencv-python-headless)
- **yuvProc.py** - YUV video format processing module
- **generateAllFrames.py** - Frame generation script (uses upscaled files)
- **generateAllFramesFromBase.py** - Frame generation script (uses base layer)
- **testOddFrames\*.py** - Evaluation scripts for different test sequences:
  - `testOddFramesAMS05.py` - AMS05 sequence (H2_H3) evaluation
  - `testOddFramesWalkInPark.py` - WalkInPark sequence evaluation
  - `testOddFramesProcession.py` - Procession sequence evaluation
  - `testOddFramesZombie.py` - Zombie Climbing 2 sequence evaluation

### orgYUV/ Folder

- Original 4K YUV video source files for the 4 test sequences

## 🎯 Project Overview

This is a Computer Vision course final project focused on **video frame generation and interpolation using optical flow-based super resolution**. The project evaluates encoding efficiency using BD-rate metrics (negative BD-rate indicates improved encoding efficiency) across 4 standard test video sequences.
