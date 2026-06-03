#!/bin/bash

set -e

# 目前位置應該在 generated 資料夾
INPUT_DIR="."
OUTPUT_DIR="../generated_mp4"

mkdir -p "$OUTPUT_DIR"

# -----------------------------
# ZombieClimbing2: 24 fps, CRF 28
# -----------------------------
for qp in 27 32 37 42; do
    input="odd_ZombieClimbing2_${qp}_0_4_gen.layer0.yuv"
    output="${OUTPUT_DIR}/odd_ZombieClimbing2_${qp}_0_4_gen_fps24.mp4"

    echo "Converting $input -> $output"

    ffmpeg -y \
        -f rawvideo -vcodec rawvideo \
        -s 3840x2160 -r 24 \
        -pix_fmt yuv420p10le \
        -i "$input" \
        -c:v libx265 -preset slow -crf 28 \
        -pix_fmt yuv420p10le \
        "$output"
done

# -----------------------------
# AMS05: 60 fps, CRF 25
# -----------------------------
for qp in 27 32 37 42; do
    input="odd_H2_H3_AMS05_${qp}_0_5_gen.layer0.yuv"
    output="${OUTPUT_DIR}/odd_H2_H3_AMS05_${qp}_0_5_gen_fps60.mp4"

    echo "Converting $input -> $output"

    ffmpeg -y \
        -f rawvideo -vcodec rawvideo \
        -s 3840x2160 -r 60 \
        -pix_fmt yuv420p10le \
        -i "$input" \
        -c:v libx265 -preset slow -crf 25 \
        -pix_fmt yuv420p10le \
        "$output"
done

# -----------------------------
# WalkInPark: 60 fps, CRF 25
# -----------------------------
for qp in 27 32 37 42; do
    input="odd_H2_WalkInPark_${qp}_0_4_gen.layer0.yuv"
    output="${OUTPUT_DIR}/odd_H2_WalkInPark_${qp}_0_4_gen_fps60.mp4"

    echo "Converting $input -> $output"

    ffmpeg -y \
        -f rawvideo -vcodec rawvideo \
        -s 3840x2160 -r 60 \
        -pix_fmt yuv420p10le \
        -i "$input" \
        -c:v libx265 -preset slow -crf 25 \
        -pix_fmt yuv420p10le \
        "$output"
done

# -----------------------------
# Procession: 60 fps, CRF 25
# 注意這組 QP 是 25, 30, 35, 40
# -----------------------------
for qp in 25 30 35 40; do
    input="odd_Procession_${qp}_0_4_gen.layer0.yuv"
    output="${OUTPUT_DIR}/odd_Procession_${qp}_0_4_gen_fps60.mp4"

    echo "Converting $input -> $output"

    ffmpeg -y \
        -f rawvideo -vcodec rawvideo \
        -s 3840x2160 -r 60 \
        -pix_fmt yuv420p10le \
        -i "$input" \
        -c:v libx265 -preset slow -crf 25 \
        -pix_fmt yuv420p10le \
        "$output"
done

echo "All conversions finished."
echo "MP4 files are saved in: $OUTPUT_DIR"