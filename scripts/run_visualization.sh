#!/bin/bash
# Generate all visualizations

echo "=========================================="
echo "Generating Visualizations"
echo "=========================================="

python src/visualize.py \
    --yolox_eval results/yolox_voc_results.json \
    --yolov5_eval results/yolov5_voc_results.json \
    --yolox_bench results/yolox_benchmark.json \
    --yolov5_bench results/yolov5_benchmark.json \
    --output_dir results/figures

echo ""
echo "Visualizations saved to results/figures/"
