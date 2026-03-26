#!/bin/bash
# Speed benchmark for both models

echo "=========================================="
echo "Speed Benchmark: YOLOX vs YOLOv5"
echo "=========================================="

# YOLOX Benchmark
echo "[1/2] Benchmarking YOLOX..."
python src/benchmark.py \
    --model_path models/yolox_tiny.onnx \
    --model_type yolox \
    --num_runs 100 \
    --output results/yolox_benchmark.json

# YOLOv5 Benchmark
echo ""
echo "[2/2] Benchmarking YOLOv5..."
python src/benchmark.py \
    --model_path models/yolov5nu.onnx \
    --model_type yolov5 \
    --num_runs 100 \
    --output results/yolov5_benchmark.json

echo ""
echo "Benchmark complete!"
