#!/bin/bash
# Evaluation script for both models

echo "=========================================="
echo "VOC 2012 Evaluation: YOLOX vs YOLOv5"
echo "=========================================="

# YOLOX Evaluation
echo "[1/2] Evaluating YOLOX..."
python3 src/eval_voc.py \
    --model_path models/yolox_tiny.onnx \
    --model_type yolox \
    --data_dir data/VOC2012 \
    --output results/yolox_voc_results.json

# YOLOv5 Evaluation  
echo ""
echo "[2/2] Evaluating YOLOv5..."
python3 src/eval_voc.py \
    --model_path models/yolov5nu.onnx \
    --model_type yolov5 \
    --data_dir data/VOC2012 \
    --output results/yolov5_voc_results.json

echo ""
echo "Evaluation complete!"
echo "Results saved to results/"
