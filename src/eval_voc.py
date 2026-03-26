import os
import sys
import json
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm

from detector import YOLODetector, COCO_TO_VOC, VOC_CLASSES


def parse_voc_annotation(xml_path: str) -> List[Tuple]:
    """Parse Pascal VOC XML annotation.
    Returns: list of (class_name, x1, y1, x2, y2)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    objects = []
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        bbox = obj.find('bndbox')
        x1 = float(bbox.find('xmin').text)
        y1 = float(bbox.find('ymin').text)
        x2 = float(bbox.find('xmax').text)
        y2 = float(bbox.find('ymax').text)
        objects.append((class_name, x1, y1, x2, y2))
    
    return objects


def voc_class_to_id(class_name: str) -> int:
    return VOC_CLASSES.index(class_name)


def compute_iou(box1: List[float], box2: List[float]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / (union + 1e-16) if union > 0 else 0


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Compute AP with 11-point interpolation (VOC style) or 101-point (COCO)."""
    # 11-point interpolation
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11.0
    return ap


def evaluate_voc(detector: YOLODetector, 
                 image_dir: str, 
                 anno_dir: str, 
                 iou_thresh: float = 0.5) -> Dict:
    """Evaluate detector on VOC dataset."""
    
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
    
    # Store predictions and ground truths per class
    predictions = defaultdict(list)  # class_id -> list of (image_id, score, bbox)
    ground_truths = defaultdict(list)  # class_id -> list of (image_id, bbox, used)
    
    print(f"Evaluating on {len(image_files)} images...")
    
    for img_id, img_file in enumerate(tqdm(image_files)):
        img_path = os.path.join(image_dir, img_file)
        anno_path = os.path.join(anno_dir, img_file.replace('.jpg', '.xml').replace('.png', '.xml'))
        
        if not os.path.exists(anno_path):
            continue
        
        # Ground truth
        gt_objects = parse_voc_annotation(anno_path)
        for class_name, x1, y1, x2, y2 in gt_objects:
            class_id = voc_class_to_id(class_name)
            ground_truths[class_id].append({
                'image_id': img_id,
                'bbox': [x1, y1, x2, y2],
                'used': False
            })
        
        # Predictions
        dets, _ = detector.predict(img_path)
        
        # Filter only VOC classes and map COCO indices to VOC
        for det in dets:
            x1, y1, x2, y2, score, coco_class_id = det
            coco_class_id = int(coco_class_id)
            
            if coco_class_id in COCO_TO_VOC:
                voc_class_id = COCO_TO_VOC[coco_class_id]
                predictions[voc_class_id].append({
                    'image_id': img_id,
                    'score': float(score),
                    'bbox': [float(x1), float(y1), float(x2), float(y2)]
                })
    
    # Calculate AP for each class
    aps = {}
    for class_id in range(20):
        cls_dets = predictions[class_id]
        cls_gts = ground_truths[class_id]
        
        if len(cls_gts) == 0:
            continue
        
        # Sort by score descending
        cls_dets = sorted(cls_dets, key=lambda x: x['score'], reverse=True)
        
        tp = np.zeros(len(cls_dets))
        fp = np.zeros(len(cls_dets))
        
        # Create dict: image_id -> list of GT indices
        gt_by_image = defaultdict(list)
        for i, gt in enumerate(cls_gts):
            gt_by_image[gt['image_id']].append(i)
        
        for i, det in enumerate(cls_dets):
            img_id = det['image_id']
            det_bbox = det['bbox']
            
            max_iou = 0
            max_gt_idx = -1
            
            for gt_idx in gt_by_image[img_id]:
                gt_bbox = cls_gts[gt_idx]['bbox']
                iou = compute_iou(det_bbox, gt_bbox)
                if iou > max_iou:
                    max_iou = iou
                    max_gt_idx = gt_idx
            
            if max_iou >= iou_thresh and max_gt_idx != -1 and not cls_gts[max_gt_idx]['used']:
                tp[i] = 1
                cls_gts[max_gt_idx]['used'] = True
            else:
                fp[i] = 1
        
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recalls = tp_cumsum / len(cls_gts)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-16)
        
        ap = compute_ap(recalls, precisions)
        aps[class_id] = {
            'ap': ap,
            'num_gt': len(cls_gts),
            'num_detections': int(np.sum(tp))
        }
    
    # Calculate mAP
    if len(aps) > 0:
        mean_ap = np.mean([v['ap'] for v in aps.values()])
    else:
        mean_ap = 0.0
    
    return {
        'mAP@0.5': mean_ap,
        'per_class': {VOC_CLASSES[k]: v for k, v in aps.items()},
        'evaluated_classes': len(aps)
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate YOLO on VOC')
    parser.add_argument('--model_path', required=True, help='Path to ONNX model')
    parser.add_argument('--model_type', choices=['yolox', 'yolov5'], required=True)
    parser.add_argument('--data_dir', default='data/VOC2012', 
                       help='Path to VOC2012 root')
    parser.add_argument('--split', default='val', choices=['train', 'val'])
    parser.add_argument('--input_size', type=int, default=416)
    parser.add_argument('--conf_thres', type=float, default=0.3)
    parser.add_argument('--output', default='results/voc_results.json')
    
    args = parser.parse_args()
    
    image_dir = os.path.join(args.data_dir, 'JPEGImages')
    anno_dir = os.path.join(args.data_dir, 'Annotations')
    
    detector = YOLODetector(
        model_path=args.model_path,
        model_type=args.model_type,
        input_size=args.input_size,
        conf_thres=args.conf_thres,
        nms_thres=0.45
    )
    
    results = evaluate_voc(detector, image_dir, anno_dir)
    
    # Add metadata
    results['model'] = args.model_type
    results['model_path'] = args.model_path
    results['input_size'] = args.input_size
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*50)
    print(f"RESULTS FOR {args.model_type.upper()}")
    print("="*50)
    print(f"mAP@0.5: {results['mAP@0.5']:.4f} ({results['mAP@0.5']*100:.2f}%)")
    print("\nPer-class AP:")
    for cls_name, data in sorted(results['per_class'].items()):
        print(f"  {cls_name:12s}: {data['ap']:.4f} ({data['num_detections']:4d}/{data['num_gt']:4d})")
    
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()