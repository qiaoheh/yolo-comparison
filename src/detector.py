import numpy as np
import cv2
import onnxruntime as ort
import time
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from collections import defaultdict
import warnings

CLASS_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
    'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

# COCO to VOC mapping (class indices)
COCO_TO_VOC = {
    4: 0,   # aeroplane
    1: 1,   # bicycle
    14: 2,  # bird
    8: 3,   # boat
    39: 4,  # bottle
    5: 5,   # bus
    2: 6,   # car
    15: 7,  # cat
    56: 8,  # chair
    19: 9,  # cow
    60: 10, # diningtable
    16: 11, # dog
    17: 12, # horse
    3: 13,  # motorbike
    0: 14,  # person
    58: 15, # pottedplant
    18: 16, # sheep
    57: 17, # sofa
    6: 18,  # train
    62: 19  # tvmonitor
}


class YOLODetector:
    def __init__(self, model_path: str, model_type: str = 'yolox', input_size: int = 416,
                 conf_thres: float = 0.3, nms_thres: float = 0.45, 
                 providers: Optional[List[str]] = None):
        self.model_path = model_path
        self.model_type = model_type.lower()
        self.input_size = input_size
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
        self.num_classes = 80
        
        if providers is None:
            available = ort.get_available_providers()
            if 'CoreMLExecutionProvider' in available:
                providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']
            print(f"Using providers: {providers}")
        
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        
        self.strides = [8, 16, 32]
        self.anchors = [
            [[10, 13], [16, 30], [33, 23]],
            [[30, 61], [62, 45], [59, 119]],
            [[116, 90], [156, 198], [373, 326]]
        ]
        
        self.yolov5_format = None
        if self.model_type == 'yolov5':
            outputs = self.session.get_outputs()
            if len(outputs) == 3:
                self.yolov5_format = 'multi'
            elif len(outputs) == 1:
                self.yolov5_format = 'single'
            else:
                raise ValueError(f"Unsupported YOLOv5 output count: {len(outputs)}")
            print(f"[INFO] YOLOv5 format: {self.yolov5_format}")
        
        self._grids = None
        self._expanded_strides = None
    
    def preprocess(self, image: Union[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, tuple]:
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"Cannot load {image}")
        else:
            img = image.copy()
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_shape = img.shape[:2]
        h, w = original_shape
        
        scale = min(self.input_size / h, self.input_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        padded = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        padded[:new_h, :new_w] = resized
        
        padded = padded.astype(np.float32)
        if self.model_type == 'yolov5':
            padded = padded / 255.0
        
        padded = padded.transpose(2, 0, 1)
        input_tensor = np.expand_dims(padded, 0)
        
        return input_tensor, img, (scale, new_h, new_w)
    
    def _make_grid(self, nx: int, ny: int) -> np.ndarray:
        xv, yv = np.meshgrid(np.arange(nx), np.arange(ny))
        return np.stack((xv, yv), axis=2).reshape((1, 1, ny, nx, 2)).astype(np.float32)
    
    def _get_yolox_grids(self):
        if self._grids is None:
            grids = []
            expanded_strides = []
            for stride in self.strides:
                hsize = self.input_size // stride
                wsize = self.input_size // stride
                yv, xv = np.meshgrid(np.arange(hsize), np.arange(wsize), indexing='ij')
                grid = np.stack((xv, yv), axis=2).reshape(-1, 2)
                grids.append(grid)
                expanded_strides.append(np.full((grid.shape[0], 1), stride))
            
            self._grids = np.concatenate(grids, axis=0)
            self._expanded_strides = np.concatenate(expanded_strides, axis=0)
        
        return self._grids, self._expanded_strides
    
    def decode(self, outputs: List[np.ndarray]) -> np.ndarray:
        if self.model_type == 'yolox':
            return self._decode_yolox(outputs[0])
        elif self.model_type == 'yolov5':
            if self.yolov5_format == 'multi':
                return self._decode_yolov5_multi(outputs)
            else:
                return self._decode_yolov5_single(outputs[0])
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _decode_yolox(self, raw_output: np.ndarray) -> np.ndarray:
        predictions = raw_output[0]
        grids, expanded_strides = self._get_yolox_grids()
        
        box_preds = predictions[:, :4]
        obj_preds = predictions[:, 4:5]
        cls_preds = predictions[:, 5:]
        
        x_center = (grids[:, 0:1] + box_preds[:, 0:1]) * expanded_strides
        y_center = (grids[:, 1:2] + box_preds[:, 1:2]) * expanded_strides
        w = np.exp(box_preds[:, 2:3]) * expanded_strides
        h = np.exp(box_preds[:, 3:4]) * expanded_strides
        
        x1 = x_center - w / 2
        y1 = y_center - h / 2
        x2 = x_center + w / 2
        y2 = y_center + h / 2
        
        decoded = np.concatenate([x1, y1, x2, y2, obj_preds, cls_preds], axis=1)
        return decoded
    
    def _decode_yolov5_multi(self, outputs: List[np.ndarray]) -> np.ndarray:
        def sigmoid(x):
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        
        all_predictions = []
        
        for i, (stride, anchor, pred) in enumerate(zip(self.strides, self.anchors, outputs)):
            batch_size, num_anchors, ny, nx, features = pred.shape
            pred = sigmoid(pred)
            
            grid = self._make_grid(nx, ny)
            anchor_grid = np.array(anchor).reshape((1, num_anchors, 1, 1, 2))
            
            pred[..., 0:2] = (pred[..., 0:2] * 2. - 0.5 + grid) * stride
            pred[..., 2:4] = (pred[..., 2:4] * 2) ** 2 * anchor_grid
            
            pred = pred.reshape((batch_size, -1, features))
            all_predictions.append(pred)
        
        predictions = np.concatenate(all_predictions, axis=1)[0]
        
        boxes = predictions[:, :4]
        xyxy = np.copy(boxes)
        xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
        predictions[:, :4] = xyxy
        
        return predictions
    
    def _decode_yolov5_single(self, raw_output: np.ndarray) -> np.ndarray:
        predictions = raw_output[0]
        predictions = predictions.T
        
        boxes = predictions[:, :4]
        class_scores = predictions[:, 4:]
        objness = np.max(class_scores, axis=1, keepdims=True)
        
        if boxes.max() <= 1.0:
            boxes = boxes * self.input_size
        
        xyxy = np.copy(boxes)
        xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
        
        decoded = np.concatenate([xyxy, objness, class_scores], axis=1)
        return decoded
    
    def postprocess(self, predictions: np.ndarray, scale_info: tuple, return_all_classes: bool = False) -> np.ndarray:
        scale, new_h, new_w = scale_info
        
        boxes = predictions[:, :4]
        objness = predictions[:, 4]
        class_scores = predictions[:, 5:]
        
        scores = objness[:, np.newaxis] * class_scores
        max_scores = np.max(scores, axis=1)
        class_ids = np.argmax(scores, axis=1)
        
        valid_mask = max_scores > self.conf_thres
        boxes = boxes[valid_mask]
        scores = max_scores[valid_mask]
        class_ids = class_ids[valid_mask]
        
        if len(boxes) == 0:
            return np.array([])
        
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            scores.tolist(),
            self.conf_thres,
            self.nms_thres
        )
        
        if len(indices) == 0:
            return np.array([])
        
        indices = indices.flatten() if hasattr(indices, 'flatten') else indices
        
        dets = []
        for idx in indices:
            dets.append([*boxes[idx], scores[idx], class_ids[idx]])
        dets = np.array(dets)
        
        dets[:, [0, 2]] = dets[:, [0, 2]] / scale
        dets[:, [1, 3]] = dets[:, [1, 3]] / scale
        
        return dets
    
    def predict(self, image: Union[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        input_tensor, original_img, scale_info = self.preprocess(image)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        predictions = self.decode(outputs)
        detections = self.postprocess(predictions, scale_info)
        return detections, original_img
    
    def benchmark_speed(self, num_runs: int = 100, warmup: int = 10) -> Dict:
        dummy_input = np.random.randn(1, 3, self.input_size, self.input_size).astype(np.float32)
        if self.model_type == 'yolov5':
            dummy_input = dummy_input / 255.0
        
        for _ in range(warmup):
            self.session.run(None, {self.input_name: dummy_input})
        
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            self.session.run(None, {self.input_name: dummy_input})
            end = time.perf_counter()
            times.append((end - start) * 1000)
        
        times = np.array(times)
        
        return {
            'fps': 1000.0 / times.mean(),
            'mean_latency_ms': times.mean(),
            'std_latency_ms': times.std(),
            'min_latency_ms': times.min(),
            'max_latency_ms': times.max()
        }
    
    def calculate_flops(self) -> Dict:
        try:
            import onnx
        except ImportError:
            warnings.warn("Install onnx: pip install onnx")
            return {'flops': 0, 'params': 0}
        
        model = onnx.load(self.model_path)
        graph = model.graph
        
        total_flops = 0
        total_params = 0
        
        value_info = {vi.name: vi for vi in list(graph.value_info) + list(graph.input) + list(graph.output)}
        
        def get_shape(tensor_name):
            if tensor_name in value_info:
                tensor = value_info[tensor_name]
                return [d.dim_value for d in tensor.type.tensor_type.shape.dim]
            return None
        
        for node in graph.node:
            op_type = node.op_type
            inputs = list(node.input)
            outputs = list(node.output)
            
            if op_type == 'Conv':
                weight_name = inputs[1] if len(inputs) > 1 else None
                if not weight_name:
                    continue
                
                attrs = {attr.name: attr for attr in node.attribute}
                kernel_shape = attrs.get('kernel_shape', None)
                if kernel_shape:
                    k_h = kernel_shape.ints[0]
                    k_w = kernel_shape.ints[1]
                else:
                    k_h = k_w = 3
                
                strides = attrs.get('strides', [1, 1])
                if hasattr(strides, 'ints'):
                    strides = strides.ints
                stride_h, stride_w = strides[0], strides[1]
                
                input_shape = get_shape(inputs[0])
                output_shape = get_shape(outputs[0])
                
                if input_shape and output_shape and len(input_shape) >= 4:
                    batch = input_shape[0]
                    in_c = input_shape[1]
                    out_c = output_shape[1]
                    out_h = output_shape[2]
                    out_w = output_shape[3]
                    
                    flops = 2 * in_c * out_c * k_h * k_w * out_h * out_w
                    total_flops += flops
                    
                    params = out_c * in_c * k_h * k_w
                    if len(inputs) > 2:
                        params += out_c
                    total_params += params
            
            elif op_type in ['MatMul', 'Gemm']:
                if len(inputs) >= 2:
                    input_shape = get_shape(inputs[0])
                    weight_name = inputs[1]
                    
                    weight_tensor = None
                    for init in graph.initializer:
                        if init.name == weight_name:
                            weight_tensor = init
                            break
                    
                    if weight_tensor and input_shape:
                        in_features = input_shape[-1]
                        out_features = weight_tensor.dims[0] if len(weight_tensor.dims) > 0 else 0
                        
                        flops = 2 * in_features * out_features
                        total_flops += flops
                        total_params += in_features * out_features
        
        return {
            'flops': total_flops,
            'params': total_params,
            'flops_g': total_flops / 1e9,
            'params_m': total_params / 1e6
        }
