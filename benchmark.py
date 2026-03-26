import argparse
import json
import time
import numpy as np
from detector import YOLODetector


def benchmark_model(model_path: str, model_type: str, input_size: int = 416, num_runs: int = 100):
    """Run speed benchmark."""
    print(f"\nBenchmarking {model_type}...")
    print(f"Model: {model_path}")
    print(f"Input size: {input_size}x{input_size}")
    print(f"Runs: {num_runs}")
    
    detector = YOLODetector(
        model_path=model_path,
        model_type=model_type,
        input_size=input_size,
        conf_thres=0.3
    )
    
    # Calculate FLOPs
    print("\nCalculating FLOPs...")
    flops_info = detector.calculate_flops()
    
    # Benchmark speed
    print("Running speed benchmark...")
    speed_info = detector.benchmark_speed(num_runs=num_runs, warmup=10)
    
    results = {
        'model_type': model_type,
        'model_path': model_path,
        'input_size': input_size,
        'flops': flops_info,
        'speed': speed_info
    }
    
    print("\n" + "="*50)
    print("BENCHMARK RESULTS")
    print("="*50)
    print(f"FLOPs:  {flops_info['flops_g']:.2f} G")
    print(f"Params: {flops_info['params_m']:.2f} M")
    print(f"FPS:    {speed_info['fps']:.2f}")
    print(f"Latency: {speed_info['mean_latency_ms']:.2f} ± {speed_info['std_latency_ms']:.2f} ms")
    print(f"Min/Max: {speed_info['min_latency_ms']:.2f} / {speed_info['max_latency_ms']:.2f} ms")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Benchmark YOLO models')
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--model_type', choices=['yolox', 'yolov5'], required=True)
    parser.add_argument('--input_size', type=int, default=416)
    parser.add_argument('--num_runs', type=int, default=100)
    parser.add_argument('--output', default='results/benchmark.json')
    
    args = parser.parse_args()
    
    results = benchmark_model(args.model_path, args.model_type, args.input_size, args.num_runs)
    
    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()