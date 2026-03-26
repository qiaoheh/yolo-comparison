import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List


def plot_ap_comparison(yolox_results: Dict, yolov5_results: Dict, output_path: str):
    """Bar chart comparing AP per class."""
    classes = list(yolox_results['per_class'].keys())
    yolox_aps = [yolox_results['per_class'][c]['ap'] * 100 for c in classes]
    yolov5_aps = [yolov5_results['per_class'][c]['ap'] * 100 for c in classes]
    
    x = np.arange(len(classes))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 6))
    rects1 = ax.bar(x - width/2, yolox_aps, width, label='YOLOX-Tiny', color='#2E86AB')
    rects2 = ax.bar(x + width/2, yolov5_aps, width, label='YOLOv5-nano', color='#A23B72')
    
    ax.set_xlabel('Classes', fontsize=12)
    ax.set_ylabel('AP (%)', fontsize=12)
    ax.set_title('Per-class AP Comparison (VOC 2012)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(rect.get_x() + rect.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved AP comparison to: {output_path}")
    plt.close()


def plot_speed_accuracy_tradeoff(yolox_bench: Dict, yolov5_bench: Dict, 
                                 yolox_eval: Dict, yolov5_eval: Dict, output_path: str):
    """Scatter plot: Speed vs Accuracy."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract metrics
    yolox_map = yolox_eval['mAP@0.5'] * 100
    yolox_fps = yolox_bench['speed']['fps']
    yolov5_map = yolov5_eval['mAP@0.5'] * 100
    yolov5_fps = yolov5_bench['speed']['fps']
    
    # Plot
    ax.scatter(yolox_fps, yolox_map, s=300, c='#2E86AB', marker='o', 
              label=f'YOLOX-Tiny\n({yolox_fps:.1f} FPS, {yolox_map:.1f} mAP)', 
              edgecolors='black', linewidth=2, zorder=5)
    ax.scatter(yolov5_fps, yolov5_map, s=300, c='#A23B72', marker='s', 
              label=f'YOLOv5-nano\n({yolov5_fps:.1f} FPS, {yolov5_map:.1f} mAP)', 
              edgecolors='black', linewidth=2, zorder=5)
    
    # Annotations
    ax.annotate('Better →', xy=(0.85, 0.15), xycoords='axes fraction',
                fontsize=12, ha='center', color='green', fontweight='bold')
    ax.annotate('↑ Accuracy\n↓ Speed', xy=(0.15, 0.85), xycoords='axes fraction',
                fontsize=10, ha='center', color='gray')
    
    ax.set_xlabel('FPS (Frames Per Second)', fontsize=12)
    ax.set_ylabel('mAP@0.5 (%)', fontsize=12)
    ax.set_title('Speed vs Accuracy Trade-off', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add quadrant lines
    ax.axhline(y=55, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=150, color='gray', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved speed-accuracy plot to: {output_path}")
    plt.close()


def plot_flops_comparison(yolox_bench: Dict, yolov5_bench: Dict, output_path: str):
    """Compare FLOPs and Params."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    models = ['YOLOX-Tiny', 'YOLOv5-nano']
    flops = [yolox_bench['flops']['flops_g'], yolov5_bench['flops']['flops_g']]
    params = [yolox_bench['flops']['params_m'], yolov5_bench['flops']['params_m']]
    colors = ['#2E86AB', '#A23B72']
    
    # FLOPs
    bars1 = ax1.bar(models, flops, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('FLOPs (G)', fontsize=12)
    ax1.set_title('Computational Complexity', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}G', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Params
    bars2 = ax2.bar(models, params, color=colors, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Parameters (M)', fontsize=12)
    ax2.set_title('Model Size', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}M', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved FLOPs comparison to: {output_path}")
    plt.close()


def generate_pr_curves(yolox_results: Dict, yolov5_results: Dict, output_dir: str):
    """Generate PR curves for worst performing classes (example visualization)."""
    # This is a simplified version - full PR curves require saving all predictions
    # For now, create a bar showing precision/recall balance
    
    problematic_classes = ['boat', 'pottedplant', 'bottle', 'chair']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, cls in enumerate(problematic_classes):
        ax = axes[idx]
        
        if cls in yolox_results['per_class'] and cls in yolov5_results['per_class']:
            yolox_ap = yolox_results['per_class'][cls]['ap'] * 100
            yolov5_ap = yolov5_results['per_class'][cls]['ap'] * 100
            
            bars = ax.bar(['YOLOX', 'YOLOv5'], [yolox_ap, yolov5_ap], 
                         color=['#2E86AB', '#A23B72'], edgecolor='black')
            ax.set_ylabel('AP (%)', fontsize=11)
            ax.set_title(f'{cls.capitalize()} Detection Performance', fontsize=12, fontweight='bold')
            ax.set_ylim(0, max(yolox_ap, yolov5_ap) * 1.2)
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
    
    plt.suptitle('Hard Classes Comparison (Low AP)', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    output_path = f"{output_dir}/hard_classes.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved hard classes comparison to: {output_path}")
    plt.close()


def create_summary_table(yolox_bench, yolov5_bench, yolox_eval, yolov5_eval, output_path):
    """Create a comparison table image."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')
    
    data = [
        ['Metric', 'YOLOX-Tiny', 'YOLOv5-nano', 'Difference'],
        ['mAP@0.5 (%)', f"{yolox_eval['mAP@0.5']*100:.2f}", 
         f"{yolov5_eval['mAP@0.5']*100:.2f}", 
         f"+{yolox_eval['mAP@0.5']*100 - yolov5_eval['mAP@0.5']*100:.2f}"],
        ['FPS', f"{yolox_bench['speed']['fps']:.2f}", 
         f"{yolov5_bench['speed']['fps']:.2f}",
         f"-{yolov5_bench['speed']['fps'] - yolox_bench['speed']['fps']:.2f}"],
        ['Latency (ms)', f"{yolox_bench['speed']['mean_latency_ms']:.2f}", 
         f"{yolov5_bench['speed']['mean_latency_ms']:.2f}",
         f"+{yolox_bench['speed']['mean_latency_ms'] - yolov5_bench['speed']['mean_latency_ms']:.2f}"],
        ['FLOPs (G)', f"{yolox_bench['flops']['flops_g']:.2f}", 
         f"{yolov5_bench['flops']['flops_g']:.2f}",
         f"+{yolox_bench['flops']['flops_g'] - yolov5_bench['flops']['flops_g']:.2f}"],
        ['Params (M)', f"{yolox_bench['flops']['params_m']:.2f}", 
         f"{yolov5_bench['flops']['params_m']:.2f}",
         f"+{yolox_bench['flops']['params_m'] - yolov5_bench['flops']['params_m']:.2f}"]
    ]
    
    table = ax.table(cellText=data[1:], colLabels=data[0], cellLoc='center', loc='center',
                    colWidths=[0.25, 0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color code differences
    for i in range(1, 6):
        val = float(data[i][3].replace('+', '').replace('-', ''))
        if i == 1:  # mAP - higher is better
            color = '#90EE90' if '+' in data[i][3] else '#FFB6C1'
        else:  # Others - lower is better (except FPS where higher is better but we show negative diff)
            color = '#FFB6C1' if '+' in data[i][3] else '#90EE90'
        table[(i, 3)].set_facecolor(color)
    
    plt.title('Model Comparison Summary (VOC 2012)', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved summary table to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize comparison results')
    parser.add_argument('--yolox_eval', required=True, help='YOLOX eval JSON')
    parser.add_argument('--yolov5_eval', required=True, help='YOLOv5 eval JSON')
    parser.add_argument('--yolox_bench', required=True, help='YOLOX benchmark JSON')
    parser.add_argument('--yolov5_bench', required=True, help='YOLOv5 benchmark JSON')
    parser.add_argument('--output_dir', default='results/figures')
    
    args = parser.parse_args()
    
    # Load results
    with open(args.yolox_eval) as f:
        yolox_eval = json.load(f)
    with open(args.yolov5_eval) as f:
        yolov5_eval = json.load(f)
    with open(args.yolox_bench) as f:
        yolox_bench = json.load(f)
    with open(args.yolov5_bench) as f:
        yolov5_bench = json.load(f)
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    print("Generating visualizations...")
    plot_ap_comparison(yolox_eval, yolov5_eval, f"{args.output_dir}/ap_comparison.png")
    plot_speed_accuracy_tradeoff(yolox_bench, yolov5_bench, yolox_eval, yolov5_eval, 
                                f"{args.output_dir}/speed_vs_accuracy.png")
    plot_flops_comparison(yolox_bench, yolov5_bench, f"{args.output_dir}/complexity.png")
    generate_pr_curves(yolox_eval, yolov5_eval, args.output_dir)
    create_summary_table(yolox_bench, yolov5_bench, yolov5_eval, yolov5_eval, 
                        f"{args.output_dir}/summary_table.png")
    
    print(f"\nAll visualizations saved to: {args.output_dir}")


if __name__ == "__main__":
    main()