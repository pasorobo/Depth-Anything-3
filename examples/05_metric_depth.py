#!/usr/bin/env python3
"""
Metric Depth Estimation Example

This script demonstrates how to estimate metric (real-world scale) depth using
the DA3METRIC model variant. Unlike relative depth, metric depth provides
actual distance measurements in meters.

Usage:
    python 05_metric_depth.py --input image.jpg --output ./output_metric
    python 05_metric_depth.py --input image.jpg --visualize-ranges
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.visualize import visualize_depth


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Depth Anything 3 - Metric Depth Estimation"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to input image",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="./output_metric",
        help="Output directory for results (default: ./output_metric)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="depth-anything/DA3METRIC-LARGE",
        help="Model name (default: depth-anything/DA3METRIC-LARGE)",
    )
    parser.add_argument(
        "--process-res",
        type=int,
        default=504,
        help="Processing resolution (default: 504)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu, default: auto-detect)",
    )
    parser.add_argument(
        "--visualize-ranges",
        action="store_true",
        help="Create visualizations for different depth ranges",
    )
    parser.add_argument(
        "--depth-unit",
        type=str,
        default="meters",
        choices=["meters", "centimeters", "feet"],
        help="Unit for depth display (default: meters)",
    )
    return parser.parse_args()


def convert_depth_units(depth: np.ndarray, unit: str) -> tuple:
    """Convert depth from meters to specified unit."""
    if unit == "centimeters":
        return depth * 100, "cm"
    elif unit == "feet":
        return depth * 3.28084, "ft"
    else:
        return depth, "m"


def create_range_visualizations(depth: np.ndarray, output_dir: Path, base_name: str):
    """Create visualizations highlighting different depth ranges."""
    ranges = [
        (0, 2, "Close (0-2m)"),
        (2, 5, "Medium (2-5m)"),
        (5, 10, "Far (5-10m)"),
        (10, float("inf"), "Very Far (>10m)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, (min_depth, max_depth, title) in enumerate(ranges):
        # Create mask for this range
        mask = (depth >= min_depth) & (depth < max_depth)

        # Create visualization
        depth_masked = np.where(mask, depth, np.nan)
        im = axes[idx].imshow(depth_masked, cmap="Spectral")
        axes[idx].set_title(f"{title}\n{mask.sum()} pixels")
        axes[idx].axis("off")
        plt.colorbar(im, ax=axes[idx], label="Depth (m)")

    plt.tight_layout()
    range_vis_path = output_dir / f"{base_name}_depth_ranges.png"
    plt.savefig(range_vis_path, dpi=150, bbox_inches="tight")
    plt.close()

    return range_vis_path


def create_histogram(depth: np.ndarray, output_dir: Path, base_name: str, unit_str: str):
    """Create depth histogram."""
    plt.figure(figsize=(10, 6))

    # Filter out invalid values
    valid_depth = depth[np.isfinite(depth)]

    plt.hist(valid_depth.flatten(), bins=100, edgecolor="black", alpha=0.7)
    plt.xlabel(f"Depth ({unit_str})")
    plt.ylabel("Number of Pixels")
    plt.title("Depth Distribution")
    plt.grid(True, alpha=0.3)

    # Add statistics
    stats_text = (
        f"Mean: {valid_depth.mean():.2f} {unit_str}\n"
        f"Median: {np.median(valid_depth):.2f} {unit_str}\n"
        f"Min: {valid_depth.min():.2f} {unit_str}\n"
        f"Max: {valid_depth.max():.2f} {unit_str}"
    )
    plt.text(
        0.98,
        0.98,
        stats_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        fontsize=10,
    )

    hist_path = output_dir / f"{base_name}_histogram.png"
    plt.savefig(hist_path, dpi=150, bbox_inches="tight")
    plt.close()

    return hist_path


def main():
    """Main execution function."""
    args = parse_args()

    # Validate input
    if not os.path.exists(args.input):
        print(f"Error: Input image not found: {args.input}")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Depth Anything 3 - Metric Depth Estimation")
    print("=" * 60)
    print(f"Input image:    {args.input}")
    print(f"Output dir:     {args.output}")
    print(f"Model:          {args.model}")
    print(f"Device:         {args.device}")
    print(f"Depth unit:     {args.depth_unit}")
    print("=" * 60)

    # Load model
    print("\n[1/4] Loading metric depth model...")
    try:
        model = DepthAnything3.from_pretrained(args.model)
        model = model.to(args.device)
        model.eval()
        print(f"✓ Model loaded successfully on {args.device}")

        # Check if model is metric
        if not hasattr(model, "is_metric") or not model.is_metric:
            print("⚠ Warning: This model may not be a metric depth model.")
            print("  For best results, use: depth-anything/DA3METRIC-LARGE")

    except Exception as e:
        print(f"✗ Error loading model: {e}")
        sys.exit(1)

    # Load and process image
    print("\n[2/4] Processing image...")
    try:
        image = Image.open(args.input).convert("RGB")
        print(f"✓ Image loaded: {image.size[0]}x{image.size[1]} pixels")

        with torch.no_grad():
            prediction = model.inference(
                image=[args.input],
                process_res=args.process_res,
                process_res_method="upper_bound_resize",
            )

        depth = prediction.depth[0]  # Depth in meters

        print(f"✓ Inference complete")
        print(f"  - Is metric: {bool(prediction.is_metric)}")

    except Exception as e:
        print(f"✗ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Analyze depth
    print("\n[3/4] Analyzing depth map...")
    input_filename = Path(args.input).stem

    # Convert to desired units
    depth_converted, unit_str = convert_depth_units(depth, args.depth_unit)

    # Calculate statistics
    valid_depth = depth_converted[np.isfinite(depth_converted)]

    print(f"\nDepth Statistics ({unit_str}):")
    print(f"  - Mean:   {valid_depth.mean():.2f} {unit_str}")
    print(f"  - Median: {np.median(valid_depth):.2f} {unit_str}")
    print(f"  - Min:    {valid_depth.min():.2f} {unit_str}")
    print(f"  - Max:    {valid_depth.max():.2f} {unit_str}")
    print(f"  - Std:    {valid_depth.std():.2f} {unit_str}")

    # Find closest and farthest points
    min_idx = np.unravel_index(np.argmin(depth_converted), depth_converted.shape)
    max_idx = np.unravel_index(np.argmax(depth_converted), depth_converted.shape)

    print(f"\nClosest point:  {depth_converted[min_idx]:.2f} {unit_str} at pixel {min_idx}")
    print(f"Farthest point: {depth_converted[max_idx]:.2f} {unit_str} at pixel {max_idx}")

    # Save results
    print("\n[4/4] Saving results...")

    try:
        # Save raw depth (in meters)
        depth_npy_path = output_dir / f"{input_filename}_depth_metric.npy"
        np.save(depth_npy_path, depth)
        print(f"✓ Raw depth saved: {depth_npy_path}")

        # Save colorized depth
        depth_colored = visualize_depth(depth_converted, cmap="Spectral")
        depth_vis_path = output_dir / f"{input_filename}_depth_vis.png"
        Image.fromarray(depth_colored).save(depth_vis_path)
        print(f"✓ Depth visualization saved: {depth_vis_path}")

        # Create histogram
        hist_path = create_histogram(
            depth_converted, output_dir, input_filename, unit_str
        )
        print(f"✓ Histogram saved: {hist_path}")

        # Create range visualizations if requested
        if args.visualize_ranges:
            range_vis_path = create_range_visualizations(
                depth, output_dir, input_filename
            )
            print(f"✓ Range visualization saved: {range_vis_path}")

        # Save statistics to text file
        stats_path = output_dir / f"{input_filename}_stats.txt"
        with open(stats_path, "w") as f:
            f.write("Metric Depth Statistics\n")
            f.write("=" * 40 + "\n")
            f.write(f"Image: {args.input}\n")
            f.write(f"Model: {args.model}\n")
            f.write(f"Units: {unit_str}\n\n")
            f.write(f"Mean depth:   {valid_depth.mean():.2f} {unit_str}\n")
            f.write(f"Median depth: {np.median(valid_depth):.2f} {unit_str}\n")
            f.write(f"Min depth:    {valid_depth.min():.2f} {unit_str}\n")
            f.write(f"Max depth:    {valid_depth.max():.2f} {unit_str}\n")
            f.write(f"Std dev:      {valid_depth.std():.2f} {unit_str}\n")
        print(f"✓ Statistics saved: {stats_path}")

    except Exception as e:
        print(f"✗ Error saving results: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("✓ Metric depth estimation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
