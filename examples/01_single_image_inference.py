#!/usr/bin/env python3
"""
Basic Single Image Depth Estimation Example

This script demonstrates the simplest way to use Depth Anything 3 to estimate
depth from a single image. It loads an image, runs inference, and saves both
the raw depth map and a colorized visualization.

Usage:
    python 01_single_image_inference.py --input path/to/image.jpg --output ./output
    python 01_single_image_inference.py --input image.jpg --model depth-anything/DA3-LARGE
"""

import argparse
import os
import sys
from pathlib import Path

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
        description="Depth Anything 3 - Single Image Depth Estimation"
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
        default="./output",
        help="Output directory for results (default: ./output)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="depth-anything/DA3-LARGE",
        help="Model name or path (default: depth-anything/DA3-LARGE)",
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
        "--colormap",
        type=str,
        default="Spectral",
        help="Colormap for visualization (default: Spectral)",
    )
    return parser.parse_args()


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
    print("Depth Anything 3 - Single Image Depth Estimation")
    print("=" * 60)
    print(f"Input image:    {args.input}")
    print(f"Output dir:     {args.output}")
    print(f"Model:          {args.model}")
    print(f"Device:         {args.device}")
    print(f"Process res:    {args.process_res}")
    print("=" * 60)

    # Load model
    print("\n[1/4] Loading model...")
    try:
        model = DepthAnything3.from_pretrained(args.model)
        model = model.to(args.device)
        model.eval()
        print(f"✓ Model loaded successfully on {args.device}")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        sys.exit(1)

    # Load image
    print("\n[2/4] Loading input image...")
    try:
        image = Image.open(args.input).convert("RGB")
        print(f"✓ Image loaded: {image.size[0]}x{image.size[1]} pixels")
    except Exception as e:
        print(f"✗ Error loading image: {e}")
        sys.exit(1)

    # Run inference
    print("\n[3/4] Running depth estimation...")
    try:
        with torch.no_grad():
            prediction = model.inference(
                image=[args.input],
                process_res=args.process_res,
                process_res_method="upper_bound_resize",
            )

        depth = prediction.depth[0]  # Get first (and only) depth map
        conf = prediction.conf[0] if prediction.conf is not None else None

        print(f"✓ Inference complete")
        print(f"  - Depth map shape: {depth.shape}")
        print(f"  - Depth range: {depth.min():.3f} to {depth.max():.3f}")
        if conf is not None:
            print(f"  - Mean confidence: {conf.mean():.3f}")
    except Exception as e:
        print(f"✗ Error during inference: {e}")
        sys.exit(1)

    # Save results
    print("\n[4/4] Saving results...")
    input_filename = Path(args.input).stem

    try:
        # Save raw depth as numpy array
        depth_npy_path = output_dir / f"{input_filename}_depth.npy"
        np.save(depth_npy_path, depth)
        print(f"✓ Raw depth saved: {depth_npy_path}")

        # Save colorized depth visualization
        depth_colored = visualize_depth(depth, cmap=args.colormap)
        depth_vis_path = output_dir / f"{input_filename}_depth_vis.png"
        Image.fromarray(depth_colored).save(depth_vis_path)
        print(f"✓ Depth visualization saved: {depth_vis_path}")

        # Save confidence map if available
        if conf is not None:
            conf_path = output_dir / f"{input_filename}_confidence.npy"
            np.save(conf_path, conf)
            print(f"✓ Confidence map saved: {conf_path}")

            # Visualize confidence
            conf_colored = visualize_depth(conf, cmap="viridis")
            conf_vis_path = output_dir / f"{input_filename}_confidence_vis.png"
            Image.fromarray(conf_colored).save(conf_vis_path)
            print(f"✓ Confidence visualization saved: {conf_vis_path}")

        # Save side-by-side comparison
        img_array = np.array(image)
        if img_array.shape[:2] != depth_colored.shape[:2]:
            from PIL import Image as PILImage
            depth_colored_resized = np.array(
                PILImage.fromarray(depth_colored).resize(
                    (img_array.shape[1], img_array.shape[0])
                )
            )
        else:
            depth_colored_resized = depth_colored

        comparison = np.hstack([img_array, depth_colored_resized])
        comparison_path = output_dir / f"{input_filename}_comparison.png"
        Image.fromarray(comparison).save(comparison_path)
        print(f"✓ Comparison image saved: {comparison_path}")

    except Exception as e:
        print(f"✗ Error saving results: {e}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("✓ Processing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
