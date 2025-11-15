#!/usr/bin/env python3
"""
Batch Image Processing Example

This script demonstrates how to efficiently process multiple images from a directory
using Depth Anything 3. It supports batch processing to maximize GPU utilization
and includes progress tracking.

Usage:
    python 02_batch_image_processing.py --input ./images --output ./output
    python 02_batch_image_processing.py --input ./images --batch-size 4 --extension jpg,png
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.visualize import visualize_depth


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Depth Anything 3 - Batch Image Processing"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Directory containing input images",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="./output_batch",
        help="Output directory for results (default: ./output_batch)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="depth-anything/DA3-LARGE",
        help="Model name or path (default: depth-anything/DA3-LARGE)",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=1,
        help="Batch size for processing (default: 1)",
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
        "--extension",
        type=str,
        default="jpg,jpeg,png,bmp,tiff",
        help="Comma-separated list of image extensions (default: jpg,jpeg,png,bmp,tiff)",
    )
    parser.add_argument(
        "--save-raw",
        action="store_true",
        help="Save raw depth arrays (.npy files)",
    )
    parser.add_argument(
        "--save-npz",
        action="store_true",
        help="Save batch results as NPZ file",
    )
    return parser.parse_args()


def find_images(directory: str, extensions: List[str]) -> List[Path]:
    """Find all images in directory with given extensions."""
    directory = Path(directory)
    images = []

    for ext in extensions:
        # Case-insensitive pattern matching
        images.extend(directory.glob(f"*.{ext}"))
        images.extend(directory.glob(f"*.{ext.upper()}"))

    return sorted(list(set(images)))  # Remove duplicates and sort


def process_batch(
    model: DepthAnything3,
    image_paths: List[Path],
    process_res: int,
    device: str,
) -> tuple:
    """Process a batch of images."""
    with torch.no_grad():
        prediction = model.inference(
            image=[str(p) for p in image_paths],
            process_res=process_res,
            process_res_method="upper_bound_resize",
        )

    return prediction


def main():
    """Main execution function."""
    args = parse_args()

    # Validate input directory
    if not os.path.isdir(args.input):
        print(f"Error: Input directory not found: {args.input}")
        sys.exit(1)

    # Parse extensions
    extensions = [ext.strip().lower() for ext in args.extension.split(",")]

    # Find images
    print("Scanning for images...")
    image_paths = find_images(args.input, extensions)

    if not image_paths:
        print(f"Error: No images found in {args.input} with extensions {extensions}")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Depth Anything 3 - Batch Image Processing")
    print("=" * 60)
    print(f"Input directory:  {args.input}")
    print(f"Output directory: {args.output}")
    print(f"Found images:     {len(image_paths)}")
    print(f"Model:            {args.model}")
    print(f"Batch size:       {args.batch_size}")
    print(f"Device:           {args.device}")
    print(f"Process res:      {args.process_res}")
    print("=" * 60)

    # Load model
    print("\n[1/3] Loading model...")
    try:
        model = DepthAnything3.from_pretrained(args.model)
        model = model.to(args.device)
        model.eval()
        print(f"✓ Model loaded successfully on {args.device}")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        sys.exit(1)

    # Process images in batches
    print("\n[2/3] Processing images...")

    all_depths = []
    all_confs = []
    processed_names = []

    num_batches = (len(image_paths) + args.batch_size - 1) // args.batch_size

    try:
        with tqdm(total=len(image_paths), desc="Processing", unit="img") as pbar:
            for batch_idx in range(num_batches):
                start_idx = batch_idx * args.batch_size
                end_idx = min(start_idx + args.batch_size, len(image_paths))
                batch_paths = image_paths[start_idx:end_idx]

                # Process batch
                prediction = process_batch(
                    model, batch_paths, args.process_res, args.device
                )

                # Save individual results
                for i, img_path in enumerate(batch_paths):
                    depth = prediction.depth[i]
                    conf = prediction.conf[i] if prediction.conf is not None else None

                    filename = img_path.stem

                    # Save visualization
                    depth_colored = visualize_depth(depth, cmap="Spectral")
                    vis_path = output_dir / f"{filename}_depth.png"
                    Image.fromarray(depth_colored).save(vis_path)

                    # Save raw depth if requested
                    if args.save_raw:
                        depth_npy_path = output_dir / f"{filename}_depth.npy"
                        np.save(depth_npy_path, depth)

                    # Collect for batch NPZ if requested
                    if args.save_npz:
                        all_depths.append(depth)
                        if conf is not None:
                            all_confs.append(conf)
                        processed_names.append(filename)

                    pbar.update(1)

        print(f"\n✓ Processed {len(image_paths)} images successfully")

    except Exception as e:
        print(f"\n✗ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Save batch NPZ if requested
    print("\n[3/3] Saving batch results...")
    if args.save_npz and all_depths:
        try:
            npz_data = {
                "depth": np.array(all_depths),
                "filenames": np.array(processed_names),
            }
            if all_confs:
                npz_data["confidence"] = np.array(all_confs)

            npz_path = output_dir / "batch_results.npz"
            np.savez_compressed(npz_path, **npz_data)
            print(f"✓ Batch NPZ saved: {npz_path}")
        except Exception as e:
            print(f"✗ Error saving NPZ: {e}")

    print("\n" + "=" * 60)
    print("✓ Batch processing complete!")
    print(f"✓ Results saved to: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
