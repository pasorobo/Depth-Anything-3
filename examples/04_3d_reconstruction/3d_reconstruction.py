#!/usr/bin/env python3
"""
3D Reconstruction and Export Example

This script demonstrates how to generate 3D point clouds and meshes from images
using Depth Anything 3. It supports various export formats including GLB, PLY,
and NPZ for use in 3D visualization tools.

Usage:
    python 04_3d_reconstruction.py --input image.jpg --output ./output_3d
    python 04_3d_reconstruction.py --input images/ --format glb,ply --export-all
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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Depth Anything 3 - 3D Reconstruction and Export"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to input image or directory of images",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="./output_3d",
        help="Output directory for results (default: ./output_3d)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="depth-anything/DA3-LARGE",
        help="Model name or path (default: depth-anything/DA3-LARGE)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="glb",
        help="Export format(s): glb, npz, or combined with '-' (default: glb)",
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
        "--point-size",
        type=float,
        default=0.01,
        help="Point size for 3D visualization (default: 0.01)",
    )
    parser.add_argument(
        "--export-all",
        action="store_true",
        help="Export all available formats (glb, npz, mini_npz)",
    )
    return parser.parse_args()


def find_images(path: str):
    """Find images from path (file or directory)."""
    path = Path(path)

    if path.is_file():
        return [path]
    elif path.is_dir():
        extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]
        images = []
        for ext in extensions:
            images.extend(path.glob(ext))
            images.extend(path.glob(ext.upper()))
        return sorted(list(set(images)))
    else:
        return []


def main():
    """Main execution function."""
    args = parse_args()

    # Find input images
    image_paths = find_images(args.input)

    if not image_paths:
        print(f"Error: No images found at: {args.input}")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine export formats
    if args.export_all:
        export_formats = "glb-npz-mini_npz"
    else:
        export_formats = args.format

    print("=" * 60)
    print("Depth Anything 3 - 3D Reconstruction")
    print("=" * 60)
    print(f"Input images:     {len(image_paths)}")
    print(f"Output dir:       {args.output}")
    print(f"Model:            {args.model}")
    print(f"Export format:    {export_formats}")
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

    # Process images and generate 3D data
    print("\n[2/3] Processing images and estimating depth...")
    try:
        with torch.no_grad():
            prediction = model.inference(
                image=[str(p) for p in image_paths],
                process_res=args.process_res,
                process_res_method="upper_bound_resize",
            )

        print(f"✓ Processed {len(image_paths)} images")
        print(f"  - Depth shape: {prediction.depth.shape}")

        if prediction.extrinsics is not None:
            print(f"  - Estimated camera poses: {prediction.extrinsics.shape}")
        if prediction.intrinsics is not None:
            print(f"  - Estimated camera intrinsics: {prediction.intrinsics.shape}")

    except Exception as e:
        print(f"✗ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Export 3D data
    print("\n[3/3] Exporting 3D data...")

    # Get base name for output files
    if len(image_paths) == 1:
        base_name = image_paths[0].stem
    else:
        base_name = "scene"

    try:
        # Export using built-in export functionality
        from depth_anything_3.utils.export import export_prediction

        export_prediction(
            prediction=prediction,
            export_dir=str(output_dir),
            export_format=export_formats,
            filename_prefix=base_name,
        )

        print(f"✓ Export complete")

        # List exported files
        exported_files = []

        if "glb" in export_formats:
            glb_file = output_dir / f"{base_name}.glb"
            if glb_file.exists():
                exported_files.append(("GLB (3D mesh)", glb_file))

        if "npz" in export_formats:
            npz_file = output_dir / f"{base_name}.npz"
            if npz_file.exists():
                exported_files.append(("NPZ (full data)", npz_file))

        if "mini_npz" in export_formats:
            mini_npz_file = output_dir / f"{base_name}_mini.npz"
            if mini_npz_file.exists():
                exported_files.append(("Mini NPZ (compact)", mini_npz_file))

        if exported_files:
            print("\nExported files:")
            for desc, filepath in exported_files:
                file_size = filepath.stat().st_size / 1024 / 1024  # MB
                print(f"  - {desc}: {filepath.name} ({file_size:.2f} MB)")

    except Exception as e:
        print(f"✗ Error during export: {e}")
        import traceback
        traceback.print_exc()

        # Fallback: Manual export
        print("\nAttempting manual export...")
        try:
            # Save as NPZ
            npz_path = output_dir / f"{base_name}_manual.npz"
            npz_data = {
                "depth": prediction.depth,
                "processed_images": prediction.processed_images,
            }

            if prediction.conf is not None:
                npz_data["confidence"] = prediction.conf
            if prediction.extrinsics is not None:
                npz_data["extrinsics"] = prediction.extrinsics
            if prediction.intrinsics is not None:
                npz_data["intrinsics"] = prediction.intrinsics

            np.savez_compressed(npz_path, **npz_data)
            print(f"✓ Manual NPZ export saved: {npz_path}")

            # Create simple point cloud
            print("\nGenerating point cloud...")
            from depth_anything_3.utils.export.glb import export_glb

            glb_path = output_dir / f"{base_name}.glb"
            export_glb(
                prediction=prediction,
                output_path=str(glb_path),
                point_size=args.point_size,
            )
            print(f"✓ GLB file saved: {glb_path}")

        except Exception as e2:
            print(f"✗ Manual export also failed: {e2}")

    print("\n" + "=" * 60)
    print("✓ 3D reconstruction complete!")
    print("\nYou can view the GLB files using:")
    print("  - Online: https://3dviewer.net/")
    print("  - Blender: File > Import > glTF 2.0")
    print("  - Windows 3D Viewer")
    print("=" * 60)


if __name__ == "__main__":
    main()
