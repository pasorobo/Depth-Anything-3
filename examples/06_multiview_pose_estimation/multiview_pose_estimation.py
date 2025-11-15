#!/usr/bin/env python3
"""
Multi-View Depth and Camera Pose Estimation Example

This script demonstrates how to process multiple images of the same scene and
estimate both depth maps and camera poses. This is useful for 3D reconstruction,
novel view synthesis, and scene understanding.

Usage:
    python 06_multiview_pose_estimation.py --input images/ --output ./output_multiview
    python 06_multiview_pose_estimation.py --input images/ --visualize-cameras
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.visualize import visualize_depth


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Depth Anything 3 - Multi-View Depth and Pose Estimation"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Directory containing input images of the same scene",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="./output_multiview",
        help="Output directory for results (default: ./output_multiview)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="depth-anything/DA3-LARGE",
        help="Model name (default: depth-anything/DA3-LARGE)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum number of images to process",
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
        "--visualize-cameras",
        action="store_true",
        help="Create 3D visualization of estimated camera poses",
    )
    parser.add_argument(
        "--export-glb",
        action="store_true",
        help="Export 3D reconstruction as GLB file",
    )
    return parser.parse_args()


def find_images(directory: str):
    """Find all images in directory."""
    directory = Path(directory)
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]
    images = []

    for ext in extensions:
        images.extend(directory.glob(ext))
        images.extend(directory.glob(ext.upper()))

    return sorted(list(set(images)))


def visualize_camera_poses(
    extrinsics: np.ndarray, output_path: str, image_names: list = None
):
    """
    Visualize camera poses in 3D space.

    Args:
        extrinsics: (N, 4, 4) or (N, 3, 4) array of world-to-camera matrices
        output_path: Path to save visualization
        image_names: Optional list of image names for labels
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    n_cameras = len(extrinsics)

    # Extract camera positions (world coordinates)
    # For world-to-camera matrix [R|t], camera position in world = -R^T @ t
    camera_positions = []

    for i, ext in enumerate(extrinsics):
        if ext.shape == (3, 4):
            R = ext[:3, :3]
            t = ext[:3, 3]
        else:  # (4, 4)
            R = ext[:3, :3]
            t = ext[:3, 3]

        # Camera position in world coordinates
        cam_pos = -R.T @ t
        camera_positions.append(cam_pos)

        # Plot camera position
        ax.scatter(cam_pos[0], cam_pos[1], cam_pos[2], c="red", s=100, marker="o")

        # Plot camera orientation (viewing direction)
        # Camera looks in -Z direction in camera space
        view_dir = R.T @ np.array([0, 0, -1])
        ax.quiver(
            cam_pos[0],
            cam_pos[1],
            cam_pos[2],
            view_dir[0],
            view_dir[1],
            view_dir[2],
            length=0.5,
            color="blue",
            arrow_length_ratio=0.3,
        )

        # Add label
        if image_names and i < len(image_names):
            label = image_names[i]
        else:
            label = f"Cam {i}"

        ax.text(cam_pos[0], cam_pos[1], cam_pos[2], f"  {label}", fontsize=8)

    camera_positions = np.array(camera_positions)

    # Set equal aspect ratio
    max_range = np.array(
        [
            camera_positions[:, 0].max() - camera_positions[:, 0].min(),
            camera_positions[:, 1].max() - camera_positions[:, 1].min(),
            camera_positions[:, 2].max() - camera_positions[:, 2].min(),
        ]
    ).max() / 2.0

    mid_x = (camera_positions[:, 0].max() + camera_positions[:, 0].min()) * 0.5
    mid_y = (camera_positions[:, 1].max() + camera_positions[:, 1].min()) * 0.5
    mid_z = (camera_positions[:, 2].max() + camera_positions[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Estimated Camera Poses ({n_cameras} cameras)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def print_camera_info(extrinsics: np.ndarray, intrinsics: np.ndarray):
    """Print camera pose and intrinsic information."""
    print("\nCamera Extrinsics (World-to-Camera):")
    print("-" * 40)

    for i, ext in enumerate(extrinsics):
        print(f"\nCamera {i}:")
        if ext.shape == (3, 4):
            R = ext[:3, :3]
            t = ext[:3, 3]
        else:
            R = ext[:3, :3]
            t = ext[:3, 3]

        # Camera position in world
        cam_pos = -R.T @ t
        print(f"  Position (world): [{cam_pos[0]:.3f}, {cam_pos[1]:.3f}, {cam_pos[2]:.3f}]")

        # Rotation angles (approximate)
        # Extract Euler angles (this is a simplified approach)
        import math

        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6

        if not singular:
            roll = math.atan2(R[2, 1], R[2, 2])
            pitch = math.atan2(-R[2, 0], sy)
            yaw = math.atan2(R[1, 0], R[0, 0])
        else:
            roll = math.atan2(-R[1, 2], R[1, 1])
            pitch = math.atan2(-R[2, 0], sy)
            yaw = 0

        print(
            f"  Rotation (deg):   Roll={math.degrees(roll):.1f}, "
            f"Pitch={math.degrees(pitch):.1f}, Yaw={math.degrees(yaw):.1f}"
        )

    if intrinsics is not None:
        print("\n\nCamera Intrinsics:")
        print("-" * 40)
        for i, intr in enumerate(intrinsics):
            fx, fy = intr[0, 0], intr[1, 1]
            cx, cy = intr[0, 2], intr[1, 2]
            print(f"\nCamera {i}:")
            print(f"  Focal length: fx={fx:.2f}, fy={fy:.2f}")
            print(f"  Principal pt: cx={cx:.2f}, cy={cy:.2f}")


def main():
    """Main execution function."""
    args = parse_args()

    # Validate input directory
    if not os.path.isdir(args.input):
        print(f"Error: Input directory not found: {args.input}")
        sys.exit(1)

    # Find images
    print("Scanning for images...")
    image_paths = find_images(args.input)

    if not image_paths:
        print(f"Error: No images found in {args.input}")
        sys.exit(1)

    # Limit number of images if specified
    if args.max_images and len(image_paths) > args.max_images:
        image_paths = image_paths[: args.max_images]
        print(f"Limited to {args.max_images} images")

    if len(image_paths) < 2:
        print("Warning: Multi-view estimation works best with 2+ images")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Depth Anything 3 - Multi-View Depth and Pose Estimation")
    print("=" * 60)
    print(f"Input directory:  {args.input}")
    print(f"Output directory: {args.output}")
    print(f"Number of images: {len(image_paths)}")
    print(f"Model:            {args.model}")
    print(f"Device:           {args.device}")
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

    # Process images
    print("\n[2/4] Processing images and estimating poses...")
    try:
        with torch.no_grad():
            prediction = model.inference(
                image=[str(p) for p in image_paths],
                process_res=args.process_res,
                process_res_method="upper_bound_resize",
            )

        print(f"✓ Processed {len(image_paths)} images")
        print(f"  - Depth maps: {prediction.depth.shape}")

        if prediction.extrinsics is not None:
            print(f"  - Extrinsics: {prediction.extrinsics.shape}")
        else:
            print("  - Warning: No camera extrinsics estimated")

        if prediction.intrinsics is not None:
            print(f"  - Intrinsics: {prediction.intrinsics.shape}")
        else:
            print("  - Warning: No camera intrinsics estimated")

    except Exception as e:
        print(f"✗ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Analyze results
    print("\n[3/4] Analyzing results...")

    # Print camera information
    if prediction.extrinsics is not None:
        image_names = [p.stem for p in image_paths]
        print_camera_info(prediction.extrinsics, prediction.intrinsics)
    else:
        print("No camera pose information available")

    # Save results
    print("\n[4/4] Saving results...")

    try:
        # Save depth maps
        for i, img_path in enumerate(image_paths):
            depth = prediction.depth[i]
            filename = img_path.stem

            # Colorized depth
            depth_colored = visualize_depth(depth, cmap="Spectral")
            vis_path = output_dir / f"{filename}_depth.png"
            Image.fromarray(depth_colored).save(vis_path)

        print(f"✓ Saved {len(image_paths)} depth visualizations")

        # Save camera pose visualization
        if args.visualize_cameras and prediction.extrinsics is not None:
            pose_vis_path = output_dir / "camera_poses.png"
            visualize_camera_poses(
                prediction.extrinsics, str(pose_vis_path), image_names
            )
            print(f"✓ Camera pose visualization saved: {pose_vis_path}")

        # Save data as NPZ
        npz_data = {
            "depth": prediction.depth,
            "processed_images": prediction.processed_images,
            "filenames": np.array([p.name for p in image_paths]),
        }

        if prediction.extrinsics is not None:
            npz_data["extrinsics"] = prediction.extrinsics
        if prediction.intrinsics is not None:
            npz_data["intrinsics"] = prediction.intrinsics
        if prediction.conf is not None:
            npz_data["confidence"] = prediction.conf

        npz_path = output_dir / "multiview_data.npz"
        np.savez_compressed(npz_path, **npz_data)
        print(f"✓ Multi-view data saved: {npz_path}")

        # Export GLB if requested
        if args.export_glb:
            from depth_anything_3.utils.export.glb import export_glb

            glb_path = output_dir / "scene.glb"
            export_glb(prediction=prediction, output_path=str(glb_path))
            print(f"✓ 3D reconstruction saved: {glb_path}")

    except Exception as e:
        print(f"✗ Error saving results: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("✓ Multi-view processing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
