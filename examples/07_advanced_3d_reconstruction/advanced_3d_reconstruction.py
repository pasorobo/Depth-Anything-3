#!/usr/bin/env python3
"""
Advanced Large-Scale 3D Reconstruction

This script performs large-scale 3D reconstruction from videos or image sequences.
It processes frames, estimates depth and camera poses, and merges all point clouds
into a unified 3D reconstruction.

Usage:
    # From video
    python 07_advanced_3d_reconstruction.py --input video.mp4 --output ./reconstruction

    # From image directory
    python 07_advanced_3d_reconstruction.py --input images/ --output ./reconstruction --fps 5

    # High quality reconstruction
    python 07_advanced_3d_reconstruction.py --input video.mp4 --model depth-anything/DA3-GIANT --process-res 672
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from depth_anything_3.api import DepthAnything3


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Depth Anything 3 - Advanced Large-Scale 3D Reconstruction"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to video file or directory of images",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="./reconstruction",
        help="Output directory for reconstruction results",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="depth-anything/DA3-LARGE",
        help="Model name or path (default: depth-anything/DA3-LARGE)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Target FPS for video processing (default: process all frames)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to process",
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
        help="Device to use (cuda/cpu)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for processing (default: 8)",
    )
    parser.add_argument(
        "--downsample-points",
        type=float,
        default=1.0,
        help="Downsample point cloud ratio (0-1, default: 1.0 = no downsampling)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Minimum confidence threshold for points (default: 0.5)",
    )
    parser.add_argument(
        "--export-format",
        type=str,
        default="glb,ply,npz",
        help="Export formats separated by comma (default: glb,ply,npz)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Create visualization video of the reconstruction process",
    )
    return parser.parse_args()


def extract_frames_from_video(
    video_path: str, target_fps: float = None, max_frames: int = None
) -> Tuple[List[np.ndarray], float]:
    """Extract frames from video file."""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if target_fps is None:
        frame_skip = 1
        effective_fps = original_fps
    else:
        frame_skip = max(1, int(original_fps / target_fps))
        effective_fps = original_fps / frame_skip

    print(f"Video info:")
    print(f"  - Original FPS: {original_fps:.2f}")
    print(f"  - Total frames: {total_frames}")
    print(f"  - Processing FPS: {effective_fps:.2f}")
    print(f"  - Frame skip: {frame_skip}")

    frames = []
    frame_idx = 0

    with tqdm(desc="Extracting frames", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_skip == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)

                if max_frames and len(frames) >= max_frames:
                    break

            frame_idx += 1
            pbar.update(1)

    cap.release()
    return frames, effective_fps


def find_images(directory: str) -> List[Path]:
    """Find all images in directory."""
    directory = Path(directory)
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]
    images = []

    for ext in extensions:
        images.extend(directory.glob(ext))
        images.extend(directory.glob(ext.upper()))

    return sorted(list(set(images)))


def depth_to_point_cloud(
    depth: np.ndarray,
    image: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray = None,
    confidence: np.ndarray = None,
    min_confidence: float = 0.5,
    downsample: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert depth map to 3D point cloud.

    Returns:
        points: (N, 3) array of 3D points
        colors: (N, 3) array of RGB colors
    """
    h, w = depth.shape

    # Create pixel coordinates
    u, v = np.meshgrid(np.arange(w), np.arange(h))

    # Filter by confidence if available
    if confidence is not None:
        mask = confidence >= min_confidence
    else:
        mask = np.ones_like(depth, dtype=bool)

    # Downsample if requested
    if downsample < 1.0:
        step = int(1.0 / downsample)
        mask[::step, ::step] = False

    # Get valid points
    valid_u = u[mask]
    valid_v = v[mask]
    valid_depth = depth[mask]

    # Unproject to 3D (camera coordinates)
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    x = (valid_u - cx) * valid_depth / fx
    y = (valid_v - cy) * valid_depth / fy
    z = valid_depth

    points_cam = np.stack([x, y, z], axis=-1)

    # Transform to world coordinates if extrinsics provided
    if extrinsics is not None:
        # extrinsics is world-to-camera, so we need camera-to-world
        if extrinsics.shape == (3, 4):
            R = extrinsics[:3, :3]
            t = extrinsics[:3, 3]
        else:
            R = extrinsics[:3, :3]
            t = extrinsics[:3, 3]

        # Camera-to-world transformation
        R_inv = R.T
        t_world = -R_inv @ t

        points_world = (R_inv @ points_cam.T).T + t_world
        points = points_world
    else:
        points = points_cam

    # Get colors
    colors = image[mask]

    return points, colors


def merge_point_clouds(
    all_points: List[np.ndarray], all_colors: List[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """Merge multiple point clouds into one."""
    if not all_points:
        return np.array([]), np.array([])

    merged_points = np.vstack(all_points)
    merged_colors = np.vstack(all_colors)

    return merged_points, merged_colors


def save_point_cloud_ply(points: np.ndarray, colors: np.ndarray, output_path: str):
    """Save point cloud as PLY file."""
    try:
        from plyfile import PlyData, PlyElement
    except ImportError:
        print("Warning: plyfile not installed. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "plyfile"])
        from plyfile import PlyData, PlyElement

    # Ensure colors are in 0-255 range
    if colors.max() <= 1.0:
        colors = (colors * 255).astype(np.uint8)
    else:
        colors = colors.astype(np.uint8)

    # Create structured array
    vertex = np.empty(
        len(points),
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )

    vertex["x"] = points[:, 0]
    vertex["y"] = points[:, 1]
    vertex["z"] = points[:, 2]
    vertex["red"] = colors[:, 0]
    vertex["green"] = colors[:, 1]
    vertex["blue"] = colors[:, 2]

    el = PlyElement.describe(vertex, "vertex")
    PlyData([el]).write(output_path)


def save_point_cloud_npz(
    points: np.ndarray, colors: np.ndarray, output_path: str, metadata: dict = None
):
    """Save point cloud as NPZ file."""
    data = {"points": points, "colors": colors}
    if metadata:
        data.update(metadata)
    np.savez_compressed(output_path, **data)


def main():
    """Main execution function."""
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Depth Anything 3 - Advanced Large-Scale 3D Reconstruction")
    print("=" * 70)

    # Determine input type and load frames
    input_path = Path(args.input)

    if input_path.is_file():
        print(f"\n[1/5] Loading video: {args.input}")
        frames, fps = extract_frames_from_video(
            args.input, args.fps, args.max_frames
        )
        # Save frames temporarily
        temp_dir = tempfile.mkdtemp()
        frame_paths = []
        for i, frame in enumerate(frames):
            frame_path = Path(temp_dir) / f"frame_{i:06d}.png"
            Image.fromarray(frame).save(frame_path)
            frame_paths.append(frame_path)
    elif input_path.is_dir():
        print(f"\n[1/5] Loading images from: {args.input}")
        frame_paths = find_images(args.input)
        if args.max_frames:
            frame_paths = frame_paths[: args.max_frames]
        frames = [np.array(Image.open(p).convert("RGB")) for p in frame_paths]
        fps = args.fps if args.fps else 30.0
    else:
        print(f"Error: Input not found: {args.input}")
        sys.exit(1)

    print(f"✓ Loaded {len(frames)} frames")

    # Load model
    print(f"\n[2/5] Loading model: {args.model}")
    try:
        model = DepthAnything3.from_pretrained(args.model)
        model = model.to(args.device)
        model.eval()
        print(f"✓ Model loaded on {args.device}")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        sys.exit(1)

    # Process frames in batches
    print(f"\n[3/5] Processing frames (batch size: {args.batch_size})...")

    all_depths = []
    all_extrinsics = []
    all_intrinsics = []
    all_confidences = []

    num_batches = (len(frame_paths) + args.batch_size - 1) // args.batch_size

    try:
        with tqdm(total=len(frame_paths), desc="Processing", unit="frame") as pbar:
            for batch_idx in range(num_batches):
                start_idx = batch_idx * args.batch_size
                end_idx = min(start_idx + args.batch_size, len(frame_paths))
                batch_paths = frame_paths[start_idx:end_idx]

                with torch.no_grad():
                    prediction = model.inference(
                        image=[str(p) for p in batch_paths],
                        process_res=args.process_res,
                        process_res_method="upper_bound_resize",
                    )

                all_depths.extend([d for d in prediction.depth])

                if prediction.extrinsics is not None:
                    all_extrinsics.extend([e for e in prediction.extrinsics])

                if prediction.intrinsics is not None:
                    all_intrinsics.extend([i for i in prediction.intrinsics])

                if prediction.conf is not None:
                    all_confidences.extend([c for c in prediction.conf])

                pbar.update(len(batch_paths))

        print(f"✓ Processed {len(all_depths)} frames")

        if all_extrinsics:
            print(f"✓ Estimated {len(all_extrinsics)} camera poses")
        else:
            print("⚠ No camera poses estimated (monocular mode)")

    except Exception as e:
        print(f"✗ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Generate point clouds
    print(f"\n[4/5] Generating point clouds...")

    all_points = []
    all_colors = []

    with tqdm(total=len(frames), desc="Generating point clouds", unit="frame") as pbar:
        for i, (frame, depth) in enumerate(zip(frames, all_depths)):
            extrinsics = all_extrinsics[i] if all_extrinsics else None
            intrinsics = all_intrinsics[i] if all_intrinsics else None
            confidence = all_confidences[i] if all_confidences else None

            # Use default intrinsics if not available
            if intrinsics is None:
                h, w = depth.shape
                fx = fy = max(h, w)
                cx, cy = w / 2, h / 2
                intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

            points, colors = depth_to_point_cloud(
                depth,
                frame,
                intrinsics,
                extrinsics,
                confidence,
                args.min_confidence,
                args.downsample_points,
            )

            all_points.append(points)
            all_colors.append(colors)
            pbar.update(1)

    # Merge all point clouds
    print("\nMerging point clouds...")
    merged_points, merged_colors = merge_point_clouds(all_points, all_colors)

    print(f"✓ Total points: {len(merged_points):,}")

    # Save reconstruction
    print(f"\n[5/5] Saving reconstruction...")

    export_formats = [fmt.strip() for fmt in args.export_format.split(",")]

    if "ply" in export_formats:
        ply_path = output_dir / "reconstruction.ply"
        save_point_cloud_ply(merged_points, merged_colors, str(ply_path))
        file_size = ply_path.stat().st_size / 1024 / 1024
        print(f"✓ PLY saved: {ply_path} ({file_size:.1f} MB)")

    if "npz" in export_formats:
        npz_path = output_dir / "reconstruction.npz"
        metadata = {
            "num_frames": len(frames),
            "model": args.model,
            "process_res": args.process_res,
        }
        save_point_cloud_npz(merged_points, merged_colors, str(npz_path), metadata)
        file_size = npz_path.stat().st_size / 1024 / 1024
        print(f"✓ NPZ saved: {npz_path} ({file_size:.1f} MB)")

    if "glb" in export_formats:
        try:
            # Try to use the built-in GLB export
            print("Generating GLB (this may take a while for large point clouds)...")

            # Create a minimal prediction object for export
            from depth_anything_3.specs import Prediction

            pred_for_export = Prediction(
                depth=np.array(all_depths),
                is_metric=0,
                conf=np.array(all_confidences) if all_confidences else None,
                extrinsics=np.array(all_extrinsics) if all_extrinsics else None,
                intrinsics=np.array(all_intrinsics) if all_intrinsics else None,
                processed_images=np.array(frames),
                gaussians=None,
                aux={},
                scale_factor=None,
            )

            from depth_anything_3.utils.export.glb import export_glb

            glb_path = output_dir / "reconstruction.glb"
            export_glb(pred_for_export, str(glb_path))
            file_size = glb_path.stat().st_size / 1024 / 1024
            print(f"✓ GLB saved: {glb_path} ({file_size:.1f} MB)")

        except Exception as e:
            print(f"⚠ GLB export failed: {e}")

    # Save metadata
    metadata_path = output_dir / "reconstruction_info.txt"
    with open(metadata_path, "w") as f:
        f.write("3D Reconstruction Information\n")
        f.write("=" * 50 + "\n")
        f.write(f"Input: {args.input}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Frames processed: {len(frames)}\n")
        f.write(f"Total points: {len(merged_points):,}\n")
        f.write(f"Processing resolution: {args.process_res}\n")
        f.write(f"Downsampling: {args.downsample_points}\n")
        f.write(f"Min confidence: {args.min_confidence}\n")

    print(f"✓ Metadata saved: {metadata_path}")

    print("\n" + "=" * 70)
    print("✓ 3D Reconstruction Complete!")
    print("=" * 70)
    print(f"\nResults saved to: {output_dir}")
    print(f"Total points: {len(merged_points):,}")
    print("\nView the reconstruction:")
    print("  - PLY/GLB: Use MeshLab, CloudCompare, or https://3dviewer.net/")
    print("  - NPZ: Load with numpy for custom processing")


if __name__ == "__main__":
    main()
