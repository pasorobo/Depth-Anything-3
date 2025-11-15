#!/usr/bin/env python3
"""
Video Depth Estimation Example

This script processes a video file, extracting frames and estimating depth for each.
The results can be saved as individual frames or combined into an output video.

Usage:
    python 03_video_processing.py --input video.mp4 --output ./output_video
    python 03_video_processing.py --input video.mp4 --fps 10 --create-video
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

import cv2
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
        description="Depth Anything 3 - Video Depth Estimation"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to input video file",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="./output_video",
        help="Output directory for results (default: ./output_video)",
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
        help="Target FPS for processing (default: process all frames)",
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
        help="Device to use (cuda/cpu, default: auto-detect)",
    )
    parser.add_argument(
        "--create-video",
        action="store_true",
        help="Create output video from depth maps",
    )
    parser.add_argument(
        "--side-by-side",
        action="store_true",
        help="Create side-by-side comparison video",
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="Save individual depth frames as images",
    )
    return parser.parse_args()


def extract_frames(video_path: str, target_fps: float = None, max_frames: int = None):
    """Extract frames from video."""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    # Get video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calculate frame skip
    if target_fps is None:
        frame_skip = 1
        effective_fps = original_fps
    else:
        frame_skip = max(1, int(original_fps / target_fps))
        effective_fps = original_fps / frame_skip

    print(f"Video info:")
    print(f"  - Resolution: {width}x{height}")
    print(f"  - Original FPS: {original_fps:.2f}")
    print(f"  - Total frames: {total_frames}")
    print(f"  - Processing FPS: {effective_fps:.2f}")
    print(f"  - Frame skip: {frame_skip}")

    # Extract frames
    frames = []
    frame_indices = []
    frame_idx = 0

    with tqdm(desc="Extracting frames", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_skip == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                frame_indices.append(frame_idx)

                if max_frames and len(frames) >= max_frames:
                    break

            frame_idx += 1
            pbar.update(1)

    cap.release()

    return frames, frame_indices, (width, height), effective_fps


def create_video_from_frames(
    frames: list, output_path: str, fps: float, size: tuple
):
    """Create video from frame list."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, size)

    for frame in frames:
        # Convert RGB to BGR for OpenCV
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            frame_bgr = frame
        out.write(frame_bgr)

    out.release()


def main():
    """Main execution function."""
    args = parse_args()

    # Validate input
    if not os.path.exists(args.input):
        print(f"Error: Input video not found: {args.input}")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.save_frames:
        frames_dir = output_dir / "frames"
        frames_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Depth Anything 3 - Video Depth Estimation")
    print("=" * 60)
    print(f"Input video:    {args.input}")
    print(f"Output dir:     {args.output}")
    print(f"Model:          {args.model}")
    print(f"Device:         {args.device}")
    print("=" * 60)

    # Extract frames
    print("\n[1/4] Extracting frames from video...")
    try:
        frames, frame_indices, video_size, effective_fps = extract_frames(
            args.input, args.fps, args.max_frames
        )
        print(f"✓ Extracted {len(frames)} frames")
    except Exception as e:
        print(f"✗ Error extracting frames: {e}")
        sys.exit(1)

    # Load model
    print("\n[2/4] Loading model...")
    try:
        model = DepthAnything3.from_pretrained(args.model)
        model = model.to(args.device)
        model.eval()
        print(f"✓ Model loaded successfully on {args.device}")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        sys.exit(1)

    # Process frames
    print("\n[3/4] Processing frames...")
    depth_maps = []
    depth_colored_frames = []

    try:
        with tqdm(total=len(frames), desc="Processing", unit="frame") as pbar:
            for i, frame in enumerate(frames):
                # Save frame to temporary file for API
                with tempfile.NamedTemporaryFile(
                    suffix=".png", delete=False
                ) as tmp_file:
                    Image.fromarray(frame).save(tmp_file.name)
                    tmp_path = tmp_file.name

                try:
                    with torch.no_grad():
                        prediction = model.inference(
                            image=[tmp_path],
                            process_res=args.process_res,
                            process_res_method="upper_bound_resize",
                        )

                    depth = prediction.depth[0]
                    depth_maps.append(depth)

                    # Create colored depth map
                    depth_colored = visualize_depth(depth, cmap="Spectral")
                    depth_colored_frames.append(depth_colored)

                    # Save individual frame if requested
                    if args.save_frames:
                        frame_path = frames_dir / f"frame_{frame_indices[i]:06d}_depth.png"
                        Image.fromarray(depth_colored).save(frame_path)

                finally:
                    # Clean up temporary file
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)

                pbar.update(1)

        print(f"✓ Processed {len(frames)} frames successfully")

    except Exception as e:
        print(f"\n✗ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Create output videos
    print("\n[4/4] Creating output videos...")
    try:
        if args.create_video:
            # Depth video
            depth_video_path = str(output_dir / "depth_output.mp4")
            create_video_from_frames(
                depth_colored_frames, depth_video_path, effective_fps, video_size
            )
            print(f"✓ Depth video saved: {depth_video_path}")

        if args.side_by_side:
            # Side-by-side comparison
            comparison_frames = []
            for orig, depth_col in zip(frames, depth_colored_frames):
                # Resize depth to match original if needed
                if orig.shape[:2] != depth_col.shape[:2]:
                    depth_col = cv2.resize(
                        depth_col, (orig.shape[1], orig.shape[0])
                    )
                comparison = np.hstack([orig, depth_col])
                comparison_frames.append(comparison)

            comparison_path = str(output_dir / "comparison.mp4")
            create_video_from_frames(
                comparison_frames,
                comparison_path,
                effective_fps,
                (video_size[0] * 2, video_size[1]),
            )
            print(f"✓ Comparison video saved: {comparison_path}")

        # Save depth data
        npz_path = output_dir / "depth_data.npz"
        np.savez_compressed(
            npz_path,
            depth=np.array(depth_maps),
            frame_indices=np.array(frame_indices),
        )
        print(f"✓ Depth data saved: {npz_path}")

    except Exception as e:
        print(f"✗ Error creating videos: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("✓ Video processing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
