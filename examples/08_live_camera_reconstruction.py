#!/usr/bin/env python3
"""
Live Camera 3D Reconstruction

This script performs real-time 3D reconstruction from a USB camera or webcam.
It continuously captures frames, estimates depth, and builds a growing point cloud.

Controls:
    - SPACE: Capture frame and add to reconstruction
    - 'a': Toggle auto-capture mode
    - 's': Save current reconstruction
    - 'c': Clear reconstruction
    - 'q' or ESC: Quit

Usage:
    # Use default camera (camera 0)
    python 08_live_camera_reconstruction.py --output ./live_reconstruction

    # Use specific camera
    python 08_live_camera_reconstruction.py --camera 1

    # Auto-capture every 30 frames
    python 08_live_camera_reconstruction.py --auto-capture --capture-interval 30
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import cv2
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
        description="Depth Anything 3 - Live Camera 3D Reconstruction"
    )
    parser.add_argument(
        "--camera",
        "-c",
        type=int,
        default=0,
        help="Camera device ID (default: 0)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="./live_reconstruction",
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
        "--camera-width",
        type=int,
        default=640,
        help="Camera capture width (default: 640)",
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=480,
        help="Camera capture height (default: 480)",
    )
    parser.add_argument(
        "--auto-capture",
        action="store_true",
        help="Enable automatic frame capture",
    )
    parser.add_argument(
        "--capture-interval",
        type=int,
        default=30,
        help="Frames between auto-captures (default: 30)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Minimum confidence threshold for points (default: 0.5)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=100,
        help="Maximum frames to capture (default: 100)",
    )
    parser.add_argument(
        "--show-depth",
        action="store_true",
        help="Show depth visualization in separate window",
    )
    return parser.parse_args()


class LiveReconstructor:
    """Live 3D reconstruction from camera."""

    def __init__(
        self,
        model: DepthAnything3,
        device: str,
        process_res: int = 504,
        min_confidence: float = 0.5,
    ):
        self.model = model
        self.device = device
        self.process_res = process_res
        self.min_confidence = min_confidence

        # Reconstruction data
        self.all_points: List[np.ndarray] = []
        self.all_colors: List[np.ndarray] = []
        self.captured_frames = 0

        # Processing stats
        self.last_inference_time = 0.0

    def process_frame(self, frame: np.ndarray) -> tuple:
        """
        Process a single frame and extract point cloud.

        Returns:
            depth, points, colors, inference_time
        """
        import tempfile

        # Save frame temporarily
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            Image.fromarray(frame).save(tmp.name)
            tmp_path = tmp.name

        try:
            # Run inference
            start_time = time.time()

            with torch.no_grad():
                prediction = self.model.inference(
                    image=[tmp_path],
                    process_res=self.process_res,
                    process_res_method="upper_bound_resize",
                )

            inference_time = time.time() - start_time

            depth = prediction.depth[0]
            conf = prediction.conf[0] if prediction.conf is not None else None
            intrinsics = (
                prediction.intrinsics[0] if prediction.intrinsics is not None else None
            )

            # Generate point cloud
            points, colors = self._depth_to_pointcloud(
                depth, frame, intrinsics, conf
            )

            return depth, points, colors, inference_time

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def _depth_to_pointcloud(
        self,
        depth: np.ndarray,
        image: np.ndarray,
        intrinsics: Optional[np.ndarray],
        confidence: Optional[np.ndarray],
    ) -> tuple:
        """Convert depth map to point cloud."""
        h, w = depth.shape

        # Use default intrinsics if not available
        if intrinsics is None:
            fx = fy = max(h, w)
            cx, cy = w / 2, h / 2
        else:
            fx, fy = intrinsics[0, 0], intrinsics[1, 1]
            cx, cy = intrinsics[0, 2], intrinsics[1, 2]

        # Create pixel coordinates
        u, v = np.meshgrid(np.arange(w), np.arange(h))

        # Filter by confidence
        if confidence is not None:
            mask = confidence >= self.min_confidence
        else:
            mask = np.ones_like(depth, dtype=bool)

        # Get valid points
        valid_u = u[mask]
        valid_v = v[mask]
        valid_depth = depth[mask]

        # Unproject to 3D
        x = (valid_u - cx) * valid_depth / fx
        y = (valid_v - cy) * valid_depth / fy
        z = valid_depth

        points = np.stack([x, y, z], axis=-1)
        colors = image[mask]

        return points, colors

    def add_frame(self, points: np.ndarray, colors: np.ndarray):
        """Add a frame's point cloud to the reconstruction."""
        self.all_points.append(points)
        self.all_colors.append(colors)
        self.captured_frames += 1

    def get_merged_pointcloud(self) -> tuple:
        """Get merged point cloud."""
        if not self.all_points:
            return np.array([]), np.array([])

        merged_points = np.vstack(self.all_points)
        merged_colors = np.vstack(self.all_colors)

        return merged_points, merged_colors

    def clear(self):
        """Clear reconstruction."""
        self.all_points.clear()
        self.all_colors.clear()
        self.captured_frames = 0

    def save(self, output_dir: Path):
        """Save reconstruction."""
        output_dir.mkdir(parents=True, exist_ok=True)

        points, colors = self.get_merged_pointcloud()

        if len(points) == 0:
            print("No points to save!")
            return

        # Save PLY
        ply_path = output_dir / f"live_reconstruction_{int(time.time())}.ply"
        self._save_ply(points, colors, str(ply_path))

        # Save NPZ
        npz_path = output_dir / f"live_reconstruction_{int(time.time())}.npz"
        np.savez_compressed(npz_path, points=points, colors=colors)

        print(f"\n✓ Saved reconstruction:")
        print(f"  - PLY: {ply_path}")
        print(f"  - NPZ: {npz_path}")
        print(f"  - Total points: {len(points):,}")

    @staticmethod
    def _save_ply(points: np.ndarray, colors: np.ndarray, output_path: str):
        """Save point cloud as PLY."""
        try:
            from plyfile import PlyData, PlyElement
        except ImportError:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "plyfile"])
            from plyfile import PlyData, PlyElement

        # Ensure colors are in 0-255 range
        if colors.max() <= 1.0:
            colors = (colors * 255).astype(np.uint8)
        else:
            colors = colors.astype(np.uint8)

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


def draw_ui(
    frame: np.ndarray,
    captured_frames: int,
    total_points: int,
    inference_time: float,
    auto_capture: bool,
) -> np.ndarray:
    """Draw UI overlay on frame."""
    overlay = frame.copy()
    h, w = frame.shape[:2]

    # Draw semi-transparent overlay
    cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
    frame_with_ui = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

    # Text settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    color = (0, 255, 0)

    # Draw info
    y_offset = 25
    cv2.putText(
        frame_with_ui,
        f"Frames captured: {captured_frames}",
        (10, y_offset),
        font,
        font_scale,
        color,
        thickness,
    )

    y_offset += 25
    cv2.putText(
        frame_with_ui,
        f"Total points: {total_points:,}",
        (10, y_offset),
        font,
        font_scale,
        color,
        thickness,
    )

    y_offset += 25
    fps = 1.0 / inference_time if inference_time > 0 else 0
    cv2.putText(
        frame_with_ui,
        f"Inference: {inference_time*1000:.0f}ms ({fps:.1f} FPS)",
        (10, y_offset),
        font,
        font_scale,
        color,
        thickness,
    )

    y_offset += 25
    mode = "AUTO" if auto_capture else "MANUAL"
    cv2.putText(
        frame_with_ui,
        f"Mode: {mode}",
        (10, y_offset),
        font,
        font_scale,
        (0, 255, 255) if auto_capture else color,
        thickness,
    )

    # Draw controls at bottom
    controls = [
        "SPACE: Capture | 'a': Auto | 's': Save | 'c': Clear | 'q': Quit"
    ]
    y_offset = h - 15
    cv2.putText(
        frame_with_ui,
        controls[0],
        (10, y_offset),
        font,
        0.5,
        (255, 255, 255),
        1,
    )

    return frame_with_ui


def main():
    """Main execution function."""
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Depth Anything 3 - Live Camera 3D Reconstruction")
    print("=" * 70)
    print(f"\nControls:")
    print("  SPACE     - Capture current frame")
    print("  'a'       - Toggle auto-capture mode")
    print("  's'       - Save reconstruction")
    print("  'c'       - Clear reconstruction")
    print("  'q'/ESC   - Quit")
    print("=" * 70)

    # Load model
    print(f"\nLoading model: {args.model}")
    try:
        model = DepthAnything3.from_pretrained(args.model)
        model = model.to(args.device)
        model.eval()
        print(f"✓ Model loaded on {args.device}")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        sys.exit(1)

    # Initialize reconstructor
    reconstructor = LiveReconstructor(
        model, args.device, args.process_res, args.min_confidence
    )

    # Open camera
    print(f"\nOpening camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        print(f"✗ Cannot open camera {args.camera}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"✓ Camera opened: {actual_width}x{actual_height}")
    print("\nStarting live reconstruction...\n")

    # Main loop
    auto_capture = args.auto_capture
    frame_count = 0
    last_depth = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_count += 1

            # Auto-capture logic
            should_capture = False
            if auto_capture and frame_count % args.capture_interval == 0:
                should_capture = True

            # Process frame if capturing
            if should_capture and reconstructor.captured_frames < args.max_frames:
                print(f"Processing frame {reconstructor.captured_frames + 1}...", end=" ")

                depth, points, colors, inference_time = reconstructor.process_frame(
                    frame_rgb
                )
                reconstructor.add_frame(points, colors)
                last_depth = depth

                print(
                    f"Done! ({len(points):,} points, {inference_time*1000:.0f}ms)"
                )

                # Show depth if requested
                if args.show_depth and last_depth is not None:
                    depth_colored = visualize_depth(last_depth, cmap="Spectral")
                    depth_bgr = cv2.cvtColor(depth_colored, cv2.COLOR_RGB2BGR)
                    cv2.imshow("Depth", depth_bgr)

            # Get current point count
            points, _ = reconstructor.get_merged_pointcloud()
            total_points = len(points)

            # Draw UI
            frame_with_ui = draw_ui(
                frame,
                reconstructor.captured_frames,
                total_points,
                reconstructor.last_inference_time
                if should_capture
                else 0.03,  # Estimate
                auto_capture,
            )

            # Show frame
            cv2.imshow("Live 3D Reconstruction", frame_with_ui)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:  # 'q' or ESC
                print("\nQuitting...")
                break

            elif key == ord(" "):  # SPACE - manual capture
                if reconstructor.captured_frames < args.max_frames:
                    print(f"\nCapturing frame {reconstructor.captured_frames + 1}...", end=" ")
                    depth, points, colors, inference_time = (
                        reconstructor.process_frame(frame_rgb)
                    )
                    reconstructor.add_frame(points, colors)
                    reconstructor.last_inference_time = inference_time
                    last_depth = depth
                    print(f"Done! ({len(points):,} points)")

                    if args.show_depth:
                        depth_colored = visualize_depth(depth, cmap="Spectral")
                        depth_bgr = cv2.cvtColor(depth_colored, cv2.COLOR_RGB2BGR)
                        cv2.imshow("Depth", depth_bgr)
                else:
                    print(f"\nMax frames ({args.max_frames}) reached!")

            elif key == ord("a"):  # Toggle auto-capture
                auto_capture = not auto_capture
                mode = "AUTO" if auto_capture else "MANUAL"
                print(f"\nMode: {mode}")

            elif key == ord("s"):  # Save
                print("\nSaving reconstruction...")
                reconstructor.save(output_dir)

            elif key == ord("c"):  # Clear
                reconstructor.clear()
                print("\nReconstruction cleared!")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()

        # Save if we have data
        if reconstructor.captured_frames > 0:
            print("\nSaving final reconstruction...")
            reconstructor.save(output_dir)

    print("\n" + "=" * 70)
    print("✓ Live reconstruction session ended")
    print("=" * 70)


if __name__ == "__main__":
    main()
