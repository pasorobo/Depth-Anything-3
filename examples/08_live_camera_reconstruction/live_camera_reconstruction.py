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
from depth_anything_3.utils.geometry import affine_inverse
from depth_anything_3.utils.visualize import visualize_depth


class FrameBuffer:
    """
    Circular buffer to store recent frames for multi-view pose estimation.
    """

    def __init__(self, max_size: int = 5):
        """
        Initialize frame buffer.

        Args:
            max_size: Maximum number of frames to keep
        """
        self.max_size = max_size
        self.frames: List[np.ndarray] = []
        self.timestamps: List[float] = []

    def add(self, frame: np.ndarray, timestamp: Optional[float] = None):
        """Add frame to buffer."""
        if timestamp is None:
            timestamp = time.time()

        self.frames.append(frame)
        self.timestamps.append(timestamp)

        # Remove oldest if exceeds max size
        if len(self.frames) > self.max_size:
            self.frames.pop(0)
            self.timestamps.pop(0)

    def get_all(self) -> List[np.ndarray]:
        """Get all frames in buffer."""
        return self.frames.copy()

    def get_recent(self, n: int) -> List[np.ndarray]:
        """Get n most recent frames."""
        return self.frames[-n:] if n <= len(self.frames) else self.frames.copy()

    def clear(self):
        """Clear buffer."""
        self.frames.clear()
        self.timestamps.clear()

    def __len__(self):
        return len(self.frames)


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
    parser.add_argument(
        "--downsample",
        action="store_true",
        help="Enable voxel downsampling of point clouds",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.01,
        help="Voxel size for downsampling in meters (default: 0.01)",
    )
    parser.add_argument(
        "--max-points-per-frame",
        type=int,
        default=500000,
        help="Maximum points per frame (default: 500000)",
    )
    return parser.parse_args()


class LiveReconstructor:
    """Live 3D reconstruction from camera with pose estimation."""

    def __init__(
        self,
        model: DepthAnything3,
        device: str,
        process_res: int = 504,
        min_confidence: float = 0.5,
        frame_buffer_size: int = 5,
        use_pose_estimation: bool = True,
    ):
        self.model = model
        self.device = device
        self.process_res = process_res
        self.min_confidence = min_confidence
        self.use_pose_estimation = use_pose_estimation

        # Frame buffer for pose estimation
        self.frame_buffer = FrameBuffer(max_size=frame_buffer_size)

        # Reconstruction data
        self.all_points: List[np.ndarray] = []
        self.all_colors: List[np.ndarray] = []
        self.captured_frames = 0
        self.total_points = 0  # Track incrementally for performance

        # World coordinate reference (first captured frame defines origin)
        self.world_reference_set = False
        self.first_extrinsics = None  # First frame's camera pose

        # Processing stats
        self.last_inference_time = 0.0

    def process_frame(self, frame: np.ndarray) -> tuple:
        """
        Process frame with multi-view pose estimation and 3D reconstruction.

        Args:
            frame: RGB frame as numpy array (H, W, 3)

        Returns:
            depth, points_world, colors, inference_time, extrinsics, intrinsics
        """
        # Add frame to buffer
        pil_image = Image.fromarray(frame)
        self.frame_buffer.add(frame)

        # Prepare images for inference
        if self.use_pose_estimation and len(self.frame_buffer) >= 2:
            # Use multiple frames for pose estimation
            images = [Image.fromarray(f) for f in self.frame_buffer.get_all()]
        else:
            # Single frame mode (fallback)
            images = [pil_image]

        # Run inference
        start_time = time.time()

        with torch.no_grad():
            prediction = self.model.inference(
                image=images,
                process_res=self.process_res,
                process_res_method="upper_bound_resize",
            )

        inference_time = time.time() - start_time

        # Get predictions for the latest frame
        idx = -1  # Last frame
        depth = prediction.depth[idx]
        conf = prediction.conf[idx] if prediction.conf is not None else None
        extrinsics = prediction.extrinsics[idx] if prediction.extrinsics is not None else None
        intrinsics = prediction.intrinsics[idx] if prediction.intrinsics is not None else None

        if self.use_pose_estimation and extrinsics is None:
            if not self.world_reference_set:
                print(
                    "Pose estimation not ready yet (need multiple frames). "
                    "Skipping capture to keep coordinates aligned."
                )
            else:
                print("Pose estimation failed for this frame, skipping capture to keep alignment consistent.")
            return depth, None, None, inference_time, extrinsics, intrinsics

        # Generate point cloud in camera coordinates
        points_camera, colors = self._depth_to_pointcloud(depth, frame, intrinsics, conf)

        # Transform to world coordinates if pose available
        if self.use_pose_estimation and extrinsics is not None:
            # Establish world reference on first capture
            if not self.world_reference_set:
                self.first_extrinsics = extrinsics.copy()
                self.world_reference_set = True
                # First frame is at origin in world coordinates
                # Apply identity-like transformation (camera coords become world coords)
                points_world = points_camera
                print("✓ World reference established (first frame at origin)")
            else:
                # Transform relative to first frame
                points_world = self._transform_to_world_relative(
                    points_camera, extrinsics, self.first_extrinsics
                )
        else:
            # Fallback: use camera coordinates
            points_world = points_camera

        return depth, points_world, colors, inference_time, extrinsics, intrinsics

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

    def _transform_to_world_relative(
        self,
        points_camera: np.ndarray,
        current_extrinsics: np.ndarray,
        reference_extrinsics: np.ndarray
    ) -> np.ndarray:
        """
        Transform points to world coordinates relative to reference frame.

        This ensures all captured point clouds share the same world coordinate system
        by transforming them relative to the first captured frame.

        Args:
            points_camera: Nx3 array of points in current camera coordinates
            current_extrinsics: Current frame's camera extrinsics (w2c)
            reference_extrinsics: Reference frame's camera extrinsics (w2c)

        Returns:
            Nx3 array of points in world coordinates
        """
        if len(points_camera) == 0:
            return points_camera

        # Ensure both are 4x4
        if current_extrinsics.shape == (3, 4):
            current_extrinsics = np.vstack([current_extrinsics, [0, 0, 0, 1]])
        if reference_extrinsics.shape == (3, 4):
            reference_extrinsics = np.vstack([reference_extrinsics, [0, 0, 0, 1]])

        # Compute relative transformation
        # ref_c2w: reference camera to world (world = reference camera space)
        # cur_c2w: current camera to world
        # We want: cur_cam -> world, where world is defined by reference camera
        ref_c2w = affine_inverse(reference_extrinsics)
        cur_c2w = affine_inverse(current_extrinsics)

        # Relative transformation: from current camera to reference camera space
        # T_rel = ref_w2c @ cur_c2w
        relative_transform = reference_extrinsics @ cur_c2w

        # Convert points to homogeneous coordinates
        points_homogeneous = np.hstack([
            points_camera,
            np.ones((len(points_camera), 1))
        ])

        # Apply transformation (current camera -> reference/world coordinates)
        points_world_homogeneous = (relative_transform @ points_homogeneous.T).T

        # Convert back to 3D
        points_world = points_world_homogeneous[:, :3]

        return points_world

    def add_frame(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        downsample: bool = False,
        voxel_size: float = 0.01,
        max_points: int = 500000
    ):
        """
        Add a frame's point cloud to the reconstruction.

        Args:
            points: Nx3 array of 3D points in world coordinates
            colors: Nx3 array of RGB colors
            downsample: If True, apply voxel downsampling
            voxel_size: Voxel size for downsampling (in meters)
            max_points: Maximum points per frame (random downsample if exceeded)
        Returns:
            Number of points that were appended
        """
        # Random downsampling if too many points
        if len(points) > max_points:
            indices = np.random.choice(len(points), max_points, replace=False)
            points = points[indices]
            colors = colors[indices]

        # Voxel downsampling
        if downsample and len(points) > 0:
            points, colors = self._voxel_downsample(points, colors, voxel_size)

        appended_points = len(points)

        self.all_points.append(points)
        self.all_colors.append(colors)
        self.captured_frames += 1
        self.total_points += appended_points  # Track incrementally for O(1) access

        return appended_points

    def _voxel_downsample(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        voxel_size: float
    ) -> tuple:
        """
        Downsample point cloud using voxel grid filtering.

        Args:
            points: Nx3 array of 3D points
            colors: Nx3 array of RGB colors
            voxel_size: Size of voxel grid

        Returns:
            Downsampled points and colors
        """
        if len(points) == 0:
            return points, colors

        # Compute voxel indices
        voxel_indices = np.floor(points / voxel_size).astype(np.int32)

        # Create unique key for each voxel
        voxel_dict = {}
        for i in range(len(points)):
            key = tuple(voxel_indices[i])
            if key not in voxel_dict:
                voxel_dict[key] = []
            voxel_dict[key].append(i)

        # Average points and colors in each voxel
        downsampled_points = []
        downsampled_colors = []

        for indices in voxel_dict.values():
            # Average position
            avg_point = points[indices].mean(axis=0)
            # Average color
            avg_color = colors[indices].mean(axis=0)

            downsampled_points.append(avg_point)
            downsampled_colors.append(avg_color)

        return np.array(downsampled_points), np.array(downsampled_colors)

    def check_memory_warning(self, max_points: int = 10_000_000):
        """
        Check if memory usage exceeds thresholds and warn user.

        Args:
            max_points: Maximum total points before warning

        Returns:
            True if warning issued
        """
        if self.total_points > max_points:
            # Estimate memory usage (rough calculation)
            # Each point: 3 floats (12 bytes) + 3 uint8 (3 bytes) = ~15 bytes
            estimated_mb = (self.total_points * 15) / (1024 * 1024)

            print(f"\n⚠ Warning: {self.total_points:,} points accumulated")
            print(f"  Estimated memory: {estimated_mb:.1f} MB")
            print("  Consider saving (s) and clearing (c) to free memory")
            return True

        return False

    def get_merged_pointcloud(self) -> tuple:
        """Get merged point cloud."""
        if not self.all_points:
            return np.array([]), np.array([])

        merged_points = np.vstack(self.all_points)
        merged_colors = np.vstack(self.all_colors)

        return merged_points, merged_colors

    def clear(self):
        """Clear reconstruction and reset world reference."""
        self.all_points.clear()
        self.all_colors.clear()
        self.captured_frames = 0
        self.total_points = 0
        self.frame_buffer.clear()
        self.world_reference_set = False
        self.first_extrinsics = None

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
        """
        Save point cloud as PLY format.

        Args:
            points: Nx3 array of 3D points
            colors: Nx3 array of RGB colors (0-255 or 0-1)
            output_path: Output PLY file path

        Raises:
            ImportError: If plyfile is not installed
        """
        try:
            from plyfile import PlyData, PlyElement
        except ImportError as e:
            raise ImportError(
                "plyfile is required for PLY export.\n"
                "Install it with: pip install plyfile\n"
                "Or install all dependencies: pip install -r requirements.txt"
            ) from e

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


def load_model(model_name: str, device: str) -> DepthAnything3:
    """
    Load Depth Anything 3 model with detailed error handling.

    Args:
        model_name: Model name or path
        device: Device to use (cuda/cpu)

    Returns:
        Loaded model

    Raises:
        Various exceptions with helpful messages
    """
    print(f"\nLoading model: {model_name}")

    try:
        model = DepthAnything3.from_pretrained(model_name)
        model = model.to(device)
        model.eval()
        print(f"✓ Model loaded on {device}")
        return model

    except FileNotFoundError as e:
        print(f"✗ Model not found: {model_name}")
        print("\nAvailable models:")
        print("  - depth-anything/DA3-LARGE (recommended)")
        print("  - depth-anything/DA3-BASE (faster)")
        print("  - depth-anything/DA3-GIANT (best quality)")
        print(f"\nTry: python {sys.argv[0]} --model depth-anything/DA3-LARGE")
        raise

    except RuntimeError as e:
        error_msg = str(e)
        if "CUDA" in error_msg or "cuda" in error_msg:
            print(f"✗ CUDA error: {e}")
            print("\nPossible solutions:")
            print("  - Use CPU instead: --device cpu")
            print("  - Check CUDA availability: python -c 'import torch; print(torch.cuda.is_available())'")
            print("  - Update GPU drivers")
        elif "out of memory" in error_msg.lower():
            print(f"✗ GPU out of memory")
            print("\nSolutions:")
            print("  - Use smaller model: --model depth-anything/DA3-BASE")
            print("  - Lower resolution: --process-res 336")
            print("  - Use CPU: --device cpu")
        else:
            print(f"✗ Runtime error: {e}")
        raise

    except Exception as e:
        print(f"✗ Unexpected error loading model: {e}")
        print(f"  Error type: {type(e).__name__}")
        raise


def open_camera(camera_id: int, width: int, height: int) -> cv2.VideoCapture:
    """
    Open camera with detailed error handling.

    Args:
        camera_id: Camera device ID
        width: Desired capture width
        height: Desired capture height

    Returns:
        Opened VideoCapture object

    Raises:
        RuntimeError: If camera cannot be opened
    """
    print(f"\nOpening camera {camera_id}...")
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print(f"✗ Cannot open camera {camera_id}")
        print("\nTroubleshooting:")
        print("  - Try different camera IDs: --camera 0, --camera 1, --camera 2")
        print("  - Check if camera is in use by another application")

        # Try to detect available cameras
        print("\nDetecting available cameras...")
        available = []
        for i in range(5):
            test_cap = cv2.VideoCapture(i)
            if test_cap.isOpened():
                available.append(i)
                test_cap.release()

        if available:
            print(f"  ✓ Available cameras: {', '.join(map(str, available))}")
            print(f"  Try: python {sys.argv[0]} --camera {available[0]}")
        else:
            print("  ✗ No cameras detected")
            print("\nOn Linux, check device permissions:")
            print("  sudo usermod -a -G video $USER")
            print("  ls -l /dev/video*")

        raise RuntimeError(f"Camera {camera_id} not available")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"✓ Camera {camera_id} opened: {actual_width}x{actual_height}")

    return cap


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

    # Load model with improved error handling
    try:
        model = load_model(args.model, args.device)
    except Exception:
        sys.exit(1)

    # Initialize reconstructor
    reconstructor = LiveReconstructor(
        model, args.device, args.process_res, args.min_confidence
    )

    # Open camera with improved error handling
    try:
        cap = open_camera(args.camera, args.camera_width, args.camera_height)
    except Exception:
        sys.exit(1)

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

                depth, points, colors, inference_time, extrinsics, intrinsics = reconstructor.process_frame(
                    frame_rgb
                )

                if points is None:
                    print("Skipped (pose not ready).")
                else:
                    appended = reconstructor.add_frame(
                        points,
                        colors,
                        downsample=args.downsample,
                        voxel_size=args.voxel_size,
                        max_points=args.max_points_per_frame,
                    )
                    reconstructor.last_inference_time = inference_time
                    last_depth = depth

                    print(
                        f"Done! ({appended:,} points, {inference_time*1000:.0f}ms)"
                    )

                    # Check memory usage every 10 frames
                    if reconstructor.captured_frames % 10 == 0:
                        reconstructor.check_memory_warning()

                    # Show depth if requested
                    if args.show_depth and last_depth is not None:
                        depth_colored = visualize_depth(last_depth, cmap="Spectral")
                        depth_bgr = cv2.cvtColor(depth_colored, cv2.COLOR_RGB2BGR)
                        cv2.imshow("Depth", depth_bgr)

            # Get current point count (O(1) with incremental tracking)
            total_points = reconstructor.total_points

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
                    depth, points, colors, inference_time, extrinsics, intrinsics = (
                        reconstructor.process_frame(frame_rgb)
                    )

                    if points is None:
                        print("Skipped (pose not ready).")
                    else:
                        appended = reconstructor.add_frame(
                            points,
                            colors,
                            downsample=args.downsample,
                            voxel_size=args.voxel_size,
                            max_points=args.max_points_per_frame,
                        )
                        reconstructor.last_inference_time = inference_time
                        last_depth = depth
                        print(f"Done! ({appended:,} points)")

                        # Check memory usage
                        if reconstructor.captured_frames % 10 == 0:
                            reconstructor.check_memory_warning()

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
