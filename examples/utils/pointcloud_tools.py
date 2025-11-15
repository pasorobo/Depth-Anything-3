#!/usr/bin/env python3
"""
Point Cloud Processing Tools

Utility script for processing, merging, and filtering point clouds from
3D reconstruction results.

Usage:
    # Merge multiple PLY files
    python pointcloud_tools.py merge --input file1.ply file2.ply --output merged.ply

    # Downsample point cloud
    python pointcloud_tools.py downsample --input cloud.ply --output downsampled.ply --ratio 0.5

    # Filter by distance
    python pointcloud_tools.py filter --input cloud.ply --output filtered.ply --max-distance 10.0

    # Convert NPZ to PLY
    python pointcloud_tools.py convert --input cloud.npz --output cloud.ply
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np


def load_ply(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load point cloud from PLY file."""
    try:
        from plyfile import PlyData
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "plyfile"])
        from plyfile import PlyData

    plydata = PlyData.read(filepath)
    vertex = plydata["vertex"]

    points = np.vstack([vertex["x"], vertex["y"], vertex["z"]]).T

    if "red" in vertex:
        colors = np.vstack([vertex["red"], vertex["green"], vertex["blue"]]).T
    else:
        colors = np.ones((len(points), 3)) * 128

    return points, colors


def save_ply(points: np.ndarray, colors: np.ndarray, filepath: str):
    """Save point cloud as PLY file."""
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
    PlyData([el]).write(filepath)


def load_npz(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load point cloud from NPZ file."""
    data = np.load(filepath)
    points = data["points"]
    colors = data["colors"]
    return points, colors


def save_npz(points: np.ndarray, colors: np.ndarray, filepath: str):
    """Save point cloud as NPZ file."""
    np.savez_compressed(filepath, points=points, colors=colors)


def merge_pointclouds(
    input_files: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Merge multiple point clouds."""
    all_points = []
    all_colors = []

    for filepath in input_files:
        filepath = Path(filepath)
        print(f"Loading {filepath.name}...")

        if filepath.suffix == ".ply":
            points, colors = load_ply(str(filepath))
        elif filepath.suffix == ".npz":
            points, colors = load_npz(str(filepath))
        else:
            print(f"Warning: Unsupported format {filepath.suffix}, skipping")
            continue

        all_points.append(points)
        all_colors.append(colors)
        print(f"  Loaded {len(points):,} points")

    merged_points = np.vstack(all_points)
    merged_colors = np.vstack(all_colors)

    return merged_points, merged_colors


def downsample_pointcloud(
    points: np.ndarray, colors: np.ndarray, ratio: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Downsample point cloud by random sampling."""
    n_points = len(points)
    n_samples = int(n_points * ratio)

    indices = np.random.choice(n_points, n_samples, replace=False)

    return points[indices], colors[indices]


def downsample_voxel(
    points: np.ndarray, colors: np.ndarray, voxel_size: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Downsample point cloud using voxel grid."""
    # Voxelize
    voxel_indices = np.floor(points / voxel_size).astype(int)

    # Create unique voxel IDs
    voxel_ids = (
        voxel_indices[:, 0].astype(np.int64) * 1000000000
        + voxel_indices[:, 1].astype(np.int64) * 1000000
        + voxel_indices[:, 2].astype(np.int64)
    )

    # Get unique voxels and their indices
    unique_voxels, inverse_indices = np.unique(voxel_ids, return_inverse=True)

    # Average points in each voxel
    downsampled_points = []
    downsampled_colors = []

    for i in range(len(unique_voxels)):
        mask = inverse_indices == i
        downsampled_points.append(points[mask].mean(axis=0))
        downsampled_colors.append(colors[mask].mean(axis=0))

    return np.array(downsampled_points), np.array(downsampled_colors)


def filter_by_distance(
    points: np.ndarray,
    colors: np.ndarray,
    max_distance: float = None,
    min_distance: float = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Filter points by distance from origin."""
    distances = np.linalg.norm(points, axis=1)

    mask = np.ones(len(points), dtype=bool)

    if max_distance is not None:
        mask &= distances <= max_distance

    if min_distance is not None:
        mask &= distances >= min_distance

    return points[mask], colors[mask]


def filter_statistical_outliers(
    points: np.ndarray, colors: np.ndarray, k: int = 20, std_ratio: float = 2.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove statistical outliers based on distance to neighbors."""
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        print("Warning: scipy not available, skipping outlier removal")
        return points, colors

    tree = cKDTree(points)

    # Find k nearest neighbors for each point
    distances, _ = tree.query(points, k=k + 1)  # +1 because point is its own neighbor

    # Compute mean distance to neighbors (excluding self)
    mean_distances = distances[:, 1:].mean(axis=1)

    # Compute global statistics
    global_mean = mean_distances.mean()
    global_std = mean_distances.std()

    # Filter outliers
    threshold = global_mean + std_ratio * global_std
    mask = mean_distances < threshold

    return points[mask], colors[mask]


def compute_statistics(points: np.ndarray, colors: np.ndarray):
    """Compute and print point cloud statistics."""
    print("\nPoint Cloud Statistics:")
    print("=" * 50)
    print(f"Total points: {len(points):,}")

    print(f"\nBounding box:")
    print(f"  X: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
    print(f"  Y: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
    print(f"  Z: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")

    center = points.mean(axis=0)
    print(f"\nCenter: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]")

    distances = np.linalg.norm(points - center, axis=1)
    print(f"\nDistance from center:")
    print(f"  Mean: {distances.mean():.3f}")
    print(f"  Std:  {distances.std():.3f}")
    print(f"  Max:  {distances.max():.3f}")

    if colors.max() <= 1.0:
        color_range = "0-1 (normalized)"
    else:
        color_range = "0-255 (uint8)"
    print(f"\nColor range: {color_range}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Point Cloud Processing Tools"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Merge command
    merge_parser = subparsers.add_parser("merge", help="Merge multiple point clouds")
    merge_parser.add_argument(
        "--input", "-i", nargs="+", required=True, help="Input point cloud files"
    )
    merge_parser.add_argument(
        "--output", "-o", required=True, help="Output file path"
    )

    # Downsample command
    downsample_parser = subparsers.add_parser(
        "downsample", help="Downsample point cloud"
    )
    downsample_parser.add_argument(
        "--input", "-i", required=True, help="Input point cloud file"
    )
    downsample_parser.add_argument(
        "--output", "-o", required=True, help="Output file path"
    )
    downsample_parser.add_argument(
        "--ratio",
        type=float,
        default=0.5,
        help="Downsampling ratio (default: 0.5)",
    )
    downsample_parser.add_argument(
        "--voxel-size",
        type=float,
        help="Voxel size for voxel grid downsampling (overrides --ratio)",
    )

    # Filter command
    filter_parser = subparsers.add_parser("filter", help="Filter point cloud")
    filter_parser.add_argument(
        "--input", "-i", required=True, help="Input point cloud file"
    )
    filter_parser.add_argument(
        "--output", "-o", required=True, help="Output file path"
    )
    filter_parser.add_argument(
        "--max-distance", type=float, help="Maximum distance from origin"
    )
    filter_parser.add_argument(
        "--min-distance", type=float, help="Minimum distance from origin"
    )
    filter_parser.add_argument(
        "--remove-outliers",
        action="store_true",
        help="Remove statistical outliers",
    )

    # Convert command
    convert_parser = subparsers.add_parser(
        "convert", help="Convert between formats"
    )
    convert_parser.add_argument(
        "--input", "-i", required=True, help="Input point cloud file"
    )
    convert_parser.add_argument(
        "--output", "-o", required=True, help="Output file path"
    )

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show statistics")
    stats_parser.add_argument(
        "--input", "-i", required=True, help="Input point cloud file"
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Execute command
    if args.command == "merge":
        print(f"Merging {len(args.input)} point clouds...")

        points, colors = merge_pointclouds(args.input)

        print(f"\nMerged point cloud: {len(points):,} points")

        output_path = Path(args.output)
        if output_path.suffix == ".ply":
            save_ply(points, colors, str(output_path))
        elif output_path.suffix == ".npz":
            save_npz(points, colors, str(output_path))
        else:
            print(f"Error: Unsupported output format {output_path.suffix}")
            sys.exit(1)

        print(f"✓ Saved to {args.output}")

    elif args.command == "downsample":
        print(f"Loading {args.input}...")

        input_path = Path(args.input)
        if input_path.suffix == ".ply":
            points, colors = load_ply(str(input_path))
        elif input_path.suffix == ".npz":
            points, colors = load_npz(str(input_path))
        else:
            print(f"Error: Unsupported format {input_path.suffix}")
            sys.exit(1)

        print(f"Original: {len(points):,} points")

        if args.voxel_size:
            print(f"Downsampling with voxel size {args.voxel_size}...")
            points, colors = downsample_voxel(points, colors, args.voxel_size)
        else:
            print(f"Downsampling to {args.ratio * 100}%...")
            points, colors = downsample_pointcloud(points, colors, args.ratio)

        print(f"Downsampled: {len(points):,} points")

        output_path = Path(args.output)
        if output_path.suffix == ".ply":
            save_ply(points, colors, str(output_path))
        elif output_path.suffix == ".npz":
            save_npz(points, colors, str(output_path))

        print(f"✓ Saved to {args.output}")

    elif args.command == "filter":
        print(f"Loading {args.input}...")

        input_path = Path(args.input)
        if input_path.suffix == ".ply":
            points, colors = load_ply(str(input_path))
        elif input_path.suffix == ".npz":
            points, colors = load_npz(str(input_path))
        else:
            print(f"Error: Unsupported format {input_path.suffix}")
            sys.exit(1)

        print(f"Original: {len(points):,} points")

        if args.max_distance or args.min_distance:
            print("Filtering by distance...")
            points, colors = filter_by_distance(
                points, colors, args.max_distance, args.min_distance
            )
            print(f"After distance filter: {len(points):,} points")

        if args.remove_outliers:
            print("Removing statistical outliers...")
            points, colors = filter_statistical_outliers(points, colors)
            print(f"After outlier removal: {len(points):,} points")

        output_path = Path(args.output)
        if output_path.suffix == ".ply":
            save_ply(points, colors, str(output_path))
        elif output_path.suffix == ".npz":
            save_npz(points, colors, str(output_path))

        print(f"✓ Saved to {args.output}")

    elif args.command == "convert":
        print(f"Loading {args.input}...")

        input_path = Path(args.input)
        if input_path.suffix == ".ply":
            points, colors = load_ply(str(input_path))
        elif input_path.suffix == ".npz":
            points, colors = load_npz(str(input_path))
        else:
            print(f"Error: Unsupported input format {input_path.suffix}")
            sys.exit(1)

        print(f"Loaded: {len(points):,} points")

        output_path = Path(args.output)
        if output_path.suffix == ".ply":
            save_ply(points, colors, str(output_path))
        elif output_path.suffix == ".npz":
            save_npz(points, colors, str(output_path))
        else:
            print(f"Error: Unsupported output format {output_path.suffix}")
            sys.exit(1)

        print(f"✓ Converted to {args.output}")

    elif args.command == "stats":
        print(f"Loading {args.input}...")

        input_path = Path(args.input)
        if input_path.suffix == ".ply":
            points, colors = load_ply(str(input_path))
        elif input_path.suffix == ".npz":
            points, colors = load_npz(str(input_path))
        else:
            print(f"Error: Unsupported format {input_path.suffix}")
            sys.exit(1)

        compute_statistics(points, colors)


if __name__ == "__main__":
    main()
