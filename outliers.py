"""
Advanced Point Cloud Outlier Removal Tool  (3DGS MVP)
Semantic-aware defaults: AttributeFilter only (MVP removes density-based filters such as SOR/ROR/voxel)
"""

import argparse
import sys
from pathlib import Path
import open3d as o3d
import json

# Add filters directory to path
sys.path.insert(0, str(Path(__file__).parent))

from filters import (
    FilterPipeline,
    AttributeFilter,
    DepthConsistencyFilter,
)
from filters.depth_consistency_filter import load_cameras_from_dir, load_depths_from_dir


def main():
    parser = argparse.ArgumentParser(
        description="Point Cloud Outlier Removal  (Conservative, 3DGS-friendly)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Attribute-only (no geometry filtering; delete only explicit zero opacity)
  python outliers.py -i input.ply -o output.ply
        """,
    )

    # Input/Output
    parser.add_argument("--input", "-i", required=True, help="Input point cloud file")
    parser.add_argument("--output", "-o", required=True, help="Output point cloud file")
    parser.add_argument(
        "--visualize", "-v", action="store_true", help="Visualize final result"
    )

    # Depth consistency filter (multi-view) options
    parser.add_argument(
        "--enable-depth-filter",
        action="store_true",
        help="Enable multi-view depth consistency filter",
    )
    parser.add_argument(
        "--cameras-json",
        type=str,
        default=None,
        help="Path to cameras JSON directory or single JSON file",
    )
    parser.add_argument(
        "--depth-dir",
        type=str,
        default=None,
        help="Directory containing per-view relative depth .npy files",
    )
    parser.add_argument("--max-views", type=int, default=12)
    parser.add_argument("--min-visible-views", type=int, default=4)
    parser.add_argument("--min-inlier-ratio", type=float, default=0.6)
    parser.add_argument("--sigma-b", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=200000)
    parser.add_argument("--fit-subset-points", type=int, default=50000)
    parser.add_argument("--fit-min-samples", type=int, default=1000)
    parser.add_argument(
        "--sampling",
        type=str,
        default="strided",
        choices=["strided", "all"],
        help="View sampling strategy for depth consistency",
    )
    parser.add_argument(
        "--depth-glob",
        type=str,
        default="*.npy",
        help="Glob pattern to select depth files in --depth-dir (e.g., '*rel*.npy' or '*metric*.npy')",
    )

    args = parser.parse_args()

    # Load point cloud
    print(f"\nLoading point cloud from: {args.input}")
    pcd = o3d.io.read_point_cloud(args.input)
    num_points = len(pcd.points)
    print(f"Loaded {num_points:,} points")

    # Load 3DGS metadata when AttributeFilter is requested
    metadata = None
    print("\nAttempting to load 3DGS metadata...")
    metadata = AttributeFilter.load_3dgs_metadata(args.input)
    if metadata:
        print(f"Loaded metadata with keys: {list(metadata.keys())}")
    else:
        print(
            "Warning: Could not load 3DGS metadata; AttributeFilter requires 3DGS fields (opacity/scale/SH)."
        )

    # Load multi-view cameras/depths if requested
    cameras = None
    depths = None
    if args.enable_depth_filter:
        # Default cameras path to input.ply's directory if not provided
        cam_path = (
            Path(args.cameras_json) if args.cameras_json else Path(args.input).parent
        )
        try:
            if cam_path.is_dir():
                cameras = load_cameras_from_dir(str(cam_path))
                print(f"Using cameras from directory: {cam_path}")
            elif cam_path.is_file() and cam_path.suffix.lower() == ".json":
                with open(cam_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    cameras = data
                elif isinstance(data, dict):
                    cameras = [data]
                else:
                    print("Warning: cameras_json not a list/dict; ignoring")
                print(f"Using cameras from file: {cam_path}")
            else:
                # Fallback: try directory scanning on parent of provided path
                fallback_dir = (
                    cam_path if cam_path.exists() else Path(args.input).parent
                )
                cameras = load_cameras_from_dir(str(fallback_dir))
                print(f"Using cameras from directory (fallback): {fallback_dir}")
        except Exception as e:
            print(f"Warning: failed to load cameras from {cam_path}: {e}")

        # Default depth directory to input.ply's directory if not provided
        depth_dir = Path(args.depth_dir) if args.depth_dir else Path(args.input).parent
        try:
            depths = load_depths_from_dir(
                str(depth_dir), pattern=getattr(args, "depth_glob", "*.npy")
            )
            print(
                f"Using depths from directory: {depth_dir} (pattern={getattr(args, 'depth_glob', '*.npy')})"
            )
        except Exception as e:
            print(f"Warning: failed to load depths from {depth_dir}: {e}")

        if cameras and depths:
            if metadata is None:
                metadata = {}
            metadata["cameras"] = cameras
            metadata["depths"] = depths
            print(
                f"Loaded {len(cameras)} camera(s) and {len(depths)} depth map(s) for depth filter"
            )
        else:
            print("Warning: cameras or depths missing; depth filter will be skipped")

    # Build pipeline
    pipeline = FilterPipeline(name="Conservative Pipeline")
    if metadata:
        pipeline.add_filter(
            AttributeFilter(
                enabled=True,
            )
        )
        # Attach depth consistency filter if enabled and inputs are present
        if (
            args.enable_depth_filter
            and metadata.get("cameras")
            and metadata.get("depths")
        ):
            pipeline.add_filter(
                DepthConsistencyFilter(
                    enabled=True,
                    max_views=args.max_views,
                    min_visible_views=args.min_visible_views,
                    min_inlier_ratio=args.min_inlier_ratio,
                    batch_size=args.batch_size,
                    fit_subset_points=args.fit_subset_points,
                    fit_min_samples=args.fit_min_samples,
                    sigma_b=args.sigma_b,
                    sampling=args.sampling,
                )
            )

    # Run pipeline
    filtered_pcd, stats_list = pipeline.run(
        pcd,
        metadata=metadata,
    )

    # Print summary
    print(pipeline.get_summary(stats_list))

    # Save result (default: preserve original 3DGS meta if present)
    print(f"\nSaving cleaned point cloud to: {args.output}")
    try:
        from ply_utils import write_filtered_ply, has_3dgs_meta
    except Exception as e:
        # If ply_utils import fails, fall back to Open3D (attributes may be lost)
        print(
            f"Warning: ply_utils import failed ({e}); falling back to Open3D (attributes may be lost)."
        )
        o3d.io.write_point_cloud(args.output, filtered_pcd)
        print("Saved with Open3D (no 3DGS meta preserved).")
    else:
        if has_3dgs_meta(args.input):
            if (
                hasattr(pipeline, "final_keep_indices")
                and pipeline.final_keep_indices is not None
            ):
                try:
                    write_filtered_ply(
                        args.input, args.output, pipeline.final_keep_indices
                    )
                    print("Saved with plyfile (preserved 3DGS attributes).")
                except Exception as e:
                    print(
                        f"Warning: plyfile save failed ({e}); falling back to Open3D (attributes may be lost)."
                    )
                    o3d.io.write_point_cloud(args.output, filtered_pcd)
                    print("Saved with Open3D (attributes may be lost).")
            else:
                # Strong warning: indices missing prevents meta-preserving write
                print(
                    "Warning: final_keep_indices not available; saved with Open3D (3DGS attributes will be lost)."
                )
                o3d.io.write_point_cloud(args.output, filtered_pcd)
        else:
            # No 3DGS meta: safe to use Open3D
            o3d.io.write_point_cloud(args.output, filtered_pcd)
            print("Saved with Open3D.")
    print("Done!")

    # Visualize if requested
    if args.visualize:
        print("\nVisualizing cleaned point cloud...")
        o3d.visualization.draw_geometries([filtered_pcd])


if __name__ == "__main__":
    main()
