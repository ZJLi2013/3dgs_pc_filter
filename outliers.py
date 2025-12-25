"""
Advanced Point Cloud Outlier Removal Tool  (3DGS MVP)
Semantic-aware friendly defaults: prioritize AttributeFilter + conservative SOR
"""

import argparse
import sys
from pathlib import Path
import open3d as o3d

# Add filters directory to path
sys.path.insert(0, str(Path(__file__).parent))

from filters import (
    FilterPipeline,
    StatisticalOutlierFilter,
    AttributeFilter,
)


def main():
    parser = argparse.ArgumentParser(
        description="Point Cloud Outlier Removal  (Conservative, 3DGS-friendly)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Attribute-only (no geometry filtering, adaptive opacity threshold ~1% removal)
  python outliers.py -i input.ply -o output.ply --attr

  # Attribute + conservative SOR
  python outliers.py -i input.ply -o output.ply --attr --sor --sor-neighbors 150 --sor-std 2.5
        """,
    )

    # Input/Output
    parser.add_argument("--input", "-i", required=True, help="Input point cloud file")
    parser.add_argument("--output", "-o", required=True, help="Output point cloud file")

    # Preset mode (MVP-focused)

    # Attribute Filter (3DGS semantics)
    parser.add_argument("--attr", action="store_true", help="Enable attribute filter")

    # Statistical Outlier Filter (geometry-only, conservative defaults)
    parser.add_argument("--sor", action="store_true", help="Enable SOR filter")
    parser.add_argument(
        "--sor-neighbors", type=int, default=150, help="SOR neighbors (default: 150)"
    )
    parser.add_argument(
        "--sor-std", type=float, default=2.5, help="SOR std ratio (default: 2.5)"
    )

    # Output options
    parser.add_argument(
        "--visualize", "-v", action="store_true", help="Visualize final result"
    )

    args = parser.parse_args()

    # Load point cloud
    print(f"\nLoading point cloud from: {args.input}")
    pcd = o3d.io.read_point_cloud(args.input)
    num_points = len(pcd.points)
    print(f"Loaded {num_points:,} points")

    # Load 3DGS metadata when AttributeFilter is requested
    metadata = None
    need_attr = args.attr
    if need_attr:
        print("\nAttempting to load 3DGS metadata...")
        metadata = AttributeFilter.load_3dgs_metadata(args.input)
        if metadata:
            print(f"Loaded metadata with keys: {list(metadata.keys())}")
        else:
            print(
                "Warning: Could not load 3DGS metadata, AttributeFilter will be skipped"
            )

    # Build pipeline
    pipeline = FilterPipeline(name="Conservative Pipeline")

    # Build filters based on flags (conservative by default)
    if args.attr and metadata:
        pipeline.add_filter(
            AttributeFilter(
                enabled=True,
            )
        )

    if args.sor:
        pipeline.add_filter(
            StatisticalOutlierFilter(
                nb_neighbors=args.sor_neighbors,
                std_ratio=args.sor_std,
                enabled=True,
            )
        )

    # Check if any filters are enabled
    if len(pipeline.filters) == 0:
        print("\nError: No filters enabled!")
        print(
            "Enable specific filters (--attr, --sor) to apply changes; by default no geometry filtering is performed."
        )
        sys.exit(1)

    # Run pipeline
    filtered_pcd, stats_list = pipeline.run(
        pcd,
        metadata=metadata,
    )

    # Print summary
    print(pipeline.get_summary(stats_list))

    # Save result (preserve 3DGS vertex attributes if metadata exists)
    print(f"\nSaving cleaned point cloud to: {args.output}")
    if metadata is not None:
        try:
            from ply_utils import write_filtered_ply

            if (
                hasattr(pipeline, "final_keep_indices")
                and pipeline.final_keep_indices is not None
            ):
                write_filtered_ply(args.input, args.output, pipeline.final_keep_indices)
                print("Saved with plyfile (preserved 3DGS attributes).")
            else:
                # Fallback if indices not available
                o3d.io.write_point_cloud(args.output, filtered_pcd)
                print(
                    "Warning: final indices not available; saved with Open3D (attributes may be lost)."
                )
        except Exception as e:
            print(
                f"Warning: plyfile save failed ({e}); falling back to Open3D (attributes may be lost)."
            )
            o3d.io.write_point_cloud(args.output, filtered_pcd)
    else:
        # No metadata: safe to use Open3D
        o3d.io.write_point_cloud(args.output, filtered_pcd)
        print("Saved with Open3D.")
    print("Done!")

    # Visualize if requested
    if args.visualize:
        print("\nVisualizing cleaned point cloud...")
        o3d.visualization.draw_geometries([filtered_pcd])


if __name__ == "__main__":
    main()
