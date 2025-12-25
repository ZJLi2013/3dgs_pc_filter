"""
Example usage of the modular point cloud filtering system (MVP-focused)
"""

import open3d as o3d
from filters import (
    FilterPipeline,
    StatisticalOutlierFilter,
    AttributeFilter,
)


def example_mvp_pipeline():
    """Example: 3DGS MVP - Attribute semantics + conservative SOR"""
    print("\n" + "=" * 60)
    print("Example: 3DGS MVP Pipeline (Attribute + Conservative SOR)")
    print("=" * 60)

    # Load point cloud
    pcd = o3d.io.read_point_cloud("input.ply")

    # Try load 3DGS metadata
    metadata = AttributeFilter.load_3dgs_metadata("input.ply")

    # Create pipeline
    pipeline = FilterPipeline(name="MVP Cleaning")

    # Attribute filter first (if metadata available)
    if metadata:
        pipeline.add_filter(
            AttributeFilter(
                opacity_threshold=0.02,
                scale_min_percentile=1.0,
                scale_max_percentile=99.0,
                enabled=True,
            )
        )
    else:
        print("Warning: 3DGS metadata not found. Skipping AttributeFilter.")

    # Conservative SOR to avoid over-removal of semantic structures
    pipeline.add_filter(
        StatisticalOutlierFilter(nb_neighbors=150, std_ratio=2.5, enabled=True)
    )

    # Run
    filtered_pcd, stats = pipeline.run(pcd, metadata=metadata)

    # Summary
    print(pipeline.get_summary(stats))

    # Save result
    o3d.io.write_point_cloud("output_mvp.ply", filtered_pcd)


def example_balanced_pipeline():
    """Example: Balanced - geometry-only SOR with moderate strictness"""
    print("\n" + "=" * 60)
    print("Example: Balanced Pipeline (SOR only)")
    print("=" * 60)

    # Load point cloud
    pcd = o3d.io.read_point_cloud("input.ply")

    # Create pipeline
    pipeline = FilterPipeline(name="Balanced Cleaning")

    # Geometry-only SOR
    pipeline.add_filter(
        StatisticalOutlierFilter(nb_neighbors=150, std_ratio=1.5, enabled=True)
    )

    # Run
    filtered_pcd, stats = pipeline.run(pcd)

    # Summary
    print(pipeline.get_summary(stats))

    # Save result
    o3d.io.write_point_cloud("output_balanced.ply", filtered_pcd)


if __name__ == "__main__":
    print("Point Cloud Filtering Examples (MVP-focused)")
    print("=" * 60)
    print("\nAvailable examples:")
    print("1. 3DGS MVP Pipeline (Attribute + Conservative SOR)")
    print("2. Balanced Pipeline (SOR only)")
    print("\nUncomment the example you want to run in the code.")

    # Uncomment the example you want to run:
    # example_mvp_pipeline()
    # example_balanced_pipeline()
