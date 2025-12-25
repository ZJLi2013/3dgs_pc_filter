"""
Statistical Outlier Removal Filter
"""

from typing import Dict, Optional, Tuple
import numpy as np
import open3d as o3d
from .base import FilterBase


class StatisticalOutlierFilter(FilterBase):
    """
    Statistical Outlier Removal (SOR) filter

    Removes points that are farther away from their neighbors compared to
    the average for the point cloud. Uses k-nearest neighbors and standard
    deviation threshold.
    """

    def __init__(
        self,
        nb_neighbors: int = 50,
        std_ratio: float = 2.0,
        enabled: bool = True,
    ):
        """
        Initialize Statistical Outlier Filter

        Args:
            nb_neighbors: Number of neighbors to analyze for each point
            std_ratio: Standard deviation ratio threshold
            enabled: Whether filter is enabled
        """
        super().__init__(
            name="StatisticalOutlierFilter",
            enabled=enabled,
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio,
        )
        self.nb_neighbors = nb_neighbors
        self.std_ratio = std_ratio

    def _apply_filter(
        self, pcd: o3d.geometry.PointCloud, metadata: Optional[Dict] = None
    ) -> Tuple[o3d.geometry.PointCloud, np.ndarray]:
        """
        Apply statistical outlier removal

        Args:
            pcd: Input point cloud
            metadata: Optional metadata (not used in this filter)

        Returns:
            Tuple of (filtered_pcd, removed_indices)
        """
        # Apply Open3D's statistical outlier removal
        cl, ind = pcd.remove_statistical_outlier(
            nb_neighbors=self.nb_neighbors, std_ratio=self.std_ratio
        )

        # Get removed indices
        all_indices = np.arange(len(pcd.points))
        removed_indices = np.setdiff1d(all_indices, ind)

        # Select inlier points
        filtered_pcd = pcd.select_by_index(ind)

        return filtered_pcd, removed_indices

    @staticmethod
    def recommend_parameters(num_points: int, point_density: str = "medium") -> Dict:
        """
        Recommend parameters based on point cloud characteristics

        Args:
            num_points: Number of points in the cloud
            point_density: Density level ('low', 'medium', 'high')

        Returns:
            Dictionary with recommended parameters
        """
        # Base recommendations
        if point_density == "low":
            nb_neighbors_base = 20
            std_ratio = 2.5
        elif point_density == "high":
            nb_neighbors_base = 150
            std_ratio = 2.0
        else:  # medium
            nb_neighbors_base = 50
            std_ratio = 2.0

        # Adjust based on point count
        if num_points < 100000:
            nb_neighbors = nb_neighbors_base
        elif num_points < 1000000:
            nb_neighbors = int(nb_neighbors_base * 1.5)
        elif num_points < 10000000:
            nb_neighbors = int(nb_neighbors_base * 2)
        else:  # > 10M points (3DGS scale)
            nb_neighbors = int(nb_neighbors_base * 3)

        return {
            "nb_neighbors": nb_neighbors,
            "std_ratio": std_ratio,
            "reasoning": f"Based on {num_points:,} points with {point_density} density",
        }
