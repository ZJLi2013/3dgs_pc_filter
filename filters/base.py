"""
Base classes for point cloud filtering framework
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import open3d as o3d
from dataclasses import dataclass, field
import time


@dataclass
class FilterStats:
    """Statistics from a filter operation"""

    filter_name: str
    points_input: int
    points_output: int
    points_removed: int
    removal_percentage: float
    processing_time: float
    parameters: Dict[str, Any] = field(default_factory=dict)
    additional_info: Dict[str, Any] = field(default_factory=dict)

    def __str__(self):
        return (
            f"\n{self.filter_name}:\n"
            f"  Input points: {self.points_input:,}\n"
            f"  Output points: {self.points_output:,}\n"
            f"  Removed: {self.points_removed:,} ({self.removal_percentage:.2f}%)\n"
            f"  Time: {self.processing_time:.2f}s\n"
            f"  Parameters: {self.parameters}"
        )


class FilterBase(ABC):
    """Base class for all point cloud filters"""

    def __init__(self, name: str, enabled: bool = True, **kwargs):
        """
        Initialize filter

        Args:
            name: Filter name
            enabled: Whether filter is enabled
            **kwargs: Additional filter-specific parameters
        """
        self.name = name
        self.enabled = enabled
        self.config = kwargs

    @abstractmethod
    def _apply_filter(
        self, pcd: o3d.geometry.PointCloud, metadata: Optional[Dict] = None
    ) -> Tuple[o3d.geometry.PointCloud, np.ndarray]:
        """
        Apply the filter logic (to be implemented by subclasses)

        Args:
            pcd: Input point cloud
            metadata: Optional metadata (e.g., 3DGS attributes)

        Returns:
            Tuple of (filtered_pcd, removed_indices)
        """
        pass

    def filter(
        self, pcd: o3d.geometry.PointCloud, metadata: Optional[Dict] = None
    ) -> Tuple[o3d.geometry.PointCloud, FilterStats]:
        """
        Apply filter and collect statistics

        Args:
            pcd: Input point cloud
            metadata: Optional metadata

        Returns:
            Tuple of (filtered_pcd, stats)
        """
        if not self.enabled:
            # Return original point cloud if filter is disabled
            stats = FilterStats(
                filter_name=self.name,
                points_input=len(pcd.points),
                points_output=len(pcd.points),
                points_removed=0,
                removal_percentage=0.0,
                processing_time=0.0,
                parameters={"enabled": False},
            )
            return pcd, stats

        start_time = time.time()
        points_input = len(pcd.points)

        # Apply the actual filter
        filtered_pcd, removed_indices = self._apply_filter(pcd, metadata)

        points_output = len(filtered_pcd.points)
        points_removed = points_input - points_output
        removal_percentage = (
            (points_removed / points_input * 100) if points_input > 0 else 0
        )
        processing_time = time.time() - start_time

        stats = FilterStats(
            filter_name=self.name,
            points_input=points_input,
            points_output=points_output,
            points_removed=points_removed,
            removal_percentage=removal_percentage,
            processing_time=processing_time,
            parameters=self.config,
            additional_info={"removed_indices": removed_indices},
        )

        return filtered_pcd, stats

    def get_config(self) -> Dict[str, Any]:
        """Get filter configuration"""
        return {"name": self.name, "enabled": self.enabled, **self.config}


class FilterPipeline:
    """Pipeline for executing multiple filters in sequence"""

    def __init__(self, name: str = "FilterPipeline"):
        """
        Initialize pipeline

        Args:
            name: Pipeline name
        """
        self.name = name
        self.filters: List[FilterBase] = []
        self.stats_history: List[FilterStats] = []
        # Track indices of original points that survive all filters
        self.final_keep_indices: Optional[np.ndarray] = None

    def add_filter(self, filter_obj: FilterBase) -> "FilterPipeline":
        """
        Add a filter to the pipeline

        Args:
            filter_obj: Filter instance

        Returns:
            Self for method chaining
        """
        self.filters.append(filter_obj)
        return self

    def run(
        self,
        pcd: o3d.geometry.PointCloud,
        metadata: Optional[Dict] = None,
        save_intermediate: bool = False,
        intermediate_dir: Optional[str] = None,
    ) -> Tuple[o3d.geometry.PointCloud, List[FilterStats]]:
        """
        Run all filters in sequence

        Args:
            pcd: Input point cloud
            metadata: Optional metadata
            save_intermediate: Whether to save intermediate results
            intermediate_dir: Directory for intermediate files

        Returns:
            Tuple of (final_pcd, list_of_stats)
        """
        current_pcd = pcd
        all_stats = []
        # Start with all original indices
        current_indices = np.arange(len(pcd.points))

        print(f"\n{'='*60}")
        print(f"Running {self.name}")
        print(f"{'='*60}")
        print(f"Input: {len(pcd.points):,} points\n")

        for i, filter_obj in enumerate(self.filters):
            if not filter_obj.enabled:
                print(f"[{i+1}/{len(self.filters)}] {filter_obj.name}: SKIPPED")
                continue

            print(f"[{i+1}/{len(self.filters)}] Running {filter_obj.name}...")

            current_pcd, stats = filter_obj.filter(current_pcd, metadata)
            all_stats.append(stats)

            print(stats)

            # Update surviving original indices using removed indices of this step
            removed = stats.additional_info.get("removed_indices", None)
            if removed is not None and len(removed) > 0:
                # removed are indices relative to current_pcd; drop them from current_indices
                if isinstance(removed, np.ndarray):
                    removed_idx = removed
                else:
                    # Convert list to numpy array for consistent behavior
                    removed_idx = np.asarray(removed)
                # Compute mask to keep indices not removed
                keep_mask_step = np.ones(current_indices.shape[0], dtype=bool)
                keep_mask_step[removed_idx] = False
                current_indices = current_indices[keep_mask_step]

            # Save intermediate result if requested
            if save_intermediate and intermediate_dir:
                import os

                os.makedirs(intermediate_dir, exist_ok=True)
                filename = os.path.join(
                    intermediate_dir, f"step_{i+1:02d}_{filter_obj.name}.ply"
                )
                o3d.io.write_point_cloud(filename, current_pcd)
                print(f"  Saved intermediate result to: {filename}")

        self.stats_history.extend(all_stats)
        # Save final surviving original indices
        self.final_keep_indices = current_indices
        return current_pcd, all_stats

    def get_summary(self, stats_list: List[FilterStats]) -> str:
        """
        Generate summary report

        Args:
            stats_list: List of filter statistics

        Returns:
            Summary string
        """
        if not stats_list:
            return "No filters were applied."

        total_input = stats_list[0].points_input
        total_output = stats_list[-1].points_output
        total_removed = total_input - total_output
        total_percentage = (total_removed / total_input * 100) if total_input > 0 else 0
        total_time = sum(s.processing_time for s in stats_list)

        summary = f"\n{'='*60}\n"
        summary += f"Pipeline Summary: {self.name}\n"
        summary += f"{'='*60}\n"
        summary += f"Initial points: {total_input:,}\n"
        summary += f"Final points: {total_output:,}\n"
        summary += f"Total removed: {total_removed:,} ({total_percentage:.2f}%)\n"
        summary += f"Total processing time: {total_time:.2f}s\n"
        summary += f"\nFilters applied: {len([s for s in stats_list if s.points_removed > 0])}\n"

        for stats in stats_list:
            if stats.points_removed > 0:
                summary += f"  - {stats.filter_name}: {stats.points_removed:,} points ({stats.removal_percentage:.2f}%)\n"

        summary += f"{'='*60}\n"

        return summary

    def clear_history(self):
        """Clear statistics history"""
        self.stats_history.clear()

    def get_config(self) -> Dict[str, Any]:
        """Get pipeline configuration"""
        return {
            "name": self.name,
            "filters": [f.get_config() for f in self.filters],
        }
