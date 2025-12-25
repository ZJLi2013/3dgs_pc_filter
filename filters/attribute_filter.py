"""
Attribute Filter for 3DGS Point Clouds
"""

from typing import Dict, Optional, Tuple
import numpy as np
import open3d as o3d
from .base import FilterBase


class AttributeFilter(FilterBase):
    """
    Attribute-based filter for 3DGS point clouds

    Filters points based on 3DGS-specific attributes like opacity, scale, and SH energy.
    This filter requires metadata containing these attributes.
    """

    def __init__(
        self,
        opacity_threshold: Optional[float] = None,
        scale_min_percentile: Optional[float] = None,
        scale_max_percentile: Optional[float] = None,
        sh_energy_threshold: Optional[float] = None,
        enabled: bool = True,
    ):
        """
        Initialize Attribute Filter

        Args:
            opacity_threshold: Minimum opacity value (0.01-0.05)
            scale_min_percentile: Remove points below this scale percentile
            scale_max_percentile: Remove points above this scale percentile
            sh_energy_threshold: Minimum SH energy (optional)
            enabled: Whether filter is enabled
        """
        super().__init__(
            name="AttributeFilter",
            enabled=enabled,
            opacity_threshold=opacity_threshold,
            scale_min_percentile=scale_min_percentile,
            scale_max_percentile=scale_max_percentile,
            sh_energy_threshold=sh_energy_threshold,
        )
        self.opacity_threshold = opacity_threshold
        self.scale_min_percentile = scale_min_percentile
        self.scale_max_percentile = scale_max_percentile
        self.sh_energy_threshold = sh_energy_threshold

    def _apply_filter(
        self, pcd: o3d.geometry.PointCloud, metadata: Optional[Dict] = None
    ) -> Tuple[o3d.geometry.PointCloud, np.ndarray]:
        """
        Apply attribute-based filtering

        Args:
            pcd: Input point cloud
            metadata: Dictionary containing 3DGS attributes
                     Expected keys: 'opacity', 'scale', 'sh_coeffs' (optional)

        Returns:
            Tuple of (filtered_pcd, removed_indices)
        """
        if metadata is None:
            print("  Warning: No metadata provided, skipping attribute filter")
            return pcd, np.array([])

        num_points = len(pcd.points)
        keep_mask = np.ones(num_points, dtype=bool)

        # Filter by opacity (adaptive if threshold is None)
        if "opacity" in metadata:
            opacity = np.asarray(metadata["opacity"], dtype=float)
            # Clip to [0,1] and remove invalids from threshold computation
            opacity = np.clip(opacity, 0.0, 1.0)
            valid_mask = np.isfinite(opacity)
            valid_opacity = opacity[valid_mask]

            # Print distribution stats (for debugging/inspection)
            nz = np.count_nonzero(valid_opacity)
            nonzero_ratio = nz / valid_opacity.size if valid_opacity.size > 0 else 0.0

            def _pct(arr, p):
                return float(np.percentile(arr, p)) if arr.size > 0 else 0.0

            print(
                f"  Opacity stats: "
                f"min={float(valid_opacity.min()) if valid_opacity.size>0 else 0.0:.6f}, "
                f"median={float(np.median(valid_opacity)) if valid_opacity.size>0 else 0.0:.6f}, "
                f"p95={_pct(valid_opacity,95):.6f}, p99={_pct(valid_opacity,99):.6f}, "
                f"mean={float(valid_opacity.mean()) if valid_opacity.size>0 else 0.0:.6f}, "
                f"nonzero_ratio={nonzero_ratio:.4f}"
            )

            threshold = self.opacity_threshold
            if threshold is None:
                # MVP: explicit-zero removal only (no adaptive/percentile)
                threshold = 1e-8
                print(f"  Opacity zero-removal epsilon: {threshold:.1e}")

            # Keep points with opacity strictly greater than epsilon
            opacity_mask = opacity > threshold
            keep_mask &= opacity_mask
            removed_by_opacity = int((~opacity_mask).sum())
            print(f"  Opacity filter: removed {removed_by_opacity:,} points")

        # Filter by scale (only if percentiles explicitly provided)
        if (
            "scale" in metadata
            and self.scale_min_percentile is not None
            and self.scale_max_percentile is not None
        ):
            scale = metadata["scale"]

            # Handle scale as 3D vector or scalar
            if scale.ndim > 1:
                # Use mean scale if 3D
                scale_values = np.mean(scale, axis=1)
            else:
                scale_values = scale

            # Calculate percentile thresholds
            scale_min = np.percentile(scale_values, self.scale_min_percentile)
            scale_max = np.percentile(scale_values, self.scale_max_percentile)

            scale_mask = (scale_values >= scale_min) & (scale_values <= scale_max)
            keep_mask &= scale_mask
            removed_by_scale = np.sum(~scale_mask)
            print(
                f"  Scale filter: removed {removed_by_scale:,} points "
                f"(range: {scale_min:.6f} - {scale_max:.6f})"
            )

        # Filter by SH energy (optional)
        if self.sh_energy_threshold is not None and "sh_coeffs" in metadata:
            sh_coeffs = metadata["sh_coeffs"]
            # Calculate energy as sum of squared coefficients
            sh_energy = np.sum(sh_coeffs**2, axis=1)
            sh_mask = sh_energy >= self.sh_energy_threshold
            keep_mask &= sh_mask
            removed_by_sh = np.sum(~sh_mask)
            print(f"  SH energy filter: removed {removed_by_sh:,} points")

        # Get indices to keep and remove
        keep_indices = np.where(keep_mask)[0]
        removed_indices = np.where(~keep_mask)[0]

        # Filter point cloud
        filtered_pcd = pcd.select_by_index(keep_indices.tolist())

        return filtered_pcd, removed_indices

    @staticmethod
    def load_3dgs_metadata(ply_file: str) -> Optional[Dict]:
        """
        Load 3DGS metadata from PLY file

        Args:
            ply_file: Path to PLY file

        Returns:
            Dictionary with metadata or None if not available
        """
        try:
            from plyfile import PlyData

            plydata = PlyData.read(ply_file)
            vertex = plydata["vertex"]

            metadata = {}

            # Try to extract opacity
            if "opacity" in vertex:
                metadata["opacity"] = np.array(vertex["opacity"])
            elif "alpha" in vertex:
                metadata["opacity"] = np.array(vertex["alpha"])

            # Try to extract scale
            scale_names = ["scale_0", "scale_1", "scale_2"]
            if all(name in vertex for name in scale_names):
                metadata["scale"] = np.column_stack(
                    [vertex[name] for name in scale_names]
                )
            elif "scale" in vertex:
                metadata["scale"] = np.array(vertex["scale"])

            # Try to extract SH coefficients
            sh_names = [
                name
                for name in vertex.data.dtype.names
                if name.startswith("f_dc") or name.startswith("f_rest")
            ]
            if sh_names:
                metadata["sh_coeffs"] = np.column_stack(
                    [vertex[name] for name in sh_names]
                )

            return metadata if metadata else None

        except ImportError:
            print("Warning: plyfile package not installed. Cannot load 3DGS metadata.")
            print("Install with: pip install plyfile")
            return None
        except Exception as e:
            print(f"Warning: Could not load 3DGS metadata: {e}")
            return None
