"""
Contribution-based Filter for 3DGS Point Clouds (MVP)

Deletes Gaussians whose long-term rendering contribution is ~0 across representative views:
Ci = sum_v (alpha_i * A_i,v), where A_i,v is the projected area approximation.

Assumptions:
- alpha: sigmoid-activated opacity in [0,1] (provided via metadata["alpha"])
- scale: per-point scale as (scale_x, scale_y, scale_z) or scalar (metadata["scale"])
- cameras: list of dicts with keys:
  {fx, fy, cx, cy, width, height, extrinsics}, where extrinsics is 4x4 world->camera
- If scale is missing, we fall back to 1.0 for both axes.
- If extrinsics are camera->world, set a flag in your JSON or convert before passing in.

Notes:
- This is an MVP: uses a simple perspective-area approximation
  A_i,v ~= (scale_x * scale_y) * (fx * fy) / Z_i,v^2, for Z_i,v > 0, else 0
- Vectorized with chunking to handle large point clouds.
"""

from typing import Dict, Optional, Tuple, List
import numpy as np
import open3d as o3d
from .base import FilterBase


class ContribFilter(FilterBase):
    def __init__(
        self,
        epsilon: float = 1e-10,  # near-zero contribution threshold
        chunk_size: int = 500_000,  # process points in chunks to avoid memory blow-ups
        enabled: bool = True,
    ):
        super().__init__(
            name="ContribFilter",
            enabled=enabled,
            epsilon=epsilon,
            chunk_size=chunk_size,
        )
        self.epsilon = epsilon
        self.chunk_size = chunk_size

    def _apply_filter(
        self, pcd: o3d.geometry.PointCloud, metadata: Optional[Dict] = None
    ) -> Tuple[o3d.geometry.PointCloud, np.ndarray]:
        if metadata is None:
            print("  Warning: No metadata provided; skipping ContribFilter")
            return pcd, np.array([], dtype=np.int64)

        # Required fields: alpha, cameras. Optional: scale
        alpha = metadata.get("alpha", None)
        cameras: Optional[List[Dict]] = metadata.get("cameras", None)

        if alpha is None or cameras is None or len(cameras) == 0:
            print("  Warning: Missing alpha or cameras; skipping ContribFilter")
            return pcd, np.array([], dtype=np.int64)

        points = np.asarray(pcd.points, dtype=np.float64)
        num_points = points.shape[0]

        # Align alpha/scale lengths with current point cloud if possible
        # MVP assumption: metadata arrays are aligned to current pcd (this filter should run first in pipeline).
        if alpha.shape[0] != num_points:
            print(
                f"  Warning: alpha length ({alpha.shape[0]}) != points ({num_points}); skipping ContribFilter"
            )
            return pcd, np.array([], dtype=np.int64)

        scale = metadata.get("scale", None)
        if scale is None:
            scale_x = np.ones(num_points, dtype=np.float64)
            scale_y = np.ones(num_points, dtype=np.float64)
        else:
            scale = np.asarray(scale, dtype=np.float64)
            if scale.ndim == 1:
                scale_x = scale
                scale_y = scale
            else:
                # assume (..., 3) or (..., >=2)
                if scale.shape[1] >= 2:
                    scale_x = scale[:, 0]
                    scale_y = scale[:, 1]
                else:
                    scale_x = scale[:, 0]
                    scale_y = scale[:, 0]

        # Accumulate contribution Ci
        Ci = np.zeros(num_points, dtype=np.float64)

        # Precompute per-point factor (scale_x * scale_y)
        area_factor = scale_x * scale_y

        # Process cameras one by one to keep memory usage bounded
        for v_idx, cam in enumerate(cameras):
            fx = float(cam.get("fx", 1.0))
            fy = float(cam.get("fy", 1.0))
            extr = np.asarray(cam.get("extrinsics", np.eye(4)), dtype=np.float64)
            if extr.shape != (4, 4):
                print(
                    f"  Warning: camera {v_idx} extrinsics shape invalid {extr.shape}; skipping this camera"
                )
                continue

            R = extr[0:3, 0:3]
            t = extr[0:3, 3]

            # Chunked processing
            for start in range(0, num_points, self.chunk_size):
                end = min(start + self.chunk_size, num_points)
                P = points[start:end]  # (M, 3)

                # world -> camera: X_cam = R * X + t
                X_cam = (P @ R.T) + t  # (M, 3)
                Z = X_cam[:, 2]  # depth

                # valid only for points with positive depth
                valid = Z > 0
                if not np.any(valid):
                    continue

                # A_i,v ~= (scale_x * scale_y) * (fx * fy) / Z^2
                Z2 = Z[valid] * Z[valid]
                A = area_factor[start:end][valid] * (fx * fy) / (Z2 + 1e-12)

                Ci[start:end][valid] += alpha[start:end][valid] * A

        # Delete points with near-zero contribution
        contrib_mask = Ci > self.epsilon
        removed_indices = np.where(~contrib_mask)[0]

        # Logging
        nz = np.count_nonzero(Ci)
        print(
            f"  Contrib stats: min={float(Ci.min() if Ci.size>0 else 0.0):.6e}, "
            f"median={float(np.median(Ci) if Ci.size>0 else 0.0):.6e}, "
            f"p95={float(np.percentile(Ci,95)):.6e} if Ci.size>0 else 0.0, "
            f"mean={float(Ci.mean() if Ci.size>0 else 0.0):.6e}, "
            f"nonzero_ratio={(nz / Ci.size) if Ci.size>0 else 0.0:.4f}"
        )
        print(f"  Contrib epsilon: {self.epsilon:.1e}")
        print(f"  Contrib filter: removed {removed_indices.size:,} points")

        filtered_pcd = pcd.select_by_index(np.where(contrib_mask)[0].tolist())
        return filtered_pcd, removed_indices
