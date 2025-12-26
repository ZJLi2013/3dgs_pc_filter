"""
Depth Consistency Filter for 3DGS Point Clouds (MVP)

Goal:
- Remove points that are geometrically inconsistent across multiple views, using per-view
  relative depth maps and calibrated cameras.

Inputs via metadata:
- cameras: List[Dict] per view with keys:
    - K: 3x3 intrinsic matrix (fx, 0, cx; 0, fy, cy; 0, 0, 1)
    - T_cw (preferred) OR T_wc: 4x4 extrinsic (world->camera or camera->world)
    - width: image width (int)
    - height: image height (int)
- depths: List[np.ndarray], each (H, W) float relative depth map for the corresponding view
- Optional:
    - scale: (N,) or (N,3) per-point 3DGS scale for adaptive thresholds

Behavior:
- Select up to K views globally (strided) to evaluate (for efficiency).
- For each selected view, project points to the view, gather pairs (z_cam, d_img),
  fit a simple linear model d ≈ a * z + b (robust-ish via sample filtering).
- Count a point as an inlier in a view if residual |(a*z + b) - d| <= (sigma_b*std_d + epsilon).
- Aggregate across views; if a point has at least min_visible_views and inlier_ratio < min_inlier_ratio,
  mark it for removal. Otherwise keep (conservative default for low-visibility points).

Notes:
- This MVP assumes the per-view relative depth is monotonic with camera-space z (up to affine transform).
- The linear mapping (a,b) is estimated per-view from a random subset of projected points.
"""

from typing import Dict, Optional, Tuple, List, Any
import numpy as np
import open3d as o3d

from .base import FilterBase


def _ensure_T_cw(cam: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Return (R, t) for world->camera. Accepts T_cw/T_wc or dataset-style keys like 'extrinsics_3x4_w2c'."""
    T = None
    if "T_cw" in cam:
        T = np.asarray(cam["T_cw"], dtype=np.float64)
    elif "T_wc" in cam:
        T_wc = np.asarray(cam["T_wc"], dtype=np.float64)
        # Invert to get T_cw
        R_wc = T_wc[:3, :3]
        t_wc = T_wc[:3, 3]
        R_cw = R_wc.T
        t_cw = -R_wc.T @ t_wc
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R_cw
        T[:3, 3] = t_cw
    elif "extrinsics_4x4_w2c" in cam:
        T = np.asarray(cam["extrinsics_4x4_w2c"], dtype=np.float64)
    elif "extrinsics_3x4_w2c" in cam:
        E = np.asarray(cam["extrinsics_3x4_w2c"], dtype=np.float64)
        if E.shape != (3, 4):
            raise ValueError("extrinsics_3x4_w2c must be 3x4")
        T = np.eye(4, dtype=np.float64)
        T[:3, :4] = E
    elif "extrinsics_4x4_c2w" in cam:
        T_wc = np.asarray(cam["extrinsics_4x4_c2w"], dtype=np.float64)
        R_wc = T_wc[:3, :3]
        t_wc = T_wc[:3, 3]
        R_cw = R_wc.T
        t_cw = -R_wc.T @ t_wc
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R_cw
        T[:3, 3] = t_cw
    elif "extrinsics_3x4_c2w" in cam:
        E = np.asarray(cam["extrinsics_3x4_c2w"], dtype=np.float64)
        if E.shape != (3, 4):
            raise ValueError("extrinsics_3x4_c2w must be 3x4")
        T_wc = np.eye(4, dtype=np.float64)
        T_wc[:3, :4] = E
        R_wc = T_wc[:3, :3]
        t_wc = T_wc[:3, 3]
        R_cw = R_wc.T
        t_cw = -R_wc.T @ t_wc
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R_cw
        T[:3, 3] = t_cw
    else:
        raise ValueError("Camera dict must contain T_cw/T_wc or extrinsics_* keys")
    R = T[:3, :3].astype(np.float64, copy=False)
    t = T[:3, 3].astype(np.float64, copy=False)
    return R, t


def _ensure_K(cam: Dict[str, Any]) -> np.ndarray:
    """Return 3x3 intrinsic matrix K; accept 'K', 'intrinsics_3x3' or 'intrinsics' with fx,fy,cx,cy."""
    if "K" in cam:
        K = np.asarray(cam["K"], dtype=np.float64)
        if K.shape != (3, 3):
            raise ValueError("Camera K must be 3x3")
        return K
    if "intrinsics_3x3" in cam:
        K = np.asarray(cam["intrinsics_3x3"], dtype=np.float64)
        if K.shape != (3, 3):
            raise ValueError("intrinsics_3x3 must be 3x3")
        return K
    intr = cam.get("intrinsics", None)
    if intr is None:
        raise ValueError("Camera dict must contain K, intrinsics_3x3 or intrinsics")
    fx = float(intr.get("fx"))
    fy = float(intr.get("fy"))
    cx = float(intr.get("cx", 0.0))
    cy = float(intr.get("cy", 0.0))
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    return K


def _project_points(
    Pw: np.ndarray, R: np.ndarray, t: np.ndarray, K: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project 3D points (world) to image pixels.

    Returns:
        u (N,), v (N,), zc (N,)
    """
    # Camera coords: Pc = R @ Pw.T + t[:,None]
    Pc = (Pw @ R.T) + t[None, :]
    zc = Pc[:, 2]
    # Avoid division by zero
    eps = 1e-12
    inv_z = 1.0 / np.maximum(zc, eps)
    x_n = Pc[:, 0] * inv_z
    y_n = Pc[:, 1] * inv_z
    u = K[0, 0] * x_n + K[0, 2]
    v = K[1, 1] * y_n + K[1, 2]
    return u, v, zc


def _bilinear_sample(depth: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Bilinear sample per-pixel depth at floating uv. Outside will be filled with NaN.

    Args:
        depth: (H, W)
        u, v: (N,) pixel coordinates

    Returns:
        sampled depth (N,)
    """
    H, W = depth.shape
    u0 = np.floor(u).astype(np.int64)
    v0 = np.floor(v).astype(np.int64)
    u1 = u0 + 1
    v1 = v0 + 1

    # We will mark out-of-bounds later
    du = u - u0
    dv = v - v0

    # Clip indexes for safe gathering (weights will zero-out OOB)
    u0c = np.clip(u0, 0, W - 1)
    u1c = np.clip(u1, 0, W - 1)
    v0c = np.clip(v0, 0, H - 1)
    v1c = np.clip(v1, 0, H - 1)

    d00 = depth[v0c, u0c]
    d10 = depth[v0c, u1c]
    d01 = depth[v1c, u0c]
    d11 = depth[v1c, u1c]

    w00 = (1.0 - du) * (1.0 - dv)
    w10 = du * (1.0 - dv)
    w01 = (1.0 - du) * dv
    w11 = du * dv

    d = w00 * d00 + w10 * d10 + w01 * d01 + w11 * d11

    # Mark OOB pixels as NaN
    in_bounds = (u0 >= 0) & (v0 >= 0) & (u1 < W) & (v1 < H)
    d[~in_bounds] = np.nan
    return d


class DepthConsistencyFilter(FilterBase):
    """
    Multi-view depth consistency filter.

    Config parameters:
        max_views: int = 12
        min_visible_views: int = 4
        min_inlier_ratio: float = 0.6
        batch_size: int = 200000
        fit_subset_points: int = 50000
        fit_min_samples: int = 1000
        sigma_b: float = 2.0       # depth std factor
        remove_if_few_views: bool = False
        sampling: str = "strided"  # "strided" | "all" (MVP: global view subset)
        use_bilinear: bool = True
    """

    def __init__(
        self,
        max_views: int = 12,
        min_visible_views: int = 4,
        min_inlier_ratio: float = 0.6,
        batch_size: int = 200000,
        fit_subset_points: int = 50000,
        fit_min_samples: int = 1000,
        sigma_b: float = 2.0,
        remove_if_few_views: bool = False,
        sampling: str = "strided",
        use_bilinear: bool = True,
        enabled: bool = True,
    ):
        super().__init__(
            name="DepthConsistencyFilter",
            enabled=enabled,
            max_views=max_views,
            min_visible_views=min_visible_views,
            min_inlier_ratio=min_inlier_ratio,
            batch_size=batch_size,
            fit_subset_points=fit_subset_points,
            fit_min_samples=fit_min_samples,
            sigma_b=sigma_b,
            remove_if_few_views=remove_if_few_views,
            sampling=sampling,
            use_bilinear=use_bilinear,
        )
        self.max_views = int(max_views)
        self.min_visible_views = int(min_visible_views)
        self.min_inlier_ratio = float(min_inlier_ratio)
        self.batch_size = int(batch_size)
        self.fit_subset_points = int(fit_subset_points)
        self.fit_min_samples = int(fit_min_samples)
        self.sigma_b = float(sigma_b)
        self.remove_if_few_views = bool(remove_if_few_views)
        self.sampling = sampling
        self.use_bilinear = bool(use_bilinear)

    def _prepare_cameras(self, cameras: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize camera dicts: compute K, (R,t), width, height."""
        cams = []
        for i, cam in enumerate(cameras):
            try:
                K = _ensure_K(cam)
                R, t = _ensure_T_cw(cam)
                W = cam.get("width") or cam.get("W") or cam.get("w")
                H = cam.get("height") or cam.get("H") or cam.get("h")
                W = int(W) if W is not None else None
                H = int(H) if H is not None else None
            except Exception as e:
                raise ValueError(f"Camera #{i} invalid: {e}")
            cams.append({"K": K, "R": R, "t": t, "W": W, "H": H})
        return cams

    def _fit_view_mapping(
        self,
        Pw: np.ndarray,
        depth: np.ndarray,
        K: np.ndarray,
        R: np.ndarray,
        t: np.ndarray,
        max_samples: int,
        fit_min_samples: int,
        use_bilinear: bool,
        rng: np.random.Generator,
    ) -> Tuple[float, float, float]:
        """
        Fit per-view mapping d ≈ a*z + b using a random subset of points.

        Returns:
            (a, b, depth_std)
        """
        N = Pw.shape[0]
        if N <= 0:
            return 1.0, 0.0, 1.0

        # Sample indices
        ns = min(N, max_samples)
        idx = rng.choice(N, size=ns, replace=False) if ns < N else np.arange(N)

        Pw_s = Pw[idx]
        u, v, zc = _project_points(Pw_s, R, t, K)
        # Only points with z>0 considered visible
        vis = zc > 0.0

        if use_bilinear:
            d_s = _bilinear_sample(depth, u, v)
        else:
            # Nearest neighbor
            ui = np.rint(u).astype(np.int64)
            vi = np.rint(v).astype(np.int64)
            H, W = depth.shape
            inb = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
            d_s = np.full(ui.shape, np.nan, dtype=np.float64)
            d_s[inb] = depth[vi[inb], ui[inb]]

        # Valid samples: visible, finite/positive depth
        valid = vis & np.isfinite(d_s) & (d_s > 0.0)
        z_fit = zc[valid]
        d_fit = d_s[valid]

        if z_fit.size < fit_min_samples:
            # Fallback: no mapping; use identity-like params and depth std from all valid pixels
            d_valid = depth[np.isfinite(depth) & (depth > 0.0)]
            depth_std = float(np.std(d_valid)) if d_valid.size > 0 else 1.0
            return 1.0, 0.0, max(depth_std, 1e-6)

        # Robust pre-filter: clamp to middle percentiles to avoid extreme outliers
        p_lo, p_hi = 2.0, 98.0
        z_lo, z_hi = np.percentile(z_fit, [p_lo, p_hi])
        d_lo, d_hi = np.percentile(d_fit, [p_lo, p_hi])
        mask_mid = (z_fit >= z_lo) & (z_fit <= z_hi) & (d_fit >= d_lo) & (d_fit <= d_hi)
        z_mid = z_fit[mask_mid]
        d_mid = d_fit[mask_mid]

        if z_mid.size < fit_min_samples:
            z_mid = z_fit
            d_mid = d_fit

        # Linear regression: least squares fit for d = a*z + b
        try:
            a, b = np.polyfit(z_mid, d_mid, 1)
        except Exception:
            a, b = 1.0, 0.0

        depth_std = float(np.std(d_mid)) if d_mid.size > 0 else 1.0
        depth_std = max(depth_std, 1e-6)

        return float(a), float(b), depth_std

    def _apply_filter(
        self, pcd: o3d.geometry.PointCloud, metadata: Optional[Dict] = None
    ) -> Tuple[o3d.geometry.PointCloud, np.ndarray]:
        if metadata is None:
            print("  Warning: No metadata provided, skipping DepthConsistencyFilter")
            return pcd, np.array([], dtype=np.int64)

        cameras = metadata.get("cameras", None)
        depths = metadata.get("depths", None)
        if cameras is None or depths is None:
            print(
                "  Warning: cameras/depths missing in metadata, skipping DepthConsistencyFilter"
            )
            return pcd, np.array([], dtype=np.int64)

        if not isinstance(cameras, (list, tuple)) or not isinstance(
            depths, (list, tuple)
        ):
            print("  Warning: cameras/depths must be lists (one per view), skipping")
            return pcd, np.array([], dtype=np.int64)

        if len(cameras) == 0 or len(depths) == 0:
            print("  Warning: empty cameras/depths, skipping DepthConsistencyFilter")
            return pcd, np.array([], dtype=np.int64)

        if len(cameras) != len(depths):
            print("  Warning: number of cameras and depth maps mismatch, skipping")
            return pcd, np.array([], dtype=np.int64)

        # Normalize camera dicts
        try:
            cams = self._prepare_cameras(cameras)
        except Exception as e:
            print(f"  Warning: invalid cameras metadata: {e}")
            return pcd, np.array([], dtype=np.int64)

        V = len(cams)
        P = np.asarray(pcd.points, dtype=np.float64)
        N = P.shape[0]
        if N == 0:
            return pcd, np.array([], dtype=np.int64)

        # Select subset of views (MVP: strided/global)
        if self.sampling == "all" or self.max_views >= V:
            view_indices = list(range(V))
        else:
            step = max(int(np.floor(V / self.max_views)), 1)
            view_indices = list(range(0, V, step))[: self.max_views]

        print(f"  Using {len(view_indices)}/{V} views for depth consistency checking")

        # Fit per-view mapping parameters
        rng = np.random.default_rng(12345)
        view_params: List[Tuple[float, float, float]] = []
        for vi in view_indices:
            depth_i = np.asarray(depths[vi], dtype=np.float64)
            if depth_i.ndim != 2:
                print(f"    View {vi}: depth map not 2D, skipping")
                view_params.append((1.0, 0.0, 1.0))
                continue
            K = cams[vi]["K"]
            R = cams[vi]["R"]
            t = cams[vi]["t"]
            a, b, dstd = self._fit_view_mapping(
                Pw=P,
                depth=depth_i,
                K=K,
                R=R,
                t=t,
                max_samples=self.fit_subset_points,
                fit_min_samples=self.fit_min_samples,
                use_bilinear=self.use_bilinear,
                rng=rng,
            )
            print(f"    View {vi}: fitted d≈{a:.4f}*z+{b:.4f}, depth_std={dstd:.6f}")
            view_params.append((a, b, dstd))

        # Prepare counters
        valid_views = np.zeros(N, dtype=np.int32)
        inliers = np.zeros(N, dtype=np.int32)

        # Process points in batches per view
        bs = self.batch_size
        for v_local, vi in enumerate(view_indices):
            depth_i = np.asarray(depths[vi], dtype=np.float64)
            K = cams[vi]["K"]
            R = cams[vi]["R"]
            t = cams[vi]["t"]
            a, b, dstd = view_params[v_local]

            for start in range(0, N, bs):
                end = min(start + bs, N)
                Pw_b = P[start:end]

                u, v, zc = _project_points(Pw_b, R, t, K)
                # visible and in front of camera
                vis = zc > 0.0

                if self.use_bilinear:
                    d_s = _bilinear_sample(depth_i, u, v)
                else:
                    ui = np.rint(u).astype(np.int64)
                    vi_pix = np.rint(v).astype(np.int64)
                    H, W = depth_i.shape
                    inb = (ui >= 0) & (ui < W) & (vi_pix >= 0) & (vi_pix < H)
                    d_s = np.full(ui.shape, np.nan, dtype=np.float64)
                    d_s[inb] = depth_i[vi_pix[inb], ui[inb]]

                valid = vis & np.isfinite(d_s) & (d_s > 0.0)

                if not np.any(valid):
                    continue

                # Compute residual
                z_valid = zc[valid]
                d_valid = d_s[valid]
                pred = a * z_valid + b
                resid = np.abs(pred - d_valid)

                # Threshold per-point (MVP: simple form)
                epsilon = 1e-6
                T = self.sigma_b * dstd + epsilon

                inl = resid <= T
                # Update counters
                idx_local = np.nonzero(valid)[0]
                idx_global = start + idx_local
                valid_views[idx_global] += 1
                inliers[idx_global] += inl.astype(np.int32)

        # Decide removal
        keep_mask = np.ones(N, dtype=bool)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.true_divide(inliers, np.maximum(valid_views, 1))
            ratio[valid_views == 0] = 0.0

        # Condition to remove: enough views and low inlier ratio
        remove_mask = (valid_views >= self.min_visible_views) & (
            ratio < self.min_inlier_ratio
        )

        if self.remove_if_few_views:
            remove_mask |= valid_views < self.min_visible_views

        keep_mask &= ~remove_mask

        removed_indices = np.where(~keep_mask)[0]
        keep_indices = np.where(keep_mask)[0]

        print(
            f"  DepthConsistencyFilter: valid_views (median)={np.median(valid_views):.2f}, "
            f"inlier_ratio (median over all points)={np.median(ratio):.3f}"
        )
        print(f"  Removed {removed_indices.size:,} points by depth consistency")

        filtered_pcd = pcd.select_by_index(keep_indices.tolist())
        return filtered_pcd, removed_indices


# Convenience loaders (optional, can be used by callers)
def load_cameras_from_dir(dir_path: str) -> List[Dict[str, Any]]:
    """
    Load per-view camera JSON files from a directory.
    Each JSON file should contain either:
      - 'K' (3x3) and 'T_cw' or 'T_wc' (4x4), 'width', 'height'
      - or 'intrinsics' with fx,fy,cx,cy and extrinsics + size.

    Returns list of camera dicts sorted by filename.
    """
    import os, json, glob

    files = sorted(glob.glob(os.path.join(dir_path, "*.json")))
    cams: List[Dict[str, Any]] = []
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            # If data itself is a dict with keys, accept directly
            if isinstance(data, dict):
                cams.append(data)
            elif isinstance(data, list):
                cams.extend(data)
        except Exception as e:
            print(f"Warning: failed to load camera json {fp}: {e}")
    return cams


def load_depths_from_dir(dir_path: str, pattern: str = "*.npy") -> List[np.ndarray]:
    """
    Load per-view depth .npy files from a directory, sorted by filename.

    Args:
        dir_path: directory containing depth npy files
        pattern: glob pattern to select files (e.g., '*_rel_depth.npy' or '*_metric_depth.npy')

    Returns:
        List of (H, W) float arrays (relative or metric depth depending on files)
    """
    import os, glob

    files = sorted(glob.glob(os.path.join(dir_path, pattern)))
    depths: List[np.ndarray] = []
    for fp in files:
        try:
            d = np.load(fp)
            if d.ndim != 2:
                print(f"Warning: depth file not 2D: {fp} (shape={d.shape})")
                continue
            depths.append(d.astype(np.float64, copy=False))
        except Exception as e:
            print(f"Warning: failed to load depth npy {fp}: {e}")
    if len(depths) == 0:
        print(f"Warning: no depth files matched pattern '{pattern}' in {dir_path}")
    return depths
