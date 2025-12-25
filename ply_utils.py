"""
Utilities for PLY read/write while preserving 3DGS vertex attributes
"""

from typing import Sequence
import numpy as np


def write_filtered_ply(
    input_ply: str, output_ply: str, keep_indices: Sequence[int]
) -> None:
    """
    Write a filtered PLY by keeping only rows at keep_indices, preserving all vertex attributes.

    Args:
        input_ply: Path to source PLY file (with full 3DGS vertex attributes)
        output_ply: Path to destination PLY file
        keep_indices: Indices (relative to source vertex list) to keep
    """
    try:
        from plyfile import PlyData, PlyElement
    except ImportError:
        raise RuntimeError(
            "plyfile package not installed. Install with: pip install plyfile"
        )

    plydata = PlyData.read(input_ply)
    vertex = plydata["vertex"]

    # Structured numpy array of shape (N,)
    src = vertex.data
    keep_indices = np.asarray(keep_indices, dtype=np.int64)

    # Safety: clamp indices to valid range
    keep_indices = keep_indices[(keep_indices >= 0) & (keep_indices < src.shape[0])]

    # Filter rows while preserving dtype and field names
    filtered = src[keep_indices]

    # Describe a new vertex element with the same dtype and name
    vertex_el = PlyElement.describe(filtered, "vertex")

    # Preserve any other elements if present (faces, etc.)
    other_elements = [e for e in plydata.elements if e.name != "vertex"]

    new_ply = PlyData([vertex_el] + other_elements, text=plydata.text)

    # Write to output
    new_ply.write(output_ply)
