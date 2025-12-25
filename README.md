# 3DGS Point Cloud 去重（MVP）

3DGS 的点是“带语义的高斯体元”，包含位置、协方差/尺度、opacity、不透明度、SH颜色系数等。传统基于密度/邻居的几何过滤（如激进 ROR/DBSCAN）容易误删“稀疏但语义强”的结构（边缘、细杆、透明区域、低密度精细几何）。因此，本MVP聚焦“语义优先、几何保守”的清理策略：

- 先用 AttributeFilter 按 3DGS 语义（opacity/scale/SH）保留有效点
- MVP 不提供 SOR/ROR/Cluster/ROI/voxel 等几何类过滤；默认仅支持语义过滤（AttributeFilter），后续将加入“基于贡献度”的去重

---

## 快速开始


```bash
pip install open3d numpy
# 3DGS属性读取（必需）
pip install plyfile

# Attribute-only (no geometry filtering, adaptive opacity threshold ~1% removal)
python outliers.py -i input.ply -o output.ply --attr

```

注意事项：
- 优先启用 `--attr`，保护高不透明度/合理尺度/高SH能量点
- MVP 不提供 SOR/ROR/Cluster/ROI/voxel 等几何类入口，避免误删语义结构；默认仅支持语义过滤（AttributeFilter），后续将加入“基于贡献度”的去重。
- 输出写出：若检测到 3DGS 元数据（opacity/scale/SH），工具默认使用 `ply_utils.write_filtered_ply` 写出，保留所有 vertex 属性；Open3D 的 `o3d.io.write_point_cloud` 仅支持标准字段（如 x,y,z,colors,normals），会丢失 3DGS 的自定义属性，不用于有语义元数据的场景

### 保存与元属性保留（PLY写出）
- 为什么不用 Open3D 写出：Open3D 的 PointCloud 模型不会保留 PLY 中的自定义顶点属性（如 opacity、scale_0..2、f_dc/f_rest 等），使用 `write_point_cloud` 会导致语义 meta 丢失
- 我们的做法：管线跟踪最终保留的原始索引（`FilterPipeline.final_keep_indices`），再用 `ply_utils.write_filtered_ply` 从原始 PLY 过滤并写出，完整保留 vertex 的 dtype 和字段
- 手动调用示例（如需单独使用）：
```python
from ply_utils import write_filtered_ply
# keep_indices 来自 FilterPipeline.final_keep_indices
write_filtered_ply('input.ply', 'output_clean.ply', keep_indices)
```

默认行为（MVP）：若检测到 3DGS 元属性（如 opacity/scale_0..2/f_dc/f_rest），工具会自动走 `ply_utils.write_filtered_ply` 路径，完整保留 vertex 的 dtype、字段与顺序，并与原始 PLY 的文本/二进制格式保持一致。无需 `--preserve-mode` 开关；默认即保留全部 meta。且不注入 RGB 字段。

兼容提醒：若 `FilterPipeline.final_keep_indices` 丢失（例如某些异常路径），会退回 Open3D 写出，此时自定义属性将丢失；请确保管线正确提供 `final_keep_indices`。

检测机制：内部通过 `ply_utils.has_3dgs_meta` 自动检测是否存在 3DGS 典型字段（opacity、scale_0..2、f_dc、f_rest），以在保存阶段选择保留元属性的写出路径。

---
