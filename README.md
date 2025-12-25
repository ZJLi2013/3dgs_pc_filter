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

# Attribute-only (no geometry filtering; delete only explicit zero opacity)
python outliers.py -i input.ply -o output.ply

python3 3dgs_pc_filter/outliers.py  -i output_gs/gs_ply/0000.ply  -o output_gs/gs_ply/clean.ply 

```

## AttributeFilter 阈值逻辑（分位数）
- alpha 来源：
  - 若 PLY 已含 alpha（不透明度的激活值，范围 [0,1]），则直接使用
  - 若仅含 opacity（logit），则先做 sigmoid(opacity) 得到 alpha
- 自动阈值（无需手动调参）：
  - 使用 alpha 的分位数作为阈值：ε = P(α, p)
  - 当前默认 p=0.5（单位为“百分比”），表示取 alpha 的“0.5% 分位值”作为阈值，通常会删除约最低 0.5% 的点（具体按分布而定）
- 删除条件：alpha ≤ ε 会被删除，alpha > ε 保留
- 日志输出（运行时会打印）：
  - Alpha stats: min/median/p95/p99/mean/nonzero_ratio（便于观察分布）
  - Alpha exact-zero count: 统计 alpha==0 的点数
  - Alpha percentile epsilon (p=0.5%): 阈值 ε
  - Alpha expected removal ≈ X% (by percentile)：按分位数预估的删除比例
  - Alpha filter: removed N points：实际删除数
- 调整删除强度（如需）：修改 `filters/attribute_filter.py` 中的 `p = 0.5` 为其他百分位（如 1.0 表示删除约 1% 的最低 alpha），无需 CLI 参数
- 设计理念：不依赖几何密度与相机；以“opacity 的激活值（alpha）分布”实现保守的自适应清理，避免误删边缘/高频/细杆等语义结构

## 保存与元属性保留（PLY写出）
- 为什么不用 Open3D 写出：Open3D 的 PointCloud 模型不会保留 PLY 中的自定义顶点属性（如 opacity、scale_0..2、f_dc/f_rest 等），使用 `write_point_cloud` 会导致语义 meta 丢失
- 我们的做法：管线跟踪最终保留的原始索引（`FilterPipeline.final_keep_indices`），再用 `ply_utils.write_filtered_ply` 从原始 PLY 过滤并写出，完整保留 vertex 的 dtype 和字段
- 手动调用示例（如需单独使用）：
```python
from ply_utils import write_filtered_ply
# keep_indices 来自 FilterPipeline.final_keep_indices
write_filtered_ply('input.ply', 'output_clean.ply', keep_indices)
```

默认行为（MVP）：若检测到 3DGS 元属性（如 opacity/scale_0..2/f_dc/f_rest），工具会自动走 `ply_utils.write_filtered_ply` 路径，完整保留 vertex 的 dtype、字段与顺序，并与原始 PLY 的文本/二进制格式保持一致。无需 `--preserve-mode` 开关；默认即保留全部 meta。且不注入 RGB 字段。


## Contribued Filter

- 核心思想：按视角累计每个 Gaussian 的渲染贡献，只删除“长期接近 0 贡献”的点。
- 指标定义：Ci = Σ_v (alpha_i · A_i,v)
  - alpha_i：该点的不透明度（opacity 的 sigmoid 激活，范围 [0,1]）
  - A_i,v：该点在视角 v 的屏幕空间投影面积近似（与相机内参/深度相关）
- 优势：
  - 不依赖空间密度；符合 3DGS 的渲染本质；不易误删边缘/高频/细杆/透明结构。
依赖与输入需求（若后续启用）
- 需要相机参数与位姿：
  - 外参（extrinsics，4×4，world→camera），用于把 3D 点变换到相机坐标系并得到深度 Z。
  - 内参（fx, fy，必要时 cx, cy, width, height），用于近似屏幕面积 A_i,v ≈ (scale_x·scale_y)·(fx·fy)/Z²。
- 示例 JSON（建议结构）：

```json
  {
    "cameras": [
      {
        "fx": 1000.0,
        "fy": 1000.0,
        "width": 1920,
        "height": 1080,
        "extrinsics": [
          [r11, r12, r13, tx],
          [r21, r22, r23, ty],
          [r31, r32, r33, tz],
          [0,   0,   0,   1]
        ]
      }
    ]
  }
```


## TODO

bbox filter
