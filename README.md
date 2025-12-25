# 3DGS Point Cloud 去重（MVP）

3DGS 的点是“带语义的高斯体元”，包含位置、协方差/尺度、opacity、不透明度、SH颜色系数等。传统基于密度/邻居的几何过滤（如激进 ROR/DBSCAN）容易误删“稀疏但语义强”的结构（边缘、细杆、透明区域、低密度精细几何）。因此，本MVP聚焦“语义优先、几何保守”的清理策略：

- 先用 AttributeFilter 按 3DGS 语义（opacity/scale/SH）保留有效点
- 再用保守的 SOR 做轻度几何去噪
- 暂不提供 ROR/Cluster/ROI 的CLI入口，避免误删语义结构

---

## 快速开始


```bash
pip install open3d numpy
# 3DGS属性读取（必需）
pip install plyfile

# Attribute-only (no geometry filtering)
python outliers.py -i input.ply -o output.ply --attr --attr-opacity 0.02

# Attribute + conservative SOR
python outliers.py -i input.ply -o output.ply --attr --sor --sor-neighbors 150 --sor-std 2.5
```

注意事项：
- 优先启用 `--attr`，保护高不透明度/合理尺度/高SH能量点
- SOR的 `std_ratio` 越大越保守（推荐 2.0~2.5）
- 默认不启用 ROR/Cluster，以避免误删语义结构

---

## 3DGS语义保留设计方案（Roadmap）

目标：在过滤时引入“语义权重”和“各向异性邻域”，让低密度但语义强的点不被误删。

### Phase A: SemanticScoreFilter（基础）
- 为每个点计算语义置信度 `score ∈ [0,1]`
- 参考：opacity归一化、SH能量、尺度一致性、孤立度罚项
- 提供 `score` 阈值过滤与分布统计

示意公式：
```
score = w_opacity·norm(alpha)
      + w_sh·norm(energy(SH))
      + w_scale·norm(scale_consistency)
      - w_iso·norm(isolation)
```

### Phase B: SemanticWeightedSOR（核心）
- 用邻居语义分数作为权重，计算加权均值/方差
- 对高分点提高 `std_ratio` 容忍度或降低 z-score
- 避免删掉语义强但几何孤立的点

### Phase C: Anisotropic ROR（进阶）
- 使用点的协方差/尺度做马氏距离（各向异性邻域）
- 长轴方向更宽松、短轴方向更严格
- 保留细长结构与边缘细节

### Phase D: Cluster Semantics Guard（优化）
- 在删除小簇前计算簇的平均语义分数与属性方差
- 对“语义高簇”即便很小也保留
- 避免粗暴删除重要的小结构

后续计划：
- 新增预设 `semantic_preserve`
- 文档与示例同步，并提供报告输出（每阶段统计）

---


## FAQ

- 去除率太低（<1%）：降低 `--sor-std` 到 2.0 或 1.5；确认已启用 `--attr`
- 去除率太高（>5%）：提高 `--sor-std` 到 2.5；检查是否误删边缘细节
- 3DGS 属性读取失败：安装 `plyfile` 并确认 PLY 包含 `opacity/scale/SH` 字段

---

本MVP专注于：语义优先 + 几何保守。随着 Phase A-D 的落地，将逐步增强对 3DGS 语义结构的保留能力，避免传统几何过滤误删关键语义。
