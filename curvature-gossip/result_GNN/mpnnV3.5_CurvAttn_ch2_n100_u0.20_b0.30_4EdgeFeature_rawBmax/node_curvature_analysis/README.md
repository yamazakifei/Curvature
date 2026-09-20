# ch2 节点曲率统计与排名分析

分析结果由 `scripts/analyze_ch2_node_curvature.py` 根据本目录的 `source_config.yaml` 复现 20 个固定验证拓扑后生成。由于训练结果目录当前禁止运行时写入，CSV/PNG 实际写入项目根目录下的 `ch2_node_curvature_analysis/`。

三种排名定义为：

1. 节点关联 AF3 边原始曲率的最小值（曲率越小越靠前）；
2. 取同一条最小 AF3 边的 `local_degree_bound` 归一化分数（分数越大越靠前）；
3. 节点关联 ORC 边原始曲率的最小值（曲率越小越靠前，ORC 使用 alpha=0.5）。

主要文件：

- `distribution_summary.csv`：节点分数的均值、总体方差、标准差、中位数和分位数。
- `node_curvature_scores.csv`：20 个场景、每个节点的三类分数及对应边。
- `node_rankings_by_scenario.csv`：每个场景内的 1-based 节点排名。
- `ranking_comparison.csv`：逐场景计算后汇总的 Spearman/Kendall 排名相关性与 Top-10 平均重合率。
- `ch2_node_curvature_analysis.png`：分布箱线图、AF3 原始/归一化散点图、Top-10 重合热图。
- `ch2_normalized_curvature_histogram.png`：归一化节点分数直方图。

注意：这里的“准确性”是结构排名一致性意义上的准确性，不能单独等同于 VAoI 性能准确性；若要验证策略效果，还需要把排名靠前节点与实际广播收益或 VAoI 做进一步配对分析。
