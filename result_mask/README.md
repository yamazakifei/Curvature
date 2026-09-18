# 单节点静默故障结果

本目录中的新脚本用于评估单个节点静默故障对平均、最大、最小 AoI 的影响。
它不修改 `curvature-gossip/src` 中的现有代码，而是复用现有仿真器，并在每个时隙将指定故障节点的广播动作强制为 `False`。

## 运行

在项目根目录 `D:\ZMF\2026Curvature` 下执行：

```powershell
conda run --no-capture-output -n GRL_AoI_cpu37 python result_mask/run_single_node_silence.py --config result_mask/config_single_node_silence.yaml
```

默认配置与 `configs/GNN/mpnnV3.5_ch1_CurvAttn.yaml` 的拓扑一致：100 节点、两个 50 节点社区、5000 个时隙、100 个 warm-up 时隙、固定广播概率 0.1，并对每个节点分别运行一次故障实验。输出位于：

```text
result_mask/outputs/single_node_silence_p01_update_u02/
```

快速验证时可以限制时隙和节点数：

```powershell
conda run --no-capture-output -n GRL_AoI_cpu37 python result_mask/run_single_node_silence.py --config result_mask/config_single_node_silence.yaml --slots 130 --limit-nodes 3 --output result_mask/outputs/smoke --overwrite
```

## 主要输出

- `single_node_failure_results.csv/json`：基准组和每个故障节点的汇总结果；
- `curvature_aoi_correlations.json`：曲率/绝对曲率与三类 AoI 退化的 Pearson、Spearman 统计量；
- `topology_<seed>/topology_nodes.csv`：节点曲率、曲率排名、度数和介数等信息；
- `topology_<seed>/channel_<seed>/update_<seed>/baseline/`：无故障基准结果；
- `topology_<seed>/channel_<seed>/update_<seed>/node_<id>/`：对应节点静默结果；
- `plots/Curv_vs_aoi_NodeMean.png`：按平均相邻边曲率绘制的 AoI 退化散点图；
- `plots/Curv_vs_aoi_NodeMin.png`：按最小相邻边曲率绘制的 AoI 退化散点图；
- `plots/EdgeCurv_Distribution.png`：边曲率计数直方图；
- `plots/per_node_aoi_impact.png`：按节点曲率排序的三类 AoI 退化图；
- `plots/baseline_vs_failure_aoi.png`：基准和每个节点故障后的 AoI 对比；
- `plots/topology_<seed>_curvature.png`：节点曲率拓扑图；
- `experiment_manifest.json`：实验参数、随机流配对方式和指标定义。

## 指标定义

- `mean_aoi`：warm-up 后，所有有序源-接收节点对和时隙上的平均 AoI；
- `mean_max_aoi`：每个时隙的最大节点对 AoI，再对时隙取平均；
- `mean_min_aoi`：每个时隙的最小节点对 AoI，再对时隙取平均；
- `max_aoi`：warm-up 后所有样本中的最大 AoI；
- `min_aoi`：warm-up 后所有样本中的最小 AoI；
- `delta_<metric>`：故障结果减去同一拓扑、信道和源更新随机流下的基准结果；
- `relative_delta_<metric>`：对应退化量除以基准值；
- `configured_broadcast_probability`、`mean_policy_probability` 和 `average_broadcast_probability`：配置的平均广播概率，固定为 0.1；
- `active_node_actual_tx_ratio`：仅在未故障节点中计算的实际广播比例，便于检查是否仍接近 0.1；
- `actual_tx_ratio`：实际采样动作中的广播比例。故障节点静默后，全网实际比例会略低于 0.1，这是预期现象。

基准组和各节点故障组使用相同的拓扑、阴影衰落、源更新、Rayleigh 衰落和随机广播流。故障节点仍存在并可以接收信息，但不会发送或转发信息。

## 节点最小曲率后处理

运行 `run_single_node_silence.py` 时，在原有 AoI 图生成后会自动生成以下后处理结果。
对于已经完成的旧场景，也可以使用以下独立脚本在不重新运行仿真的情况下补图：

```powershell
conda run --no-capture-output -n GRL_AoI_cpu37 python result_mask/plot_node_min_curvature.py
```

脚本将节点曲率重新定义为相邻边曲率的最小值：

```text
node_min_curvature(v) = min{edge_curvature(v, u) | u belongs to N(v)}
```

每个场景的 `plots/` 目录新增或保留以下结果：

- `Curv_vs_aoi_NodeMean.png`：按平均相邻边曲率重新绘制的 AoI 散点图；
- `Curv_vs_aoi_NodeMin.png`：节点最小相邻边曲率与三类 AoI 退化的散点图；
- `EdgeCurv_Distribution.png`：边曲率计数直方图；
- `topology_<seed>/topology_nodes_node_min.csv`：节点最小曲率及其对应最小曲率边；
- `curvature_postprocess_manifest.json`：后处理定义、直方图 bins 和平均广播概率。

旧的 `curvature_vs_aoi_impact.png` 不再由主实验脚本生成。图标题会标注结果中的平均广播概率。
