# Curvature-Aware Distributed Gossip Simulator

这是固定单位圆盘无线图上的完整缓存 gossip 仿真器。代码兼容项目指定的 Python 3.7 环境 `GRL_AoI_cpu37`，不包含神经网络、强化学习、移动性、功率控制或广播次数约束。

## 快速运行

```powershell
cd D:\ZMF\2026Curvature\curvature-gossip
$env:PYTHONPATH="$PWD\src"

conda run -n GRL_AoI_cpu37 python -m pytest tests
conda run -n GRL_AoI_cpu37 python -m curvature_gossip.cli run --config configs/smoke_test.yaml
conda run -n GRL_AoI_cpu37 python -m curvature_gossip.cli plot --results results/heuristic_smoke
```

完整首个实验：

```powershell
conda run -n GRL_AoI_cpu37 python -m curvature_gossip.cli run --config configs/first_experiment.yaml
```

已有目录默认不会覆盖；明确需要重跑时添加 `--overwrite`。

## 模型与时隙顺序

每个源以概率 `p_update` 生成新版本。节点保存所有源的最新已知版本，VAoI 为真实源版本减去本地缓存版本。每个时隙严格执行：源更新、冻结包快照、构造本地观测、独立采样广播、生成完整衰落矩阵、最强信号 SINR 解码、同时合并、更新邻居估计、统计指标。同槽收到的信息不能再次转发。

固定拓扑初始化后只计算一次全局 ORC。ORC 使用惰性测度和无权最短路代价，通过 `scipy.optimize.linprog(method="highs")` 求一阶 Wasserstein 距离。运行期策略只看到自身缓存、邻居估计、incident 曲率和本地历史。

## 策略

- `random`：固定广播概率。
- `freshness`：按自身缓存相对邻居估计的正版本优势计算概率。
- `curvature_freshness`：用标准化负 ORC 乘性放大创新增益。

曲率不会形成独立广播奖励，因此无新信息时不会仅因桥边曲率而反复广播。版本 1 不约束或匹配广播次数；输出会报告实际活动量，在归因曲率收益时必须同时检查 `avg_tx_per_slot`。

## 无 ACK 的本地拥塞退避

`freshness` 和 `curvature_freshness` 可选使用本地退避，随机策略保持为无退避基线。节点仅在上一时隙静默时测量噪声加干扰功率，并维护：

```text
c_i[t] = alpha * c_i[t-1] + (1-alpha) * log(1 + I_i[t] / N)
r_i[t] = 连续发送尝试次数
```

广播概率先由新鲜度分数计算为 `q_base`，再修正为：

```text
q_i = clip(q_base * exp(-lambda * c_i) * beta^r_i, q_min, q_max)
```

其中 `congestion_backoff_weight` 是 `lambda`，`congestion_ewma_alpha` 是 `alpha`，`attempt_backoff_factor` 是 `beta`。发送节点不更新干扰测量，且没有任何 ACK 或真实解码结果输入策略。默认 `lambda=0`、`beta=1`，与原启发式策略完全一致。[congestion_backoff_smoke.yaml](configs/congestion_backoff_smoke.yaml) 同时包含关闭/开启退避的新鲜度和 ORC 策略，便于直接作成对比较。

## 输出

每次运行保存在 `results/<experiment>/<topology_seed>/<policy>/`：

- `resolved_config.yaml`
- `summary.json`
- `per_slot.csv`
- `dissemination_delays.csv`
- `edge_curvature.csv`
- `topology_nodes.csv`
- `topology_edges.csv`

实验根目录包含跨独立种子的 `policy_comparison.csv/json` 和六类图片：曲率拓扑、曲率直方图、最大 VAoI 轨迹、最大 VAoI ECDF、策略 VAoI 对比以及 VAoI-广播量散点图。

## 扩展

新增拓扑时继承 `TopologyGenerator` 并使用注册装饰器；新增曲率后端时实现 `CurvatureProvider.compute()`；新增策略时实现仅接收 `NodeObservation` 的 `DistributedBroadcastPolicy`。仿真引擎不需要拓扑或曲率类型条件分支。
