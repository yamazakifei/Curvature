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

## CTDE-PPO V3

V3 使用共享的节点级 MLP Actor（14 维本地输入）和仅训练时使用的集中式 Critic（8 维输入）。Actor 不读取全局真实缓存、全局动作计数或 `broadcast_debt`；V2 的 checkpoint 与 V3 不兼容，必须从头训练。

Actor 特征按固定索引为：度归一化、发送年龄、连续发送次数、拥塞、AF3 瓶颈 max/mean、自上次发送后的信息增量、Freshness mean/max/bottleneck、邻居置信度、`log1p(N)`、更新率 `u`、目标发送率 `b`。Critic 输入为四项 `log1p` VAoI 统计、上一时隙发送率、`log1p(N)`、`u`、`b`。

每个时隙中，发送节点先保存由时隙开始时缓存冻结得到的包快照；只有成功解码才更新接收端保存的邻居缓存估计和解码时隙。因此，下一时隙的 Freshness 只依赖本地可见信息。

预算项使用有界线性拉格朗日优势 `-clip(mu, 0, mu_max) * (a-q_old)`。发送率 EMA 和乘子仅在完整 rollout 结束后更新，带相对误差死区、误差截断和显式上限。

V3 配置：

- `configs/nn_ctde_v3_single.yaml`：100 节点单场景正式训练配置（300 episode）。
- `configs/nn_ctde_v3_smoke.yaml`：2 episode smoke 配置。

### 固定验证：节点级广播选择

`configs/nn_ctde_v3_fixed_validation.yaml` 在不改变 Actor/Critic、特征、PPO、奖励或预算定义的前提下，增加固定的 10 个验证场景。每个场景配置独立的拓扑、源更新、阴影衰落、快衰落和策略动作种子；这些随机流均位于 `fixed_validation` 命名空间，不会消耗训练随机流。

每次验证在相同外生随机流下比较三种策略：冻结 Actor 的 Bernoulli 采样、固定 `q=b` 随机策略、以及使用该场景 Actor 平均概率 `q_bar` 的 matched-rate 随机策略。输出 `validation_history.csv` 和 `validation_per_scenario.csv`，其中配对差值为 `delta_vaoi_matched = VAoI_nn - VAoI_matched`；负值表示 Actor 优于匹配发送率的随机策略。

训练和验证都会记录 Actor 概率的分位数及三类离散度：时隙内节点标准差均值、全部节点时隙总体标准差、节点跨时隙均值的节点间标准差。验证同时检查模型和 Adam 变量在推理前后完全一致。

每 20 个 episode 归档一个 checkpoint；`latest`、验证 VAoI 最低的 `best` 与 matched-rate 差值最低的 `best_matched` 分目录保存。checkpoint 包含 Actor、Critic、Adam 状态，以及记录当前 episode 和配置位置的 `checkpoint_info.json`。

固定验证正式训练命令：

```powershell
conda run -n GRL_AoI_cpu37 python scripts/train_nn_ctde.py --config configs/nn_ctde_v3_fixed_validation.yaml
```

运行 V3 验证：

```powershell
conda run -n GRL_AoI_cpu37 python -m pytest -q
conda run -n GRL_AoI_cpu37 python scripts/train_nn_ctde.py --config configs/nn_ctde_v3_smoke.yaml

cd D:\ZMF\2026Curvature\curvature-gossip
conda run -n GRL_AoI_cpu37 python scripts/train_nn_ctde.py --config configs/nn_ctde_v3_single.yaml
```

## NN_2layers staged work: Stage 0

Stage 0 adds evaluation-only baselines for the staged two-layer Actor work;
it does not train or load a new Actor.  The unified configuration runs:

- `uniform_random`: fixed `q=b` random baseline;
- `curvature_fixed_alpha`: local AF3 incident-bottleneck maximum with a fixed alpha and no residual layer;
- `alpha_override_zero`: strict `alpha_override=0.0` degeneration to `q=b`;
- `matched_rate_random`: a random policy whose probability is set to the reference policy's realized attempt rate for the same topology/channel/update seed tuple.

Run the short smoke evaluation with:

```powershell
conda run -n GRL_AoI_cpu37 python -m curvature_gossip.cli run --config configs/nn_2layers_stage0_smoke.yaml
```

Each per-run `summary.json` and aggregate comparison reports `target_tx_ratio`,
`mean_policy_probability`, `actual_tx_ratio`, `mean_VAoI`, topology count, and
the topology/channel/update seeds.  A `matched_rate_random` entry must follow
its `reference_policy` in YAML because its probability is calibrated from that
completed reference run.

## NN_2layers staged work: Stage 1

Stage 1 trains the shared curvature-only Actor
`sigmoid(logit(b) + softplus(alpha_raw) * (s_kappa - c_kappa))`.  Its only
trainable Actor parameter is `alpha_raw`; no residual MLP is instantiated.
The offline `c_kappa` value is calculated from `training.center_topology_seeds`
when `actor.curvature.center: auto`, then written into the saved training YAML,
model metadata, and checkpoint manifest.

```powershell
conda run --no-capture-output -n GRL_AoI_cpu37 python scripts/train_nn_ctde.py --config configs/nn_2layers_stage1_smoke.yaml
```

Inspect `training_history.csv` for `alpha_raw`, `alpha_kappa`, the `q_base_*`
statistics, `corr_s_kappa_q_base`, PPO diagnostics, VAoI, and actual sending
rate.  The checkpoint under `checkpoints/latest/` is a Stage-1-only actor and
is not compatible with legacy V3 Actor checkpoints.

The paired evaluation smoke configuration uses the frozen center written by
the supplied training smoke run:

```powershell
conda run --no-capture-output -n GRL_AoI_cpu37 python -m curvature_gossip.cli run --config configs/nn_2layers_stage1_eval_smoke.yaml
```

## Stage-2 配置维护

`configs/nn_2layers_stage2_heuristic_channel.yaml` 使用 YAML 锚点统一维护默认
实验条件：`n_nodes`、`update_probability`、`target_tx_ratio` 和验证时长。
训练列表和每个固定验证场景都引用这些锚点，因此修改默认值会同步生效，避免遗漏。
其中 `training.rollout_slots=400` 是训练长度，验证场景使用
`experiment.slots=500`；两者有意独立，若需统一可将前者改为
`rollout_slots: *evaluation_slots`。修改网络规模后，仍须手动检查默认
`topology.params.cluster_sizes` 是否与节点数相符。

## 启发式物理信道下的固定 alpha 对照（2026-07-28）

`configs/nn_2layers_stage1_fixed_alpha_heuristic_channel.yaml` 用于关闭 Stage-2 residual 后，在当前启发式物理信道（无 shadowing/fading）下评估固定 Stage-1 alpha。已完成 `u=0.05/0.10/0.20`、`b=0.10`、`alpha=0.5/1.0/1.2/1.5/1.8` 的配对验证；完整结果及与 Stage-2 最佳验证 checkpoint 的比较见 `result_2layers/fixed_alpha_grid_heuristic_channel_b0.10/comparison_report.md`。
