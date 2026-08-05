# GNN/CTDE YAML 配置说明（中文）

本文说明 `configs/GNN/*.yaml` 中与 Stage-1 curvature base、Stage-2 MPNN、CTDE PPO、验证和随机种子有关的配置。新推荐配置为：

```text
configs/GNN/mpnn_newC_ch1_search.yaml
```

运行方式不变：

```bash
python scripts/train_nn_ctde.py --config configs/GNN/mpnn_newC_ch1_search.yaml
```

---

## 1. 核心概念

### 1.1 `max_tx_ratio`：广播概率上限 Bmax

```yaml
constraints:
  max_tx_ratio: 0.20
```

含义：全网、跨节点和时间的平均策略广播概率上限。

它表示“不能长期超过多少”，不是“策略应当围绕多少工作”。例如 `Bmax=0.20` 时，最优工作点可以是 `0.10`。

新代码中它用于：

- Stage-1 搜索范围和可行性过滤；
- 预算 advantage / multiplier（启用时）；
- Critic 的场景预算信息；
- 验证场景和固定随机基线；
- 节点活动率诊断。

### 1.2 `base_tx_ratio`：Stage-1 基础工作点 b

```yaml
actor:
  curvature:
    base_tx_ratio: 0.10
```

含义：Stage-1 curvature-only 在校准拓扑分布上的目标平均基础概率。

第一层公式为：

\[
q_i^{\mathrm{base}}
=
\sigma\left(\beta_0+\alpha_\kappa(s_i-c_\kappa)\right).
\]

- `b=base_tx_ratio` 控制整体工作点；
- `alpha` 控制曲率造成的节点间差异；
- `center` 是公共曲率中心；
- `beta0` 是离线校准的公共截距。

### 1.3 Stage-2 residual

\[
q_{i,t}
=
\sigma\left(\operatorname{logit}q_i^{\mathrm{base}}+\delta_{i,t}\right).
\]

Stage-2 根据动态本地状态对第一层进行正负修正。`delta=0` 时最终概率应等于 Stage-1 base。

---

## 2. `experiment`

```yaml
experiment:
  id: my_experiment
  description: "实验说明"
  master_seed: 20260723
  slots: 500
```

### 可选项

- `id`：输出目录中的实验标识，建议唯一；
- `description`：自由文本说明；
- `master_seed`：整次实验唯一根种子；
- `slots`：通用评估时隙数，可通过 YAML anchor 复用。

修改 `master_seed` 会联动改变自动模式下的：

1. beta0 校准拓扑池；
2. Stage-1 搜索池；
3. Stage-2 训练 episode；
4. 固定验证池。

同一个配置和 `master_seed` 重跑应完全可复现。

---

## 3. `randomness`

```yaml
randomness:
  mode: derived_namespaces
  save_manifest: true
  derivation_version: v1
```

### `mode`

- `derived_namespaces`：推荐。使用 `master_seed + 固定命名空间 + 索引` 派生所有随机流；
- 旧 YAML 没有此段时，继续使用现有显式 seed 或旧训练 seed 逻辑。

### `save_manifest`

- `true`：保存 `resolved_seed_manifest.json`；
- `false`：不保存，不推荐。

### `derivation_version`

随机流规则的版本标识。以后若修改命名空间或索引结构，应升级版本，避免同一 master seed 的含义悄然改变。

固定命名空间应区分：

```text
stage1_calibration
stage1_search
stage2_training
fixed_validation
```

每个池内再区分 topology、source updates、shadowing、fading、actions。

---

## 4. `topology`

示例：

```yaml
topology:
  type: random_geometric
  params:
    n_nodes: 100
    area_width_m: 200.0
    area_height_m: 200.0
    communication_radius_m: 40.0
    max_attempts: 500
```

### 常用 `type`

具体可用类型以仓库 topology generator 注册表为准。当前 GNN 实验常用：

- `random_geometric`：随机几何图；
- `soft_two_community`：带两个软社区的拓扑。

### `random_geometric` 常用参数

- `n_nodes`：节点数；
- `area_width_m/area_height_m`：区域尺寸；
- `communication_radius_m`：通信半径；
- `max_attempts`：生成满足约束拓扑的最大尝试次数。

### `soft_two_community` 常用参数

- `cluster_sizes`；
- `center_separation_m`；
- `cluster_radius_m`；
- `cross_edge_count_range`；
- `min_degree`；
- `min_edge_connectivity`；
- `require_induced_cluster_connected`；
- `max_attempts`。

校准、搜索、训练和验证默认使用相同的 topology 类型与参数，但不同随机流。

---

## 5. `source`

```yaml
source:
  update_probability: 0.20
```

- `update_probability`：每个节点每时隙产生新版本的概率；
- 必须位于 `[0,1]`；
- 也可以在 `training.update_probabilities` 中设置多场景训练列表。

---

## 6. `channel`

```yaml
channel:
  tx_power_dbm: 20.0
  pathloss_reference_db: 30.0
  reference_distance_m: 1.0
  pathloss_exponent: 2.5
  shadowing_std_db: 0.0
  reciprocal_shadowing: true
  rayleigh_fading: false
  noise_psd_dbm_per_hz: -174.0
  bandwidth_hz: 1000000.0
  noise_figure_db: 5.0
  sinr_threshold_db: -3.0
```

### 主要字段

- `tx_power_dbm`：发送功率；
- `pathloss_reference_db`：参考距离处路径损耗；
- `reference_distance_m`：参考距离；
- `pathloss_exponent`：路径损耗指数；
- `shadowing_std_db`：阴影衰落标准差；
- `reciprocal_shadowing`：链路阴影是否互易；
- `rayleigh_fading`：是否启用每时隙瑞利衰落；
- `noise_psd_dbm_per_hz`：噪声功率谱密度；
- `bandwidth_hz`：带宽；
- `noise_figure_db`：噪声系数；
- `sinr_threshold_db`：成功解码门限。

Stage-1 搜索必须让所有候选复用相同的 channel realizations。

---

## 7. `curvature`

```yaml
curvature:
  method: distributed_af3
  normalization: local_degree_bound
```

### `method`

当前分布式设计推荐：

- `distributed_af3`。

其他已实现曲率方法以仓库 curvature provider 为准。

### `normalization`

AF3 当前推荐：

- `local_degree_bound`。

该归一化只依赖边两端局部度数，不需要全网负曲率最大值。

---

## 8. `constraints`

```yaml
constraints:
  max_tx_ratio: 0.20
  per_node_cap_multiplier: 1.5
```

### `max_tx_ratio`

新字段，表示 `Bmax`。

兼容旧字段：

```yaml
constraints:
  target_tx_ratio: 0.20
```

若新旧字段同时存在，以 `max_tx_ratio` 为准；建议记录 warning 或拒绝不一致值。

### `per_node_cap_multiplier`

定义节点长期活动率诊断阈值：

\[
q_{\mathrm{node,diag}}=\min(1,B_{\max}\times m).
\]

当前实现中它不是神经网络输出概率的硬裁剪，除非以后另行实现 hard cap。

---

## 9. `actor`

### 9.1 `stage`

```yaml
actor:
  stage: 2
```

- `1`：只运行曲率基础策略；
- `2`：曲率基础策略 + 动态 residual；
- 旧 `0`：legacy Actor，按现有兼容逻辑处理。

### 9.2 `actor.curvature`

```yaml
actor:
  curvature:
    enabled: true
    score: incident_bottleneck_max
    center: auto
    alpha_parameterization: softplus
    base_tx_ratio: 0.10
    alpha_init: 1.5
    alpha_override: 1.5
```

#### `enabled`

- `true`：使用曲率；
- `false`：no-curvature ablation。Stage-2 还需关闭 Stage-1 reference 和曲率边特征。

#### `score`

当前 Stage-1 推荐：

- `incident_bottleneck_max`。

#### `center`

- `auto`：在 calibration topology pool 上计算并冻结公共中心；
- 数值：直接使用固定中心；
- 搜索启用时推荐 `auto`。

#### `alpha_parameterization`

当前实现：

- `softplus`。

#### `base_tx_ratio`

- 搜索关闭时直接使用；
- 搜索开启时作为回退值和配置参考，最终由搜索结果覆盖；
- 必须满足 `0 < base_tx_ratio <= max_tx_ratio`。

#### `alpha_init`

`alpha` 可学习时的 softplus 初始值。

#### `alpha_override`

- 数值：固定 alpha；
- `null`：允许使用可学习 `alpha_init`；
- 搜索开启后，最终用 `selected_alpha` 覆盖并冻结。

### 9.3 `actor.curvature.base_search`

```yaml
base_search:
  enabled: true
```

- `true`：训练前自动搜索 Stage-1；
- `false`：直接使用 `base_tx_ratio + alpha_override`。

#### `calibration`

```yaml
calibration:
  topology_count: 32
  center_statistic: mean
  intercept_tolerance: 1.0e-6
  intercept_max_iterations: 100
```

- `topology_count`：校准拓扑数量；
- `center_statistic`：公共曲率中心统计方式，第一版建议支持 `mean`，可扩展 `median`；
- `intercept_tolerance`：beta0 二分校准的平均概率误差容限；
- `intercept_max_iterations`：最大迭代次数。

校准池只计算拓扑和曲率，不运行动态仿真。

#### `base_candidates`

```yaml
base_candidates:
  mode: coarse_to_fine
  min_fraction_of_max: 0.25
  max_fraction_of_max: 1.0
  coarse_count: 7
  refine: true
  refine_count: 5
  refine_per_alpha: true
```

- `mode`：第一版使用 `coarse_to_fine`；
- `min_fraction_of_max`：粗搜索最小 b 与 Bmax 的比例；
- `max_fraction_of_max`：最大比例，不得超过 1；
- `coarse_count`：粗搜索点数；
- `refine`：是否局部细化；
- `refine_count`：每个 alpha 的细化点数；
- `refine_per_alpha`：必须为 true，表示每个 alpha 独立细化。

#### `alpha_candidates`

```yaml
alpha_candidates: [0.0, 0.5, 1.0, 1.5, 2.0]
```

- 只使用显式给定列表；
- 不根据 Bmax 自动生成；
- 省略时可默认 `[alpha_override]`；
- 建议包含 `0.0` 作为均匀基线。

#### `evaluation`

```yaml
evaluation:
  topology_count: 20
  dynamic_repeats: 2
  coarse_topology_count: 10
  coarse_dynamic_repeats: 1
  slots: 500
  metric: mean_VAoI
  max_probability_tolerance: 0.005
  tie_relative_tolerance: 0.01
  prefer_lower_probability_on_tie: true
```

- `topology_count`：完整搜索池 topology 数；
- `dynamic_repeats`：每张 topology 的动态随机重复数；
- `coarse_topology_count`：粗搜索使用完整池前多少张 topology；
- `coarse_dynamic_repeats`：粗搜索每张 topology 使用多少动态重复；
- `slots`：每个搜索场景时隙数；
- `metric`：当前推荐 `mean_VAoI`；
- `max_probability_tolerance`：允许超过 Bmax 的数值容差；
- `tie_relative_tolerance`：VAoI 近似相同的判定阈值；
- `prefer_lower_probability_on_tie`：近似相同时优先低概率方案。

#### `save_per_scenario`

- `true`：保存每个候选、每个场景的结果；
- `false`：只保存聚合结果；
- 推荐 true，便于配对统计。

### 9.4 `actor.residual`

```yaml
residual:
  enabled: true
  architecture: mpnn
  hidden_dims: [64, 64]
  activation: relu
  delta_max: 1.0
  zero_init_output: true
  use_stage1_reference: true
  detach_stage1_reference: true
  include_scenario_context: false
  include_curvature_mean: false
  include_curvature_freshness_interaction: false
  freeze_stage1: true
```

- `enabled`：Stage-2 必须为 true；
- `architecture`：`mlp` 或 `mpnn`；
- `hidden_dims`：当前实现要求 `[64,64]`；
- `activation`：当前要求 `relu`；
- `delta_max`：residual 的 tanh 范围；
- `zero_init_output`：输出层零初始化，使初始 residual=0；
- `use_stage1_reference`：Stage-2 是否输入 Stage-1 概率；
- `detach_stage1_reference`：阻断 residual 通过参考输入反传到 Stage-1；
- `include_scenario_context`：Actor 是否输入 N/u/Bmax 等场景变量；严格单场景可关闭；
- `freeze_stage1`：Stage-1 搜索后应为 true。

#### `residual.mpnn`

```yaml
mpnn:
  message_hidden_dims: [32]
  message_dim: 16
  update_hidden_dims: [64, 64]
  aggregation: mean_max
  condition_message_on_receiver: true
  use_sender_node_features: false
  use_curvature_edge_feature: true
  bmax: 1.0
```

- `aggregation`：当前要求 `mean_max`；
- `condition_message_on_receiver`：消息是否结合接收节点状态；
- `use_sender_node_features`：当前分布式设计为 false；
- `use_curvature_edge_feature`：是否输入局部边瓶颈分数；
- `bmax`：曲率边特征裁剪/归一化上界，不是广播概率 Bmax。

---

## 10. `critic`

```yaml
critic:
  architecture: node_conditioned
  hidden_dims: [64, 64]
  activation: relu
  include_scenario_context: true
  include_exact_node_features: true
  include_channel_features: true
  include_curvature_features: true
  advantage_mode: node_gae
  bootstrap_rollout_end: true
```

### 可选项

- `architecture`：`node_conditioned` 或旧 `scalar_global`；
- `include_scenario_context`：加入 N/u/Bmax；
- `include_exact_node_features`：训练期精确节点创新/VAoI 特征；
- `include_channel_features`：训练期信道统计；
- `include_curvature_features`：是否包含曲率特征；
- `advantage_mode`：当前 node-conditioned Critic 要求 `node_gae`；
- `bootstrap_rollout_end`：使用 rollout 结束后的真实下一状态 bootstrap。

Critic 可使用全局训练信息，不影响 Actor 的分布式执行。

---

## 11. `observation`

```yaml
observation:
  consecutive_tx_scale: 3.0
  neighbor_confidence_time_constant: 20.0
  congestion_ewma_beta: 0.8
  congestion_feature_scale: 5.0
```

- `consecutive_tx_scale`：连续发送次数归一化尺度；
- `neighbor_confidence_time_constant`：邻居缓存估计置信度衰减常数；
- `congestion_ewma_beta`：拥塞 EWMA 平滑系数；
- `congestion_feature_scale`：拥塞特征缩放。

---

## 12. `training`

```yaml
training:
  episodes: 300
  rollout_slots: 400
  max_tx_ratios: [0.20]
  node_counts: [100]
  update_probabilities: [0.20]
```

### 场景字段

- `episodes`：PPO episode 数；
- `rollout_slots`：每个 episode 时隙数；
- `max_tx_ratios`：Bmax 列表；
- `node_counts`：N 列表；
- `update_probabilities`：u 列表。

三者笛卡尔积构成训练场景。旧字段 `target_tx_ratios` 继续兼容。

自动 seed 模式不需要：

```yaml
training:
  topology_seed_start: 40000
```

旧配置存在该字段时应继续支持。

### PPO 参数

```yaml
actor_learning_rate: 0.0001
critic_learning_rate: 0.0003
common_mode_coefficient: 0.05
clip_ratio: 0.2
entropy_coefficient: 0.0
gamma: 0.99
gae_lambda: 0.95
ppo_epochs: 4
vaoi_reward_scale: 0.1
```

- `actor_learning_rate/critic_learning_rate`：独立学习率；
- `common_mode_coefficient`：惩罚同一时隙 residual 均值偏离 0；
- `clip_ratio`：PPO clipping；
- `entropy_coefficient`：Bernoulli entropy 权重；当前 q<0.5 时正 entropy 会推动概率上升，当前实验建议 0；
- `gamma`：回报折扣；
- `gae_lambda`：GAE 参数；
- `ppo_epochs`：同一 rollout 更新轮数；
- `vaoi_reward_scale`：VAoI reward 缩放。

### 预算约束参数

```yaml
use_budget_advantage: false
update_multiplier: false
initial_multiplier: 0.0
multiplier_learning_rate: 0.01
multiplier_max: 0.15
tx_ratio_ema_beta: 0.5
budget_relative_tolerance: 0.05
multiplier_error_clip: 1.0
```

组合含义：

- 两者都 false：不通过 PPO budget advantage 限制；
- `use_budget_advantage=true, update_multiplier=false`：使用固定乘子；初始乘子为 0 时无作用；
- 两者都 true：根据 rollout 实际广播率更新乘子并加入 advantage。

Stage-1 搜索中的 Bmax 可行性过滤与此开关独立，即开关关闭时搜索仍不会选择明显超过 Bmax 的基础策略。

### checkpoint

```yaml
initial_checkpoint: null
restore_actor_only: true
```

- `initial_checkpoint`：初始 checkpoint 路径；
- `restore_actor_only`：只恢复 Actor，Critic 重新初始化。

---

## 13. `validation`

### 13.1 自动模式（推荐）

```yaml
validation:
  enabled: true
  trigger: every_episodes
  every_episodes: 10
  checkpoint_mode: validation_best_only
  scenario_generation:
    mode: auto
    scenario_count: 20
    slots: 500
    n_nodes: 100
    update_probability: 0.20
    max_tx_ratio: 0.20
```

- 启动时从 `master_seed` 生成一次固定验证集；
- 整个训练过程每次 checkpoint 评估复用同一组场景；
- 自动场景写入 seed manifest。

### 13.2 显式模式（兼容旧实验）

```yaml
validation:
  scenario_generation:
    mode: explicit
  scenarios:
    - id: v00
      topology_seed: 51000
      source_update_seed: 61000
      channel_shadowing_seed: 71000
      channel_fading_seed: 81000
      policy_action_seed: 91000
      slots: 500
      n_nodes: 100
      update_probability: 0.20
      max_tx_ratio: 0.20
```

旧场景中的 `target_tx_ratio` 继续作为 `max_tx_ratio` 别名。

### 13.3 触发方式

- `trigger: every_episodes`：按固定 episode 间隔；
- 已有其他触发方式以当前代码支持为准，例如训练 VAoI 改善触发。

### 13.4 checkpoint 模式

- `validation_best_only`：只保存验证最优模型；
- `periodic_and_latest`：保留周期 checkpoint 和 latest。

---

## 14. `output`

```yaml
output:
  root: result_GNN
  save_per_slot: false
  make_plots: false
```

- `root`：输出根目录；
- `save_per_slot`：是否保存逐时隙仿真结果；
- `make_plots`：是否自动绘图。

Stage-1 搜索启用后还应保存：

```text
stage1_calibration.json
stage1_search_results.csv
stage1_search_per_scenario.csv
stage1_search_summary.json
resolved_seed_manifest.json
resolved_training_config.yaml
```

原始 YAML 不应被覆盖。

---

## 15. 四个集合的默认划分

| 集合 | 默认规模 | 是否运行完整仿真 | 作用 |
|---|---:|---:|---|
| beta0 校准池 | 32 topology | 否 | center 与 beta0 校准 |
| Stage-1 搜索池 | 20 topology x 2 repeats | 是 | 选择 b 与 alpha |
| Stage-2 训练池 | 300 episodes | 是 | PPO 梯度训练 |
| 固定验证池 | 20 scenarios | 是 | checkpoint 选择 |

默认由不同 namespace 完全隔离。

粗搜索使用搜索池的子集，不单独构成验证集。最终验证池禁止参与 Stage-1 参数选择。

---

## 16. 常用配置组合

### 16.1 自动搜索 Stage-1

```yaml
constraints:
  max_tx_ratio: 0.20
actor:
  curvature:
    base_tx_ratio: 0.10
    alpha_override: 1.5
    base_search:
      enabled: true
      alpha_candidates: [0.0, 0.5, 1.0, 1.5, 2.0]
```

搜索结果覆盖 `base_tx_ratio` 和 `alpha_override`，然后冻结第一层。

### 16.2 不搜索，直接指定 Stage-1

```yaml
actor:
  curvature:
    base_tx_ratio: 0.10
    alpha_override: 1.5
    base_search:
      enabled: false
```

适合复现实验或消融。

### 16.3 只搜索 b，固定 alpha

```yaml
base_search:
  enabled: true
  alpha_candidates: [1.5]
```

### 16.4 无曲率 Stage-2 消融

```yaml
actor:
  curvature:
    enabled: false
    alpha_override: 0.0
  residual:
    use_stage1_reference: false
    mpnn:
      use_curvature_edge_feature: false
```

### 16.5 启用平均广播率预算乘子

```yaml
training:
  use_budget_advantage: true
  update_multiplier: true
```

这不会增加 Actor 网络层，只改变训练 advantage 和乘子状态。

---

## 17. 兼容性规则摘要

新旧字段建议按以下优先级解析：

| 规范语义 | 新字段 | 旧字段回退 |
|---|---|---|
| Bmax | `constraints.max_tx_ratio` | `constraints.target_tx_ratio` |
| 训练 Bmax 列表 | `training.max_tx_ratios` | `training.target_tx_ratios` |
| 验证 Bmax | `max_tx_ratio` | `target_tx_ratio` |
| Stage-1 b | `actor.curvature.base_tx_ratio` | Bmax，保持旧行为 |
| 验证场景 | `scenario_generation` | 旧 `validation.scenarios` |
| 训练 seed | derived namespace | 旧 `topology_seed_start` |

新旧字段同时存在但值不一致时，不应静默选择；建议报错或至少给出高可见度 warning。

---

## Stage-1 搜索实现说明（2026-08-04）

训练入口在创建 PPO Actor/Critic 之前执行 `actor.curvature.base_search`：

1. 由 `constraints.max_tx_ratio` 生成默认 7 个 coarse `b` 候选；
2. 在独立 calibration topology pool 上计算公共 `c_kappa`；
3. 对每个 `(b, alpha)` 用 pooled mean probability 二分校准一个公共 `beta0`；
4. 每个 alpha 独立选择 coarse 最优 b，并在相邻区间内 refine；
5. 用完整的 `20 x 2` 配对搜索场景复核，并按约束、mean VAoI、概率和 b 排序；
6. 将选中的 `base_tx_ratio`、`alpha_override`、`center` 和
   `calibrated_intercept` 写入 resolved config，并冻结 Stage-1。

搜索候选不会进入随机流命名空间。四个集合使用
`stage1_calibration`、`stage1_search`、`stage2_training` 和
`fixed_validation`，具体 topology/source/shadowing/fading/action seed 记录在
`resolved_seed_manifest.json`。自动验证场景只生成一次，所有 checkpoint 复用同一场景定义。

实验目录会保存：

```text
stage1_calibration.json
stage1_search_results.csv
stage1_search_per_scenario.csv
stage1_search_summary.json
resolved_seed_manifest.json
resolved_training_config.yaml
```

旧 YAML 仍支持 `target_tx_ratio`、`target_tx_ratios` 和显式
`validation.scenarios`；加载时会给出一次 warning，并分别规范化为 Bmax、Bmax
列表和 explicit validation 模式。

搜索完成后，终端会打印最优 `b`、`alpha`、`beta0`、`c_kappa`、搜索平均概率、
实际广播比例、mean VAoI、约束可行性和结果目录，便于直接记录本次搜索选择。

## 18. 配置检查建议

程序启动时应检查：

- `0 < base_tx_ratio <= max_tx_ratio <= 1`；
- alpha candidates 非空、有限、非负；
- coarse/refine count 合法；
- calibration/search/validation 数量为正；
- coarse 数量不超过完整搜索池；
- `freeze_stage1=true`（搜索启用且 Stage-2）；
- auto validation 不同时要求显式 scenarios；
- explicit validation 场景字段完整；
- 同一 master seed 可重复得到同一 seed manifest；
- 搜索候选不参与随机流派生。
