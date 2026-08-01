# 在新分支 GNN 中实现分布式 local edge-message MPNN

## 0. 基准与分支要求
新建目录result_GNN保存结果

要求：

1. 不要在 `NN_2layers` 分支直接修改；
2. 不要强制覆盖已经存在的远程分支；
3. 不要提交训练结果、checkpoint、CSV 或 `result_GNN/`；
4. 不要引入 PyTorch、PyTorch Geometric、DGL 或其他新依赖；
5. 继续兼容现有 Python 3.7 和 TensorFlow 1.x/`tensorflow.compat.v1` 环境。

---

# 1. 修改目标

当前 Stage 2 使用共享节点 MLP：

[
q_i
===

\sigma\left[
z_i^{\mathrm{base}}
+
\Delta_\theta
\left(
x_i^{\mathrm{aggregated}},
\operatorname{sg}(q_i^{\mathrm{base}})
\right)
\right].
]

其中邻居状态已经被人工聚合成：

* `neighbor_freshness_mean`；
* `neighbor_freshness_max`；
* `neighbor_confidence_mean`。

本次修改增加一种新的 Stage-2 residual 架构：

```yaml
actor:
  stage: 2
  residual:
    architecture: mpnn
```

目标公式：

[
z_i^{\mathrm{base}}
===================

\operatorname{logit}(b)
+
\alpha_\kappa(s_{\kappa,i}-c_\kappa),
]

[
q_i^{\mathrm{base}}=\sigma(z_i^{\mathrm{base}}),
]

[
m_{j\rightarrow i}
==================

\phi_m
\left(
x_i,e_{j\rightarrow i}
\right),
]

[
m_i^{\mathrm{agg}}
==================

\left[
\operatorname{mean}*{j\in N(i)}m*{j\rightarrow i},
\operatorname{max}*{j\in N(i)}m*{j\rightarrow i}
\right],
]

[
\Delta_i
========

\delta_{\max}
\tanh
\left(
\phi_u
\left[
x_i,
m_i^{\mathrm{agg}},
\operatorname{sg}(q_i^{\mathrm{base}})
\right]
\right),
]

[
q_i
===

\sigma
\left(
z_i^{\mathrm{base}}+\Delta_i
\right).
]

这是一个严格本地执行的 edge-message MPNN：

* 不读取邻居当前真实状态；
* 不读取邻居当前隐藏表示；
* 不增加每时隙控制消息；
* 不增加免费通信；
* 仅使用 `NodeObservation` 中节点自己已经维护的逐邻居缓存估计。

---

# 2. 架构定位

不要新增 `actor.stage: 3`。

GNN 不是第三层，而是 Stage 2 residual 的另一种实现，因此保持：

```yaml
actor:
  stage: 2
```

新增：

```yaml
actor:
  residual:
    architecture: mpnn
```

现有配置没有 `architecture` 字段时，必须默认：

```yaml
architecture: mlp
```

从而保证所有旧配置、旧测试和旧 checkpoint 路径维持原行为。

支持：

```text
residual.architecture = mlp
residual.architecture = mpnn
```

拒绝其他值。

---

# 3. 分布式信息约束

本版本不允许 MPNN 使用：

* 邻居当前 `node_context`；
* 邻居当前隐藏状态；
* 全局 cache；
* 全局 VAoI；
* 当前时隙全局动作；
* 全局发送数；
* 未通过已有无线接收过程获得的信息；
* 节点标签或 one-hot node ID；
* 任何免费同步控制信道。

虽然训练代码会一次性编码全部节点以便向量化，但每个节点 (i) 的输出必须仅依赖：

1. 节点 (i) 自己的本地状态；
2. 节点 (i) 已保存的逐邻居估计；
3. 节点 (i) 的第一层曲率基准；
4. 共享网络参数。

改变节点 (j) 的当前自身状态时，如果节点 (i) 的本地观测没有变化，则节点 (i) 的结果不得变化。

---

# 4. 节点自身特征

为 MPNN 增加独立节点特征编码，不要直接继续使用当前 8 维 `residual_context`。

节点自身特征固定为：

```text
normalized_degree
time_since_last_tx
consecutive_tx_attempts
congestion_ewma
self_information_increment
```

公式保持与现有实现一致。

## 4.1 normalized degree

[
x_{i,1}
=======

\frac{\log(1+d_i)}{\log(1+N)}.
]

## 4.2 time since last transmission

[
x_{i,2}
=======

\tanh
\left(
b,\tau_i^{\mathrm{tx}}
\right).
]

## 4.3 consecutive attempts

[
x_{i,3}
=======

\tanh
\left(
\frac{r_i^{\mathrm{tx}}}
{\text{consecutive_tx_scale}}
\right).
]

## 4.4 congestion

[
x_{i,4}
=======

\tanh
\left(
\frac{c_i}
{\text{congestion_feature_scale}}
\right).
]

## 4.5 self-information increment

[
x_{i,5}
=======

\frac{1}{N}
\sum_{k=1}^{N}
\mathbf 1
\left[
V_{i,k}>
V_{i,k}^{\mathrm{last\ tx}}
\right].
]

默认：

```text
node_context_dim = 5
```

若：

```yaml
include_scenario_context: true
```

则在节点特征末尾追加：

```text
log_network_size
update_probability
target_tx_ratio
```

此时：

```text
node_context_dim = 8
```

正式首轮实验保持：

```yaml
include_scenario_context: false
```

---

# 5. 逐边动态特征

物理拓扑是无向图，但本地邻居估计具有方向性。

对于每条无向边：

```text
{i, j}
```

生成两条有向记录：

```text
j -> i
i -> j
```

其中 `j -> i` 表示使用节点 (i) 本地保存的节点 (j) 状态估计，最终聚合到节点 (i)。

每条有向边固定使用三个特征：

```text
neighbor_estimate_valid
neighbor_freshness_gain
neighbor_estimate_confidence
```

## 5.1 validity

[
v_{ij}
======

\begin{cases}
1,&\text{节点 }i\text{ 的邻居 }j\text{ 估计有效},\
0,&\text{否则}.
\end{cases}
]

## 5.2 potential freshness gain

[
g_{ij}
======

v_{ij}
\frac{1}{N}
\sum_{k=1}^{N}
\mathbf 1
\left[
V_{i,k}>
\widehat V_{j,k}^{(i)}
\right].
]

无效估计时：

```text
g_ij = 0
```

## 5.3 confidence

[
c_{ij}
======

v_{ij}
\exp
\left(
-\frac{a_{ij}}{\tau_c}
\right).
]

无效估计时：

```text
c_ij = 0
```

因此：

```text
edge_feature_dim = 3
```

所有边特征必须有限且位于 `[0,1]`。

第一版禁止加入：

* AF3 edge importance；
* incident curvature；
* curvature mean；
* curvature-freshness interaction；
* physical distance；
* 全局中心性；
* 邻居当前节点特征；
* 邻居隐藏 embedding。

---

# 6. 新的编码数据结构

修改：

```text
curvature-gossip/src/curvature_gossip/learning/features.py
```

新增：

```python
@dataclass(frozen=True)
class Stage2MPNNEncodedObservations:
    curvature_scores: np.ndarray
    target_tx_ratios: np.ndarray
    node_context: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray
```

形状：

```text
curvature_scores: [N, 1], float32
target_tx_ratios: [N], float32
node_context: [N, F_node], float32
edge_index: [2, E_directed], int32
edge_features: [E_directed, 3], float32
```

`edge_index[0]` 为 sender row，`edge_index[1]` 为 receiver row。

虽然第一版 Actor 不读取 sender 当前特征，仍保存完整 `edge_index`：

* 便于测试图结构；
* 便于检查方向；
* 便于未来扩展缓存 embedding；
* 避免把它退化成无图索引的普通矩阵接口。

新增常量：

```python
STAGE2_MPNN_NODE_FEATURE_NAMES = (
    "normalized_degree",
    "time_since_last_tx",
    "consecutive_tx_attempts",
    "congestion_ewma",
    "self_information_increment",
)

STAGE2_MPNN_EDGE_FEATURE_NAMES = (
    "neighbor_estimate_valid",
    "neighbor_freshness_gain",
    "neighbor_estimate_confidence",
)
```

新增函数：

```python
encode_stage2_mpnn_observations(...)
```

编码必须：

1. 建立 `node_id -> row` 映射；
2. 验证所有邻居都存在于当前 observation batch；
3. 为每个 `observation.neighbor_ids` 生成一条有向边；
4. 保持边特征与 `neighbor_cache_estimates`、validity、age 严格对齐；
5. 不读取全局 simulator state；
6. 不调用全局图算法；
7. 支持节点输入顺序发生排列；
8. 支持理论上的零度节点；
9. 输出 deterministic ordering；
10. 不把旧的手工 freshness mean/max 放入 `node_context`。

可以抽取公共 helper 复用节点自身特征公式，但不要改变 legacy V3 和 MLP Stage-2 的数值行为。

同步修改：

```text
curvature-gossip/src/curvature_gossip/learning/__init__.py
```

导出新的 dataclass、feature names 和 encoder。

---

# 7. MPNN 网络结构

修改：

```text
curvature-gossip/src/curvature_gossip/learning/ctde_ppo.py
```

在 Stage 2 内根据：

```python
residual_architecture = residual.get("architecture", "mlp")
```

选择旧 MLP 或新 MPNN。

旧 `mlp` 路径必须保持原样。

## 7.1 Placeholder

MPNN 新增：

```python
self.mpnn_node_context
self.mpnn_edge_features
self.mpnn_edge_receivers
```

形状：

```text
mpnn_node_context: [None, F_node]
mpnn_edge_features: [None, 3]
mpnn_edge_receivers: [None]
```

`mpnn_edge_receivers` 使用 `int32`。

不需要把 sender node context 输入 TensorFlow 图。

## 7.2 Receiver-conditioned edge message

通过：

```python
receiver_context = tf.gather(
    self.mpnn_node_context,
    self.mpnn_edge_receivers,
)
```

形成：

```python
message_input = tf.concat(
    [receiver_context, self.mpnn_edge_features],
    axis=1,
)
```

消息网络：

```text
[node_context_dim + 3]
    -> Dense(32, ReLU)
    -> Dense(16, ReLU)
```

作用域：

```text
actor/residual_mpnn/message_mlp
```

输出：

```text
edge_messages: [E_directed, 16]
```

固定配置：

```yaml
message_hidden_dims: [32]
message_dim: 16
```

## 7.3 Mean aggregation

使用：

```python
tf.math.unsorted_segment_sum
```

计算每个 receiver 的消息和和边数，再计算均值。

不能依赖边已经按 receiver 排序。

对于零度节点：

```text
mean_message = all zeros
```

不得出现除零或 NaN。

## 7.4 Max aggregation

使用：

```python
tf.math.unsorted_segment_max
```

计算每个 receiver 的逐维最大消息。

TensorFlow 对空 segment 可能给出极小值，因此必须用 edge count mask 将零度节点的 max message 显式设置为零。

最终：

```text
aggregated_message =
concat(mean_message, max_message)
```

形状：

```text
[N, 32]
```

配置记录：

```yaml
aggregation: mean_max
```

首版不实现 attention、softmax weighting 或多层传播。

## 7.5 Node update

完整曲率 MPNN：

```python
decoder_input = concat([
    node_context,
    aggregated_message,
    stop_gradient(q_base[:, None]),
])
```

默认维数：

```text
5 + 32 + 1 = 38
```

严格无曲率 MPNN：

```python
decoder_input = concat([
    node_context,
    aggregated_message,
])
```

默认维数：

```text
5 + 32 = 37
```

节点更新网络：

```text
decoder_input
    -> Dense(64, ReLU)
    -> Dense(64, ReLU)
    -> Dense(1, linear, zero initialized)
```

作用域：

```text
actor/residual_mpnn/update_mlp
```

输出：

```python
residual_delta = delta_max * tf.tanh(residual_raw)
```

最终：

```python
final_logits = base_logits + residual_delta
probabilities = tf.sigmoid(final_logits)
```

禁止改为概率空间加法。

---

# 8. 配置结构

建议格式：

```yaml
actor:
  stage: 2

  curvature:
    score: incident_bottleneck_max
    center: auto
    alpha_parameterization: softplus
    alpha_init: 1.0
    alpha_override: 1.0

  residual:
    enabled: true
    architecture: mpnn
    activation: relu
    delta_max: 1.0
    zero_init_output: true

    use_stage1_reference: true
    detach_stage1_reference: true
    freeze_stage1: true

    include_scenario_context: false
    include_curvature_mean: false
    include_curvature_freshness_interaction: false

    mpnn:
      message_hidden_dims: [32]
      message_dim: 16
      update_hidden_dims: [64, 64]
      aggregation: mean_max
      condition_message_on_receiver: true
      use_sender_node_features: false
      use_curvature_edge_feature: false
```

---

# 9. 配置校验与版本信息

修改：

```text
curvature-gossip/src/curvature_gossip/learning/trainer.py
```

扩展 `_stage1_actor_config()`，但可以保留函数名以避免破坏现有测试。

## 9.1 MLP

没有 `architecture` 或设置为：

```yaml
architecture: mlp
```

时维持现有行为和版本：

```text
curvature_stage2_residual_v1
no_curvature_stage2_residual_v1
```

## 9.2 MPNN

完整曲率版本：

```text
curvature_stage2_mpnn_v1
```

严格无曲率版本：

```text
no_curvature_stage2_mpnn_v1
```

## 9.3 MPNN 强制校验

要求：

```text
message_hidden_dims == [32]
message_dim == 16
update_hidden_dims == [64, 64]
aggregation == mean_max
condition_message_on_receiver == true
use_sender_node_features == false
use_curvature_edge_feature == false
zero_init_output == true
activation == relu
```

如果：

```yaml
use_stage1_reference: true
```

则必须：

```yaml
detach_stage1_reference: true
```

如果：

```yaml
use_stage1_reference: false
```

则必须：

```yaml
alpha_override: 0.0
```

无曲率模式中以下任意项开启都必须报错：

```text
use_curvature_edge_feature
include_curvature_mean
include_curvature_freshness_interaction
```

## 9.4 Metadata

MPNN Actor metadata 至少记录：

```text
architecture_version
residual_architecture
node_input_feature_names
edge_input_feature_names
node_input_dim
edge_input_dim
message_input_dim
message_dim
aggregation
aggregated_message_dim
decoder_input_dim
use_stage1_reference
effective_alpha
condition_message_on_receiver
use_sender_node_features
use_curvature_edge_feature
```

完整默认维数：

```text
node_input_dim = 5
edge_input_dim = 3
message_input_dim = 8
message_dim = 16
aggregated_message_dim = 32
decoder_input_dim = 38
```

严格无曲率默认：

```text
decoder_input_dim = 37
```

---

# 10. Batch 与 PPO

当前 rollout 每个 slot 对应一个独立图。训练时不能简单拼接 edge receiver index，否则不同 slot 的边会错误连接到第一个图。

修改：

```text
CTDEPPO.actor_batch_inputs()
CTDEPPO._actor_batch_feed()
CTDEPPO._encoded_feed()
```

对 MPNN：

1. 拼接所有 slot 的节点张量；
2. 拼接所有 slot 的边特征；
3. 对每个 slot 的 `edge_index` 加节点累计 offset；
4. 将多个 slot 构造成 disjoint graph batch；
5. 保证不同 slot 之间没有边；
6. 保证 action、old log probability 和 advantage 的节点顺序保持不变。

示意：

```python
node_counts = [
    step.node_context.shape[0]
    for step in encoded_steps
]

offsets = np.cumsum([0] + node_counts[:-1])

batched_edge_index = np.concatenate([
    step.edge_index + offset
    for step, offset in zip(encoded_steps, offsets)
], axis=1)
```

需要正确广播 offset 到 `edge_index` 的两行。

Actor advantage 仍按每个 slot 的所有节点复制该 slot 的 VAoI advantage，保持当前 PPO 定义不变。

不修改：

* reward；
* GAE；
* PPO clip；
* entropy coefficient；
* Critic；
* budget advantage；
* multiplier；
* validation checkpoint 规则。

---

# 11. Trainer 编码路径

在训练循环中：

```python
if actor.stage == 2 and residual.architecture == "mpnn":
    encoded = encode_stage2_mpnn_observations(...)
elif actor.stage == 2:
    encoded = encode_stage2_observations(...)
```

不要把 MPNN 伪装成 legacy `residual_context`。

当前：

```python
actor_vaoi_advantages
```

可以继续根据：

```python
step.curvature_scores.shape[0]
```

确定节点数量。

训练日志增加：

```text
residual_architecture
node_input_dim
edge_input_dim
message_dim
aggregation
aggregated_message_dim
decoder_input_dim
directed_edge_count
mean_directed_degree
valid_edge_fraction
message_l2_mean
message_l2_std
aggregated_message_l2_mean
aggregated_message_l2_std
```

同时继续记录：

```text
q_base_mean
q_base_std
residual_delta_mean
residual_delta_std
q_final_mean
q_final_std
actual tx ratio
mean VAoI
effective alpha
```

不要将整个 edge message tensor 写入 CSV。

---

# 12. Validation

修改：

```text
curvature-gossip/src/curvature_gossip/learning/validation.py
```

固定验证中根据 residual architecture 选择 MLP 或 MPNN encoder。

保持以下行为完全不变：

* 相同 topology seed；
* 相同 source update seed；
* 相同 shadowing/fading seed；
* 相同 Bernoulli random stream；
* fixed random；
* matched-rate random；
* validation 前后变量完全一致检查；
* paired VAoI difference；
* probability statistics。

MPNN 验证不得更新：

* Actor；
* Critic；
* optimizer；
* BatchNorm state；
* 任何缓存参数。

首版不要引入 BatchNorm、Dropout 或 stochastic message masking。

---

# 13. Simulator 与观测

原则上不修改：

```text
curvature-gossip/src/curvature_gossip/simulator/observations.py
curvature-gossip/src/curvature_gossip/simulator/engine.py
```

因为 `NodeObservation` 已经包含：

```text
neighbor_ids
neighbor_cache_estimates
neighbor_estimate_valid
neighbor_estimate_age
own_cache_versions
last_tx_cache_versions
congestion_ewma
consecutive_tx_attempts
```

如确实需要修改 observation 文件，必须解释为什么现有字段不足。

禁止增加：

* 邻居真实 cache；
* 邻居当前隐藏状态；
* 每时隙 embedding 广播；
* 额外 ACK；
* 免费控制消息。

---

# 14. 新增配置

## 14.1 完整曲率 MPNN smoke

新增：

```text
configs/nn_2layers_stage2_mpnn_smoke.yaml
```

使用较小：

```text
N
episodes
rollout slots
validation slots
```

但必须完整执行：

* rollout；
* PPO update；
* fixed validation；
* checkpoint；
* restore；
* diagnostics。

## 14.2 无曲率 MPNN smoke

新增：

```text
configs/nn_2layers_stage2_mpnn_no_curvature_smoke.yaml
```

设置：

```yaml
alpha_override: 0.0
use_stage1_reference: false
architecture: mpnn
```

## 14.3 正式完整曲率 MPNN

新增：

```text
configs/nn_2layers_stage2_mpnn_heuristic_channel.yaml
```

必须从：

```text
configs/nn_2layers_stage2_heuristic_channel.yaml
```

复制环境、训练和验证条件。

只允许修改：

* experiment id；
* residual architecture；
* MPNN architecture fields；
* output root。

保持：

```text
N = 100
u = 0.20
b = 0.15
cross_edge_count_range = [4, 5]
alpha_override = 1.0
episodes = 300
rollout_slots = 400
validation slots = 500
validation scenarios = v00 ... v09
```

建议：

```yaml
output:
  root: result_GNN
```

## 14.4 正式无曲率 MPNN

新增：

```text
configs/nn_2layers_stage2_mpnn_no_curvature_heuristic_channel.yaml
```

必须与正式完整曲率 MPNN 完全相同，唯一区别：

```yaml
alpha_override: 0.0
use_stage1_reference: false
```

## 14.5 配对的无曲率 MLP

当前已有：

```text
configs/nn_2layers_stage2_no_curvature.yaml
```

使用的 `b` 和跨社区边范围与当前正式曲率配置不一致，不能直接作为配对基线。

不要覆盖该历史配置。

新增：

```text
configs/nn_2layers_stage2_no_curvature_mlp_paired.yaml
```

从当前正式完整曲率 MLP 配置复制，只修改：

```yaml
residual:
  architecture: mlp
  use_stage1_reference: false

curvature:
  alpha_override: 0.0
```

这样形成严格的 (2\times2)：

```text
curvature + MLP
no curvature + MLP
curvature + MPNN
no curvature + MPNN
```

---

# 15. 单元测试

新增：

```text
tests/test_stage2_mpnn_features.py
tests/test_stage2_mpnn_actor.py
```

至少覆盖以下测试。

## Test 1：有向边数量

对于无向图：

```text
E undirected edges
```

要求：

```text
E_directed = 2E
```

每个节点的 outgoing local record 和 receiver mapping 正确。

## Test 2：边特征公式

人工构造 own cache、neighbor estimate、validity 和 age。

逐项验证：

```text
valid
freshness gain
confidence
```

与定义一致。

## Test 3：无效邻居

无效估计必须：

```text
valid = 0
freshness gain = 0
confidence = 0
```

所有输出有限。

## Test 4：零初始化

完整曲率 MPNN 初始化后：

```text
delta = 0
q_final = q_base
```

严格无曲率 MPNN：

```text
delta = 0
q_base = b
q_final = b
```

误差：

```text
atol <= 1e-6
```

## Test 5：输入维数

完整默认配置：

```text
node input = 5
edge input = 3
message input = 8
message = 16
aggregate = 32
decoder input = 38
```

无曲率默认：

```text
decoder input = 37
```

## Test 6：邻居排列不变性

只改变同一节点的邻居记录顺序，同时正确排列：

```text
neighbor_ids
neighbor_cache_estimates
neighbor_estimate_valid
neighbor_estimate_age
```

要求该节点：

```text
aggregated message unchanged
q_final unchanged
```

## Test 7：节点置换等变性

对节点行、curvature scores、node context 和 edge index 使用同一置换。

要求：

```text
q_permuted = permutation(q_original)
```

## Test 8：无 sender-state 泄漏

改变节点 (j) 的当前 `node_context`，但保持节点 (i) 的：

```text
node context
local edge features
curvature base
```

不变。

要求节点 (i) 的：

```text
decoder input
q_final
```

均不变。

## Test 9：单条边局部性

只修改 receiver 为节点 (i) 的一条边特征。

要求其他 receiver 的：

```text
aggregated messages
```

不变。

## Test 10：disjoint batch 无跨 slot 泄漏

分别预测两个 encoded steps，再将其通过 `actor_batch_inputs()` 构成 disjoint graph batch。

要求批量预测结果等于逐 step 预测拼接结果。

该测试必须捕获忘记给 edge index 加 offset 的错误。

## Test 11：MPNN 参数可学习

执行一次或多次 PPO update：

* 至少一个 `message_mlp` 参数变化；
* 至少一个 `update_mlp` 参数变化；
* 固定 alpha 的 `alpha_raw` 不变；
* residual delta 不再全为零；
* Critic 可以更新。

## Test 12：无曲率独立性

固定节点和边特征，只改变 curvature scores。

严格无曲率 MPNN 中：

```text
q_base unchanged
aggregated message unchanged
q_final unchanged
```

## Test 13：配置泄漏拒绝

以下组合必须报错：

```yaml
architecture: mpnn
use_stage1_reference: false
alpha_override: 1.0
```

以下也必须报错：

```yaml
use_curvature_edge_feature: true
```

在 v1 中无论完整还是无曲率，都先拒绝。

以下必须报错：

```yaml
use_sender_node_features: true
```

因为当前版本没有邻居隐藏状态的合法分布式来源。

## Test 14：legacy MLP 回归

现有：

```text
tests/test_stage2_actor.py
```

必须全部通过。

旧配置未指定：

```yaml
architecture
```

时必须得到与修改前相同：

* variable names；
* input dimensions；
* initial probabilities；
* checkpoint behavior。

## Test 15：checkpoint

MPNN checkpoint：

1. 保存；
2. 关闭模型；
3. 新建相同架构模型；
4. restore；
5. 对同一 encoded graph 预测。

要求 restore 前保存结果和 restore 后结果完全一致或满足极小数值误差。

---

# 16. Smoke run

执行：

```powershell
cd D:\ZMF\2026Curvature\curvature-gossip

conda run --no-capture-output -n GRL_AoI_cpu37 python -m pytest -q

conda run --no-capture-output -n GRL_AoI_cpu37 python scripts/train_nn_ctde.py --config configs/nn_2layers_stage2_mpnn_smoke.yaml

conda run --no-capture-output -n GRL_AoI_cpu37 python scripts/train_nn_ctde.py --config configs/nn_2layers_stage2_mpnn_no_curvature_smoke.yaml
```

检查：

1. 所有测试通过；
2. 两个 smoke training 完成；
3. fixed validation 完成；
4. checkpoint 保存成功；
5. checkpoint restore 成功；
6. 没有 NaN/Inf；
7. 完整 MPNN 初始 `q_final=q_base`；
8. 无曲率 MPNN 初始 `q_final=b`；
9. `directed_edge_count` 正确；
10. `decoder_input_dim` 分别为 38 和 37；
11. MPNN 参数在 PPO update 后变化；
12. alpha 在固定模式下不变化。

---

# 17. README

更新：

```text
curvature-gossip/README.md
```

新增：

```text
NN_2layers GNN branch: local edge-message MPNN
```

说明：

1. 第一层曲率公式；
2. 五个 node features；
3. 三个 directed edge features；
4. receiver-conditioned message；
5. mean-max aggregation；
6. 最终 logit residual；
7. 不增加 over-the-air communication；
8. 不使用邻居当前 hidden state；
9. 完整和无曲率 MPNN 配置；
10. smoke 和正式命令；
11. MLP checkpoint 与 MPNN residual checkpoint 不兼容；
12. 现有 MLP 配置默认行为保持不变。

不要宣称该模型进行了同步 GNN 通信。

建议名称：

```text
distributed local edge-message MPNN
```

---

# 18. 代码质量要求

1. 不复制大量已有特征公式；
2. 提取小型公共 helper，但不改变 legacy 数值；
3. 所有 shape 和 dtype 显式校验；
4. 所有 edge index 使用 `int32`；
5. 所有模型输入使用 `float32`；
6. 支持空 edge tensor；
7. 不依赖节点 ID 连续；
8. 不依赖边排序；
9. 不使用 Python-side per-edge TensorFlow inference；
10. TensorFlow 图中使用向量化 segment operation；
11. 不引入新依赖；
12. 不修改奖励或环境；
13. 不提交运行结果；
14. 为新架构使用独立 variable scope；
15. 旧 MLP variable scope 保持不变。

---

# 19. 最终交付说明

完成后请输出：

1. 当前分支名；
2. 基准 commit；
3. 最终 commit SHA；
4. 修改文件列表；
5. 新增文件列表；
6. 最终数学公式；
7. node feature 顺序；
8. edge feature 顺序；
9. message、aggregate 和 decoder 的准确维数；
10. batching offset 实现说明；
11. 如何保证严格分布式局部性；
12. 为什么没有增加控制通信；
13. architecture version；
14. 新增配置；
15. 新增测试；
16. `pytest` 结果；
17. 两个 smoke run 结果；
18. checkpoint restore 验证；
19. MLP backward compatibility 验证；
20. 尚未实现的内容。

尚未实现的内容必须明确列出：

* 邻居隐藏 embedding；
* 异步 embedding cache；
* 多跳 message passing；
* attention；
* curvature edge attention；
* 通信开销建模。

不要在本次任务中顺带实现这些后续功能。
