# GNN_Actor：节点条件化 Critic 与节点级 GAE 修改说明

## 0. 任务范围

目标分支：

```text
https://github.com/yamazakifei/Curvature/tree/GNN_Actor
```

本次修改只调整 **训练阶段的 Critic、GAE、Critic 特征和相关日志/测试**。

必须保持以下部分不变：

- Actor 的 Stage 1、Stage 2 MLP、Stage 2 MPNN 结构；
- Actor 的分布式执行约束；
- Actor 的输入白名单；
- Bernoulli 动作采样；
- PPO ratio、clipping 和 entropy 计算；
- VAoI 奖励定义；
- budget advantage 和 multiplier 逻辑；
- 现有验证策略和测试场景；
- Stage 1 curvature base 和 Stage 2 residual 的计算方式。

本次实现应新增可配置的节点条件化 Critic，同时保留旧的 scalar global Critic，便于严格消融和旧配置兼容。

---

# 1. 修改目标

现有 Critic 每个时隙只输出一个全局标量：

\[
V_\phi(s_t),
\]

现有 VAoI advantage 在同一时隙被复制给全部节点：

\[
A_{i,t}=A_t,\qquad \forall i.
\]

修改为节点条件化 Critic：

\[
V_{\phi,i}(t)=V_\phi(x_{i,t}),
\]

其中：

\[
x_{i,t}=\left[g_t,\;\ell_{i,t},\;p_{i,t},\;h_i\right].
\]

各部分含义：

- \(g_t\)：全局动态状态和可选场景上下文；
- \(\ell_{i,t}\)：Actor 可见的节点本地特征；
- \(p_{i,t}\)：仅集中训练阶段可见的节点级真实状态；
- \(h_i\)：节点的静态平均信道质量特征。

Critic 对一个时隙的所有节点输出：

\[
\mathbf V_t=[V_{1,t},\ldots,V_{N,t}]\in\mathbb R^N.
\]

VAoI advantage 改为节点级 GAE：

\[
\delta_{i,t}=r_t+\gamma V_{i,t+1}-V_{i,t},
\]

\[
A_{i,t}^{\mathrm{GAE}}=\delta_{i,t}+\gamma\lambda A_{i,t+1}^{\mathrm{GAE}}.
\]

Critic target 为：

\[
R_{i,t}^{\lambda}=A_{i,t}^{\mathrm{GAE}}+V_{i,t}.
\]

---

# 2. 关键设计原则

## 2.1 Critic 与 Actor 输入解耦

不要让 Critic 简单复用“当前 Actor stage 实际使用的输入”。

原因：

- Stage 1 Actor 只使用曲率分数和目标发送率；
- Stage 2 MLP 和 MPNN 使用不同局部输入；
- Critic 的任务是估计长期全局 VAoI return，不应被 Actor stage 的输入限制；
- Critic 在 CTDE 下允许使用集中训练信息。

无论 Actor 为 Stage 1、Stage 2 MLP 还是 Stage 2 MPNN，节点条件化 Critic 都应使用同一套固定含义的输入。

## 2.2 不共享 Actor 与 Critic 参数

第一版不要共享：

- Actor MLP encoder；
- Stage 2 MPNN message encoder；
- Stage 2 residual hidden layers。

Critic 使用独立 MLP，避免 Critic loss 改变 Actor 表示，保证实验解释清晰。

## 2.3 保持节点置换等变性

禁止加入：

- node ID；
- node one-hot；
- 固定节点索引 embedding；
- community ID 等依赖特定拓扑标号的输入。

若对节点顺序做同一置换，Critic 输出必须按相同置换变化。

---

# 3. 节点级 GAE

## 3.1 张量形状

rollout 长度为 \(T\)，节点数为 \(N\)。

需要维护：

```text
rewards:                 [T]
values:                  [T, N]
bootstrap_values:        [N]
advantages:              [T, N]
returns:                 [T, N]
critic_inputs:           [T, N, D]
```

Actor batch 继续使用 time-major 展平：

```text
t0-node0, t0-node1, ..., t0-nodeN-1,
t1-node0, t1-node1, ..., t1-nodeN-1,
...
```

Critic batch、actions、old log probabilities 和 advantages 必须使用完全相同的展平顺序。

## 3.2 节点级 GAE 函数

新增独立函数，例如：

```python
def _node_gae(
    rewards,
    values,
    bootstrap_values,
    gamma,
    gae_lambda,
):
    """
    rewards:          [T]
    values:           [T, N]
    bootstrap_values: [N]
    returns:
        advantages: [T, N]
        returns:    [T, N]
    """
```

参考逻辑：

```python
rewards = np.asarray(rewards, dtype=np.float32)
values = np.asarray(values, dtype=np.float32)
bootstrap_values = np.asarray(bootstrap_values, dtype=np.float32)

if rewards.ndim != 1:
    raise ValueError(...)
if values.ndim != 2:
    raise ValueError(...)
if values.shape[0] != rewards.shape[0]:
    raise ValueError(...)
if bootstrap_values.shape != (values.shape[1],):
    raise ValueError(...)

advantages = np.zeros_like(values, dtype=np.float32)
accumulator = np.zeros(values.shape[1], dtype=np.float32)

for t in range(values.shape[0] - 1, -1, -1):
    next_value = bootstrap_values if t == values.shape[0] - 1 else values[t + 1]
    delta = rewards[t] + gamma * next_value - values[t]
    accumulator = delta + gamma * gae_lambda * accumulator
    advantages[t] = accumulator

returns = advantages + values
return advantages, returns
```

## 3.3 rollout 末端必须 bootstrap

当前 rollout 长度只是训练截断，不是环境真实 terminal。

因此最后一个时隙不能固定使用：

\[
V_{i,T}=0.
\]

应计算下一决策状态的：

\[
V_{i,T}=V_\phi(x_{i,T}).
\]

推荐实现：

1. 正常完成最后一个 `complete_step()`；
2. 调用一次下一时隙的 `begin_step(T)`，生成下一决策状态；
3. 构造下一时隙 Critic 输入；
4. 计算 `bootstrap_values`；
5. 不采样动作，不调用 `complete_step()`；
6. 该 simulator 在 rollout 后立即丢弃，因此保留 pending state 不影响下一 episode。

如果需要更整洁的实现，也可以增加只用于训练的 next-decision-state helper，但不得改变正常仿真路径和随机数语义。

## 3.4 Advantage 标准化

节点级 VAoI advantage 在整个 rollout 的 `[T*N]` 样本上统一标准化：

```python
flat_vaoi_advantages = advantages.reshape(-1)
normalized_vaoi_advantages = _standardize(flat_vaoi_advantages)
```

禁止：

- 每个节点单独标准化；
- 每个时隙在节点间标准化；
- 每个 community 单独标准化。

预算 advantage 保持现有逻辑：

```python
actor_advantages = normalized_vaoi_advantages + budget_advantages
```

Critic 使用未标准化的 `returns`。

---

# 4. Critic 输入总览

推荐正式版本使用：

- 全局动态特征：4 维；
- 场景上下文：3 维，可配置；
- Actor 可见节点特征：12 维；
- 集中式真实节点特征：6 维；
- 静态信道特征：2 维，可配置。

多场景且启用信道特征时总维度：

\[
4+3+12+6+2=27.
\]

单场景且关闭场景上下文、关闭信道特征时：

\[
4+12+6=22.
\]

模型元数据必须记录实际输入顺序和维度。

---

# 5. 全局动态特征

使用全网非对角 version-age 元素：

\[
\mathcal A_t=\{A_{h,s}(t):h\ne s\}.
\]

定义 4 维动态全局输入：

\[
g_t^{\mathrm{dynamic}}=
\left[
\log(1+\operatorname{mean}\mathcal A_t),
\log(1+\operatorname{std}\mathcal A_t),
\log(1+\operatorname{tail5mean}\mathcal A_t),
\rho_{t-1}
\right].
\]

其中：

- `mean_vaoi`：与即时奖励直接对应；
- `std_vaoi`：描述全网 VAoI 分布离散程度；
- `tail5mean_vaoi`：最差 5% version-age 的平均值；
- `last_tx_ratio`：上一时隙实际发送比例。

## 5.1 删除冗余全局统计

节点条件化 Critic 路径中不要同时使用：

- max VAoI；
- p95 VAoI；
- top-5% mean VAoI。

三者高度相关。正式节点 Critic 保留 `top-5% mean`，删除 `max` 和 `p95`。

旧 scalar Critic 路径保持原有 8 维定义，避免破坏旧 checkpoint 和旧实验。

---

# 6. 场景上下文

可选 3 维：

\[
g^{\mathrm{scenario}}=[\log(1+N),u,b].
\]

分别为：

- 网络节点数；
- source update probability；
- target transmission ratio。

配置：

```yaml
critic:
  include_scenario_context: true
```

要求：

- 多 \(N,u,b\) 联合训练时必须启用；
- 单一固定场景训练时允许关闭；
- 固定场景下启用不属于错误，只是这些维度在单次训练内为常数。

---

# 7. Actor 可见节点特征

保留当前 `encode_observations()` 前 11 维的语义，并增加 1 个版本增量幅度特征，总计 12 维。

固定顺序：

```text
1. normalized_degree
2. time_since_last_tx
3. consecutive_tx_attempts
4. congestion_ewma
5. incident_bottleneck_max
6. incident_bottleneck_mean
7. self_information_increment_fraction
8. self_information_increment_magnitude
9. neighbor_freshness_mean
10. neighbor_freshness_max
11. bottleneck_weighted_neighbor_freshness
12. neighbor_confidence_mean
```

## 7.1 现有 11 维应保留

### normalized_degree

保持现有定义：

\[
\frac{\log(1+d_i)}{\log(1+N)}.
\]

### time_since_last_tx

保持现有与目标发送率相关的归一化方式，例如：

\[
\tanh(b\cdot \mathrm{time\_since\_last\_tx}).
\]

### consecutive_tx_attempts

保持现有归一化。

它不仅表示连续发送行为，还间接表示该节点的拥塞观测有多陈旧，因为半双工发送节点无法可靠测量当时干扰。

### congestion_ewma

保持现有归一化。

### incident_bottleneck_max / mean

两者均保留：

- max：是否连接强瓶颈边；
- mean：节点整体入射边的瓶颈程度。

### self_information_increment_fraction

保持现有定义：

\[
\frac1N\sum_s\mathbf 1[C_{i,s}>C^{\mathrm{lastTX}}_{i,s}].
\]

### neighbor freshness 和 confidence

保留：

- mean；
- max；
- bottleneck-weighted；
- confidence mean。

这些是节点本地可获得的估计，Critic 可以使用，但不能用它们替代集中式真实收益特征。

## 7.2 新增 self-information increment magnitude

新增：

\[
f_{i,t}^{\mathrm{increment\ magnitude}}=
\log\left(
1+\frac1N\sum_s(C_{i,s}-C^{\mathrm{lastTX}}_{i,s})_+
\right).
\]

要求：

- 若节点从未发送，`last_tx_cache_versions` 当前为零，按现有状态定义正常计算；
- 输出必须有限；
- 使用 `float32`；
- 不需要裁剪到 `[0,1]`，但必须使用 `log1p` 控制尺度。

---

# 8. 集中式真实节点特征

这些特征只供 Critic 使用，不得进入 Actor。

定义真实 version-age 矩阵：

\[
A_{h,s}(t)=v_s(t)-C_{h,s}(t).
\]

共 6 维。

## 8.1 节点作为接收端的陈旧程度：2 维

取第 \(i\) 行：

\[
\{A_{i,s}:s\ne i\}.
\]

新增：

\[
p_{i,t}^{\mathrm{receiver\ mean}}=
\log\left(1+\frac1{N-1}\sum_{s\ne i}A_{i,s}\right),
\]

\[
p_{i,t}^{\mathrm{receiver\ tail}}=
\log\left(1+\operatorname{tail5mean}_{s\ne i}A_{i,s}\right).
\]

解释：

- 值越高，节点 \(i\) 自己越需要静默并接收；
- 对应半双工发送的机会成本。

## 8.2 节点自身源信息的全网陈旧程度：2 维

取第 \(i\) 列：

\[
\{A_{h,i}:h\ne i\}.
\]

新增：

\[
p_{i,t}^{\mathrm{source\ mean}}=
\log\left(1+\frac1{N-1}\sum_{h\ne i}A_{h,i}\right),
\]

\[
p_{i,t}^{\mathrm{source\ tail}}=
\log\left(1+\operatorname{tail5mean}_{h\ne i}A_{h,i}\right).
\]

解释：

- 值越高，其他节点越缺少源 \(i\) 的最新信息；
- 表示源 \(i\) 信息向外传播的紧迫度。

## 8.3 精确邻居创新收益：2 维

在决策时刻使用真实缓存矩阵。

节点 \(i\) 当前 packet 为其完整缓存：

\[
C_{i,:}.
\]

对每个物理邻居 \(j\in\mathcal N(i)\)，比较：

\[
(C_{i,s}-C_{j,s})_+.
\]

### 创新条目比例

\[
p_{i,t}^{\mathrm{innovation\ fraction}}=
\frac{1}{d_iN}
\sum_{j\in\mathcal N(i)}\sum_s
\mathbf 1[C_{i,s}>C_{j,s}].
\]

范围应在 `[0,1]`。

### 版本提升幅度

\[
p_{i,t}^{\mathrm{innovation\ magnitude}}=
\log\left(
1+
\frac{1}{d_iN}
\sum_{j\in\mathcal N(i)}\sum_s
(C_{i,s}-C_{j,s})_+
\right).
\]

若 `degree == 0`，两项均返回 0。

注意：

- 这是“若成功接收时的潜在信息收益”；
- 不乘当前动作；
- 不使用当前采样动作；
- 不读取本时隙尚未生成的衰落；
- 不把该真实信息传给 Actor。

---

# 9. 静态信道特征

默认建议启用，允许配置关闭。

使用 `PropagationModel.mean_rx_power_mw`，不使用当前时隙瞬时 Rayleigh fading。

对发送节点 \(i\) 和邻居 \(j\)，定义无干扰平均链路裕量：

\[
m_{ij}=10\log_{10}\frac{\bar P_{i\rightarrow j}}{N_0}-\Gamma_{\mathrm{th,dB}}.
\]

节点级 2 维：

\[
h_i^{\mathrm{mean}}=
\tanh\left(
\frac{\operatorname{mean}_{j\in\mathcal N(i)}m_{ij}}{10}
\right),
\]

\[
h_i^{\mathrm{weak}}=
\tanh\left(
\frac{Q_{0.1,j\in\mathcal N(i)}m_{ij}}{10}
\right).
\]

要求：

- 使用发送方向 `mean_rx_power_mw[i, j]`；
- 若 degree 为 0，返回 0；
- 计算 log 前使用安全下界；
- 输出范围应为 `[-1,1]`。

配置：

```yaml
critic:
  include_channel_features: true
```

---

# 10. 不应加入 Critic 的输入

第一版禁止加入：

```text
node_id
node one-hot
current sampled action a_i,t
current joint action
current actor probability q_i,t
Actor hidden representation
Actor MPNN hidden message
community ID
betweenness centrality
global node ranking
broadcast debt
```

说明：

- 当前 Critic 是 \(V\)，不能依赖当前动作；依赖动作应定义为 \(Q\)；
- 不加入 `q_i,t`，避免 Critic 输入随 Actor 参数变化而产生额外非平稳性；
- 当前 VAoI Critic 只学习 VAoI return，且预算 advantage 独立处理，因此不加入 broadcast debt；
- 不加入 node ID 和全局社区标签，避免记忆固定拓扑。

---

# 11. 推荐数据结构

建议在 `features.py` 中增加：

```python
@dataclass(frozen=True)
class EncodedNodeCriticInputs:
    critic_inputs: np.ndarray  # [N, D]
```

必须同时保存 feature names 和各分组维度，便于 checkpoint metadata 和调试。

建议常量：

```python
NODE_CRITIC_DYNAMIC_GLOBAL_FEATURE_NAMES = (...)
NODE_CRITIC_SCENARIO_FEATURE_NAMES = (...)
NODE_CRITIC_LOCAL_FEATURE_NAMES = (...)
NODE_CRITIC_EXACT_FEATURE_NAMES = (...)
NODE_CRITIC_CHANNEL_FEATURE_NAMES = (...)
```

建议提供：

```python
def node_critic_feature_names(
    include_scenario_context: bool,
    include_channel_features: bool,
) -> Tuple[str, ...]:
    ...
```

---

# 12. Critic 网络结构

旧 scalar Critic 保留原路径。

新增 node-conditioned Critic：

\[
D_{\mathrm{critic}}\rightarrow64\rightarrow64\rightarrow1.
\]

建议配置：

```yaml
critic:
  architecture: node_conditioned
  hidden_dims: [64, 64]
  activation: relu
  include_scenario_context: true
  include_exact_node_features: true
  include_channel_features: true
  advantage_mode: node_gae
  bootstrap_rollout_end: true
```

TensorFlow 1.x 参考：

```python
self.critic_inputs = tf.placeholder(
    tf.float32,
    [None, critic_input_dim],
    name="critic_inputs",
)

with tf.variable_scope("critic"):
    hidden = tf.layers.dense(
        self.critic_inputs,
        64,
        activation=tf.nn.relu,
        name="dense_1",
    )
    hidden = tf.layers.dense(
        hidden,
        64,
        activation=tf.nn.relu,
        name="dense_2",
    )
    self.values = tf.squeeze(
        tf.layers.dense(hidden, 1, name="value"),
        axis=1,
    )
```

不需要使用 LayerNorm、BatchNorm 或 dropout。

Critic loss：

\[
\mathcal L_V=
\frac1{TN}\sum_{t,i}
(R_{i,t}^{\lambda}-V_{i,t})^2.
\]

第一版不使用 value clipping。

---

# 13. 推荐代码改动位置

## 13.1 `src/curvature_gossip/learning/features.py`

新增：

- 节点 Critic 特征名和维度常量；
- `EncodedNodeCriticInputs`；
- 12 维本地 Critic 特征编码；
- 4 维全局动态特征编码；
- 6 维集中式真实节点特征编码；
- 2 维信道特征编码；
- 统一拼接和校验逻辑。

旧 `encode_global_state()` 保留给 scalar Critic 和兼容路径。

## 13.2 `src/curvature_gossip/learning/ctde_ppo.py`

修改：

- 读取 `critic` 配置；
- 支持 `scalar_global` 和 `node_conditioned`；
- node-conditioned 路径使用动态 `critic_input_dim`；
- `value()` 接口支持 `[N,D]`；
- `update()` 的 Critic feed 改为节点批次；
- checkpoint metadata 记录 Critic architecture、input dimension、feature names；
- 新增 actor-only restore。

## 13.3 `src/curvature_gossip/learning/trainer.py`

修改：

- 每个 slot 构造 `[N,D]` Critic 输入；
- 每个 slot 存储 `[N]` value；
- rollout 后构造下一决策状态并计算 bootstrap value；
- 使用 `_node_gae()`；
- `[T,N]` advantage 统一展平并标准化；
- Critic returns 使用 `[T,N]` 展平；
- 保持 budget advantage 原逻辑；
- 新增诊断日志。

## 13.4 `src/curvature_gossip/simulator/engine.py`

仅在有必要时增加集中式 Critic helper。

允许 Critic 访问：

- `state.version_age()`；
- `state.cache_versions`；
- topology graph；
- propagation mean receive power；
- channel noise 和 SINR threshold。

禁止将这些字段加入 `NodeObservation`，因为 `NodeObservation` 是 Actor 的分布式观测接口。

## 13.5 配置文件

新增正式配置，例如：

```text
configs/GNN/mpnn_heuristic_channel_node_critic.yaml
```

从当前正式 MPNN 配置复制，只修改：

```yaml
experiment:
  id: mpnnV2_node_critic_...

critic:
  architecture: node_conditioned
  hidden_dims: [64, 64]
  activation: relu
  include_scenario_context: true
  include_exact_node_features: true
  include_channel_features: true
  advantage_mode: node_gae
  bootstrap_rollout_end: true

training:
  restore_actor_only: true
```

保留一个完全相同、仅 Critic 为旧 scalar 的 paired baseline 配置。

---

# 14. Checkpoint 恢复要求

当前兼容恢复逻辑按变量名和 shape 匹配。修改 Critic 输入维度后，其他 Critic 层可能仍被错误地部分恢复。

禁止部分恢复旧 Critic。

新增：

```python
def restore_actor_only(self, checkpoint_prefix):
    ...
```

只恢复：

```text
actor/*
```

不恢复：

```text
critic/*
*/Adam/*
beta1_power
beta2_power
```

要求：

- 新节点 Critic 完整随机初始化；
- Actor 权重可从已有 checkpoint 恢复；
- optimizer state 默认重新初始化；
- 日志记录实际恢复的变量名；
- 若配置 `restore_actor_only=true`，不得隐式恢复旧 Critic。

---

# 15. 训练日志与诊断

新增至少以下字段：

```text
critic_architecture
critic_input_dim
critic_value_mean
critic_value_std
critic_within_slot_value_std_mean
critic_within_slot_value_std_max
vaoi_advantage_within_slot_std_mean
vaoi_advantage_global_std
critic_return_mean
critic_return_std
critic_explained_variance
bootstrap_value_mean
bootstrap_value_std
```

节点 value 差异：

```python
within_slot_value_std = np.std(values, axis=1)
```

节点 advantage 差异：

```python
within_slot_adv_std = np.std(advantages, axis=1)
```

Critic explained variance：

\[
EV=1-
\frac{\operatorname{Var}(R-V)}
{\operatorname{Var}(R)+\epsilon}.
\]

建议额外记录以下输入的 rollout 均值和标准差：

```text
receiver_vaoi_mean
source_vaoi_mean
innovation_fraction
innovation_magnitude
mean_link_margin
weak_link_margin
```

---

# 16. 单元测试

## 16.1 节点级 GAE

文件建议：

```text
tests/test_node_gae.py
```

测试：

1. 输入 shape 校验；
2. 手工构造小数组，验证数值；
3. \(N=1\) 时与标量 GAE 等价；
4. 所有节点 value 相同时，节点 advantage 相同；
5. 不同节点 value 时，同一时隙 advantage 不同；
6. bootstrap 非零时，最后时隙结果正确；
7. `lambda=0` 时退化为 one-step TD；
8. `gamma=0` 时结果正确。

## 16.2 Critic 特征

文件建议：

```text
tests/test_node_critic_features.py
```

测试：

1. 输出维度与配置一致；
2. 所有特征有限；
3. bounded 特征范围正确；
4. 节点置换后，输出按相同置换变化；
5. row VAoI 和 column VAoI 的定义正确；
6. innovation fraction 和 magnitude 使用真实邻居缓存；
7. degree 为 0 的边界情况；
8. `include_scenario_context` 开关；
9. `include_channel_features` 开关；
10. link margin 使用发送方向 `i -> j`。

## 16.3 Critic 网络

文件建议：

```text
tests/test_node_conditioned_critic.py
```

测试：

1. 一个时隙输入 `[N,D]` 输出 `[N]`；
2. batch 输入 `[T*N,D]` 输出 `[T*N]`；
3. node permutation equivariance；
4. Actor stage 1/2 MLP/2 MPNN 都能构造相同 Critic；
5. Critic update 不修改 frozen Actor；
6. Actor inference 不要求 Critic 输入；
7. scalar Critic 旧路径仍可运行；
8. actor-only restore 不恢复任何 critic 变量。

## 16.4 训练 smoke test

小规模配置：

```text
N <= 10
rollout_slots <= 8
episodes <= 2
ppo_epochs = 1
```

验证：

- 可完成一个 update；
- loss 有限；
- `values.shape == [T,N]`；
- `advantages.shape == [T,N]`；
- bootstrap 被实际使用；
- 保存和恢复 checkpoint 成功。

---

# 17. 验收标准

实现完成后必须满足：

1. Actor 的变量名、输入和推理输出不因 Critic 修改而变化；
2. 节点 Critic 每个时隙输出 \(N\) 个 value；
3. 节点级 GAE 使用 `[T,N]` value；
4. rollout 末端使用真实 bootstrap，而非固定 0；
5. Actor VAoI advantage 在 `[T*N]` 上统一标准化；
6. Critic target 不标准化；
7. budget advantage 逻辑保持原样；
8. 新 Critic 能用于 Stage 1、Stage 2 MLP 和 Stage 2 MPNN；
9. 旧 scalar Critic 配置仍可运行；
10. 旧 checkpoint 只能恢复 Actor，不能部分恢复旧 Critic；
11. feature names 和 input dimension 写入模型 metadata；
12. 全部新增测试通过；
13. 原有测试不得因本次修改失败。

---

# 18. 推荐实验对照

至少运行以下 paired comparison。

## A. 旧 Critic

```yaml
critic:
  architecture: scalar_global
```

## B. 节点 Critic，仅本地特征

```yaml
critic:
  architecture: node_conditioned
  include_exact_node_features: false
  include_channel_features: false
```

## C. 节点 Critic + 真实节点状态

```yaml
critic:
  architecture: node_conditioned
  include_exact_node_features: true
  include_channel_features: false
```

## D. 完整节点 Critic

```yaml
critic:
  architecture: node_conditioned
  include_exact_node_features: true
  include_channel_features: true
```

所有实验必须保持一致：

- topology seeds；
- update RNG；
- fading RNG；
- actor initialization；
- training hyperparameters；
- rollout length；
- validation scenarios；
- Stage 1 alpha；
- Actor architecture；
- target transmission ratio。

重点比较：

```text
best validation mean VAoI
delta vs matched-rate random
critic explained variance
critic within-slot value std
advantage within-slot std
VAoI policy-gradient norm
action probability node std
actual transmission ratio
```

---

# 19. 理论边界说明

节点条件化 Critic 使用共享 team reward：

\[
r_t=-\operatorname{meanVAoI}_t.
\]

因此它仍然是节点相关 baseline，而不是严格的反事实 credit assignment。

本次修改的目标是：

- 缓解所有节点共享同一 scalar advantage 的问题；
- 利用节点角色和真实状态降低 policy-gradient 方差；
- 让同一时隙不同节点获得不同的 GAE；
- 在不改变 Actor 分布式执行结构的前提下改善训练。

如果训练后出现：

```text
critic_within_slot_value_std ≈ 0
advantage_within_slot_std ≈ 0
```

且性能没有改善，不应继续无止境增加 \(V_i\) 输入。下一阶段应考虑：

- action-conditioned \(Q_i\)；
- COMA-style counterfactual baseline；
- difference reward；
- 节点广播边际收益的辅助监督。

本次任务不实现这些方法。
