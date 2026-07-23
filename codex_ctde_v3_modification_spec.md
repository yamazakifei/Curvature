# Curvature-Gossip CTDE-PPO V3 修改说明

## 0. 任务目标与修改边界

请基于当前仓库 `repo_nn/curvature-gossip` 实现 CTDE-PPO V3。当前版本是 `node_only_budget_centered_v2`，本次修改的主目标是：

1. 共享 Actor 仍使用节点级 MLP，不引入 GNN 或逐边神经网络；
2. 用节点级曲率、信息增量、邻居 Freshness 增益和估计置信度重构 Actor 的 14 维输入；
3. Critic 从 9 维改为 8 维，只预测 VAoI 回报；
4. 从 Actor 和优势函数中删除广播债务；
5. 将预算优势改为显式有界的线性拉格朗日项；
6. 使用 rollout 级广播率 EMA、相对预算误差、死区和显式上限更新乘子；
7. 当前仍将每个 500 时隙 rollout 当作训练 episode，末状态保持零 Bootstrap；
8. 当前先训练单一固定组合 \((N,u,b)\)，但 Actor/Critic 保留 \(N,u,b\) 上下文接口。

不要在本次修改中实现：

- GNN、Edge MLP 或逐邻居可变长网络输入；
- 多条件联合训练专用的 context encoder、FiLM、PopArt 或条件乘子网络；
- 公平性/债务版本；
- continuing rollout 的末状态 Bootstrap；
- 每个 episode 独立的状态或奖励归一化。

本次修改会使旧 V2 checkpoint 不再兼容。即使 Actor 输入宽度仍是 14，特征语义已经完全改变，而且 Critic 输入从 9 维变为 8 维，因此必须从头训练，并把模型版本更新为 `node_only_freshness_lagrangian_v3`。

---

## 1. 现有代码与本次修改的对应关系

重点检查和修改以下文件：

| 文件 | 当前职责 | 本次修改 |
|---|---|---|
| `src/curvature_gossip/state/local_knowledge.py` | 邻居缓存估计、动作历史、拥塞和债务 | 新增上次发送缓存快照、邻居最近成功解码时隙；债务可为旧指标保留，但不得进入 V3 网络和优势 |
| `src/curvature_gossip/simulator/observations.py` | 构造严格本地的 `NodeObservation` | 暴露计算 \(U_i,g_{ij},c_{ij}\) 所需的只读本地量 |
| `src/curvature_gossip/simulator/engine.py` | 时隙更新、缓存合并、全局 Critic 摘要 | 按正确时序更新新增本地历史；全局基础状态从 6 维改为 5 维 |
| `src/curvature_gossip/learning/features.py` | Actor/Critic 输入编码 | 重写 14 维 Actor 特征；Critic 编码改为 8 维并对 VAoI 与 \(N\) 使用 `log1p` |
| `src/curvature_gossip/learning/ctde_ppo.py` | 共享 Actor 和集中 Critic | 更新预算索引与 Critic 输入宽度；网络隐藏层结构不变 |
| `src/curvature_gossip/learning/trainer.py` | rollout、GAE、PPO、债务/预算优势、乘子更新 | 删除债务优势和 `tanh` 预算优势；实现新乘子更新与日志 |
| `src/curvature_gossip/policies/neural_policy.py` | checkpoint 推理 | 适配新的观测编码参数和 V3 checkpoint |
| `configs/nn_*.yaml` | 训练配置 | 删除债务优势配置，增加置信度与乘子稳定化配置 |
| `tests/test_nn_local_af3.py` | 神经策略、约束优势测试 | 重写为 V3 特征、线性预算优势和初始预算测试 |
| `tests/test_local_knowledge.py` | 本地状态更新测试 | 增加发送快照和解码时间戳测试 |

不要为了清理 V3 而直接删除模拟器中的 `broadcast_debt`、`broadcast_limit` 或相关旧指标，因为现有启发式策略、指标收集器和历史实验可能仍依赖它们。要求是：

> V3 Actor、V3 Critic、V3 优势和 V3 乘子更新均不得读取或使用广播债务。

---

## 2. 时隙因果顺序：必须保持

现有 `GossipSimulator` 的因果顺序是正确的，本次新增状态必须嵌入同一顺序：

1. `begin_step(t)` 生成本时隙源更新；
2. 冻结 `packet_versions` 和 `packet_slots`；
3. 基于本时隙源更新后的本地缓存和上一时隙结束时的本地知识构造观测；
4. Actor 产生 \(q_{i,t}\)，再采样 \(a_{i,t}\)；
5. `complete_step` 计算信道、成功解码和缓存合并；
6. 用本时隙冻结的发送包快照更新“上次发送缓存快照”；
7. 只对成功解码的邻居更新缓存估计和最近解码时隙；
8. 更新动作历史、拥塞 EWMA 和指标；
9. 新增状态从下一时隙 \(t+1\) 起才能出现在观测中。

不允许使用 `complete_step(t)` 后的真实全局缓存反向计算时隙 \(t\) 的 Actor 输入。Freshness 必须只来自 Actor 决策前已经存在的：

- 本节点当前缓存；
- 本节点对邻居缓存的本地估计；
- 本节点本地动作/接收历史；
- 静态邻接和静态 AF3 瓶颈重要度。

---

## 3. `LocalKnowledge` 新增状态

### 3.1 本节点上次发送的缓存快照

在 `LocalKnowledge` 中新增：

```python
self.last_tx_cache_versions = np.zeros((N, N), dtype=np.int64)
self.has_transmitted = np.zeros(N, dtype=bool)
```

新增一个职责单一的方法，例如：

```python
update_transmitted_snapshots(transmitted, packet_snapshot)
```

其中：

- `transmitted.shape == (N,)`；
- `packet_snapshot.shape == (N, N)`；
- 对所有 `transmitted[i] == True` 的节点执行：

  \[
  \mathbf v_i^{last\_tx}\leftarrow
  \mathbf v_i^{packet}(t),
  \qquad
  has\_transmitted_i\leftarrow True.
  \]

这里的“广播”按发送尝试定义，而不是按至少一个邻居成功接收定义。只要 \(a_{i,t}=1\)，该节点本时隙发送的内容就成为它的“上次广播快照”，即使该包最终因碰撞而无人成功解码。

必须保存 `begin_step` 冻结的 `packet_versions[i]`，不能保存接收合并后的 `state.cache_versions[i]`，否则会错误地把本时隙刚接收的信息当成本时隙已经发送过的信息。

### 3.2 邻居最近成功解码时隙

在 `LocalKnowledge` 中新增：

```python
self.neighbor_last_decode_slot = np.full((N, N), -1, dtype=np.int64)
```

修改：

```python
update_from_decodes(decoded_senders, packet_snapshot)
```

使其接收当前 `slot`，例如：

```python
update_from_decodes(slot, decoded_senders, packet_snapshot)
```

当接收端 \(i\) 在时隙 \(t\) 成功解码发送端 \(j\) 时，同时执行：

\[
\widehat{\mathbf v}_{j|i}\leftarrow
\mathbf v_j^{packet}(t),
\]

\[
valid_{ij}\leftarrow True,
\qquad
t_{ij}^{last}\leftarrow t.
\]

碰撞、未解码、非邻居、仅检测到能量等情况都不得更新时间戳。

---

## 4. `NodeObservation` 新增和保留的字段

Actor 仍只接收 `NodeObservation`，不要把 `VersionState`、全局动作计数或全局真实缓存直接传给特征编码器。

建议新增以下只读字段：

```python
last_tx_cache_versions: np.ndarray      # [N]
has_transmitted: bool
neighbor_estimate_age: np.ndarray       # [degree]
```

`ObservationBuilder.build(i,t,...)` 中：

\[
age_{ij}=
\begin{cases}
t-t_{ij}^{last},&valid_{ij}=1,\\
\text{一个无效哨兵值},&valid_{ij}=0.
\end{cases}
\]

建议无效哨兵使用 `-1`，并继续保留 `neighbor_estimate_valid` 作为显式 mask。不要仅依靠一个很大的 age 表示“从未成功解码”。

以下已有字段继续用于 V3：

- `own_cache_versions`；
- `neighbor_ids`；
- `incident_bottleneck_importance`；
- `neighbor_cache_estimates`；
- `neighbor_estimate_valid`；
- `time_since_last_tx`；
- `congestion_ewma`；
- `consecutive_tx_attempts`。

以下已有字段不再进入 V3 Actor：

- `incident_curvatures` 的 min/mean/negative fraction；
- `transmitted_previous_slot`；
- `previous_interference_valid`；
- `broadcast_debt`；
- `broadcast_limit`。

它们可暂时保留在 `NodeObservation` 中以兼容旧策略，但 `features.py` 的 V3 编码不得读取它们。

所有数组仍必须通过 `_readonly_copy` 深拷贝并设为只读。

---

## 5. Actor 14 维输入：精确定义

### 5.1 特征顺序

将 `NODE_FEATURE_DIM` 保持为 14，但把顺序固定为：

| 索引 | 名称 | 定义 |
|---:|---|---|
| 0 | `degree_normalized` | \(\log(1+d_i)/\log(1+N)\) |
| 1 | `tx_age_normalized` | \(\tanh(b\tau_i^{tx})\) |
| 2 | `consecutive_tx_normalized` | \(\tanh(n_i^{consecutive}/c_n)\) |
| 3 | `congestion_normalized` | 保持当前的 \(\tanh(congestion\_ewma/5)\) |
| 4 | `bottleneck_max` | \(\max_j b_{ij}\) |
| 5 | `bottleneck_mean` | \(d_i^{-1}\sum_j b_{ij}\) |
| 6 | `self_information_increment` | \(U_i\) |
| 7 | `freshness_gain_mean` | \(G_i^{mean}\) |
| 8 | `freshness_gain_max` | \(G_i^{max}\) |
| 9 | `freshness_gain_bottleneck` | \(G_i^{bottle}\) |
| 10 | `neighbor_confidence_mean` | \(C_i^{nbr}\) |
| 11 | `log_network_size` | \(\log(1+N)\) |
| 12 | `update_probability` | \(u\) |
| 13 | `target_tx_ratio` | \(b\) |

同步修改常量：

```python
TARGET_TX_RATIO_INDEX = 13
UPDATE_PROBABILITY_INDEX = 12
NETWORK_SIZE_INDEX = 11
```

`N` 可以从 `observation.own_cache_versions.size` 获得；同一批观测必须具有相同的 \(N\)。不要使用 `len(observations)` 推断 \(N\)，因为部署时 `NeuralCTDEPolicy.transmission_probability()` 会单独编码一个节点。

### 5.2 节点度

\[
\widetilde d_i=
\frac{\log(1+d_i)}{\log(1+N)}.
\]

其中 \(d_i=len(neighbor\_ids)\)。由于 \(0\le d_i\le N-1\)，该值位于 \([0,1]\)。

### 5.3 距离上次发送的时间

\[
\widetilde\tau_i^{tx}
=\tanh(b\tau_i^{tx}).
\]

使用 \(b\tau\) 而不是 `log1p(tau)/5`。原因是预算为 \(b\) 时，均匀随机发送的典型间隔约为 \(1/b\)，所以 \(b\tau\) 表示已经沉默了多少个“预算允许的典型发送周期”。

保留现有 `time_since_last_tx` 的定义：

- 从未发送时，在时隙 \(t\) 取 \(t+1\)；
- 已发送时取 \(t-last\_tx\_slot_i\)。

### 5.4 连续发送次数

\[
\widetilde n_i^{consecutive}
=\tanh\left(
\frac{n_i^{consecutive}}{c_n}
\right).
\]

新增配置：

```yaml
training:
  consecutive_tx_scale: 3.0
```

要求 `consecutive_tx_scale > 0`。该状态沿用当前逻辑：本时隙实际采样到发送动作就加一，静默则归零。

### 5.5 拥塞

现有 `LocalKnowledge` 已先把静默时测得的干扰噪声比处理为：

\[
m_i=\log(1+\max((I_i-N_0)/N_0,0)),
\]

再做 EWMA。V3 保留现有 Actor 压缩：

\[
\widetilde c_i=\tanh(congestion\_ewma_i/5).
\]

不要重新对 `congestion_ewma` 做一次 `log1p`。

### 5.6 AF3 瓶颈重要度

继续使用现有 `bottleneck_importance(..., "local_degree_bound")`：

\[
AF3(i,j)=4-d_i-d_j+3|\mathcal N_i\cap\mathcal N_j|,
\]

\[
b_{ij}
=
\frac{\max(-AF3(i,j),0)}
{\max(1,d_i+d_j-4)}
\in[0,1].
\]

Actor 不再输入原始 AF3 的 min、mean 或负边比例，而使用：

\[
B_i^{max}=\max_{j\in\mathcal N_i}b_{ij},
\]

\[
B_i^{mean}
=\frac1{d_i}\sum_{j\in\mathcal N_i}b_{ij}.
\]

无邻居时两者都定义为 0；虽然当前拓扑通常要求正的最小度数，编码器仍必须安全处理空邻居。

### 5.7 本节点自上次发送后的信息增量 \(U_i\)

定义：

\[
U_i(t)
=
\frac1N
\sum_{s=1}^{N}
\mathbf 1
\left[
v_i^s(t)>v_{i,last\_tx}^s
\right].
\]

这里统计的是“当前发送包中有多少个源的缓存条目比本节点上次实际发送的包更新”，不是版本差值之和，因此天然位于 \([0,1]\)。

具体边界：

- 若节点从未发送，`last_tx_cache_versions` 初始为全零，按同一公式计算；
- 因此初始全零缓存时 \(U_i=0\)；
- 一旦本节点产生或接收到新版本，对应条目计为 1；
- 同一源无论领先 1 个还是多个版本，本版都只计为一个新条目；
- 本节点每次尝试发送后，下一时隙以本次冻结包快照作为新基准。

不要用邻居成功接收数决定是否重置 \(U_i\)。

### 5.8 逐邻居 Freshness 增益 \(g_{ij}\)

对节点 \(i\) 的每个物理邻居 \(j\)，令：

- \(v_i^s\)：节点 \(i\) 当前缓存中的源 \(s\) 版本；
- \(\hat v_{j|i}^s\)：节点 \(i\) 最近一次成功解码 \(j\) 时获得的“节点 \(j\) 缓存快照估计”。

逐源版本领先量为：

\[
\Delta_{ij}^s=
\max(v_i^s-\hat v_{j|i}^s,0).
\]

V3 不使用领先量的数值大小，只统计可更新条目比例：

\[
g_{ij}
=
\frac1N
\sum_{s=1}^{N}
\mathbf 1(\Delta_{ij}^s>0)
=
\frac1N
\sum_s
\mathbf 1(v_i^s>\hat v_{j|i}^s).
\]

对从未成功解码过的邻居，缓存数组虽然为零，但该估计必须由 `neighbor_estimate_valid=False` 屏蔽，不能把未知邻居误当作“缓存全为零”。

### 5.9 邻居估计置信度 \(c_{ij}\)

新增配置：

```yaml
training:
  neighbor_confidence_time_constant: 20.0
```

要求该值 \(T_c>0\)。定义：

\[
c_{ij}(t)=
\begin{cases}
\exp[-(t-t_{ij}^{last})/T_c],&
valid_{ij}=1,\\
0,&valid_{ij}=0.
\end{cases}
\]

刚在上一时隙成功解码的估计，在下一时隙 age 为 1，对应置信度为 \(\exp(-1/T_c)\)。不要人为改成 age 0；因为时隙 \(t\) 的接收结果只能供 \(t+1\) 决策使用。

### 5.10 三个节点级 Freshness 聚合

先计算每个邻居的 \(g_{ij}\)、\(c_{ij}\) 和 \(b_{ij}\)，再计算：

\[
G_i^{mean}
=
\begin{cases}
\dfrac{\sum_j c_{ij}g_{ij}}
{\sum_j c_{ij}},&\sum_jc_{ij}>0,\\
0,&\text{其他},
\end{cases}
\]

\[
G_i^{max}
=
\begin{cases}
\max_{j:valid_{ij}=1}g_{ij},&
\exists j:valid_{ij}=1,\\
0,&\text{其他},
\end{cases}
\]

\[
G_i^{bottle}
=
\begin{cases}
\dfrac{\sum_j c_{ij}b_{ij}g_{ij}}
{\sum_j c_{ij}b_{ij}},&
\sum_jc_{ij}b_{ij}>0,\\
0,&\text{其他}.
\end{cases}
\]

建议用显式分支处理零分母，不要仅在分母添加 epsilon；显式分支能保证定义清晰且结果严格在 \([0,1]\)。

注意：按照当前冻结方案，\(G_i^{max}\) 只用 valid mask，不乘置信度；置信度衰减已经通过 \(G_i^{mean}\)、\(G_i^{bottle}\) 和独立的 \(C_i^{nbr}\) 提供。如果后续实验发现陈旧估计导致 `max` 过于激进，再单独消融 `max_j(c_ij g_ij)`，本次不要擅自改变定义。

### 5.11 节点总体置信度

\[
C_i^{nbr}
=
\begin{cases}
\dfrac1{d_i}\sum_{j\in\mathcal N_i}c_{ij},&d_i>0,\\
0,&d_i=0.
\end{cases}
\]

分母是全部物理邻居数，不是有效邻居数。未知邻居以 0 进入平均值，因此该特征同时表达覆盖率和估计新鲜程度。

### 5.12 场景上下文

Actor 最后三维固定为：

\[
[\log(1+N),u,b].
\]

当前单场景阶段它们是常数，不会提供同一训练内的状态区分能力；保留它们是为了后续多条件训练时不改变网络宽度和 checkpoint 接口。

### 5.13 数值和形状检查

`encode_observations` 返回：

```python
node_features.shape == (number_of_observations, 14)
node_features.dtype == np.float32
```

编码结束前应检查：

- 所有值有限，不允许 `NaN/Inf`；
- 索引 0–10、12、13 应在数值误差允许下位于 \([0,1]\)；
- 索引 11 为非负 `log1p(N)`；
- 所有邻居对齐数组的第一维必须等于 `len(neighbor_ids)`；
- `own_cache_versions`、`last_tx_cache_versions` 以及 `neighbor_cache_estimates` 的源维必须一致为 \(N\)。

---

## 6. Actor 网络保持不变

网络仍为：

\[
\mathbb R^{14}
\rightarrow64
\rightarrow64
\rightarrow\Delta_\theta.
\]

输出为：

\[
q_i
=
\sigma\left[
\operatorname{logit}(b)+\Delta_\theta(o_i)
\right].
\]

`transmission_residual_logit` 的 kernel 和 bias 必须继续零初始化，因此未训练模型对任意输入均满足：

\[
q_i=b.
\]

只需要把 `ctde_ppo.py` 中读取预算的索引从 8 改为 13。不要添加固定 `q_max`，最终概率只做数值稳定所需的 \([10^{-6},1-10^{-6}]\) 裁剪。

---

## 7. Critic 输入从 9 维改为 8 维

### 7.1 模拟器基础全局状态

`GossipSimulator.centralized_state(last_tx_ratio)` 当前返回 6 维，其中最后一维是平均广播债务。本次改为只返回 5 维原始统计：

\[
\mathbf s_t^{base}
=
[
\overline z_t,\,
z_t^{max},\,
z_t^{p95},\,
z_t^{tail},\,
C_{t-1}
].
\]

其中：

- 前四项继续按当前 `version_age` 计算；
- `tail` 继续取全部非对角 VAoI 中最大 5% 的均值；
- \(C_{t-1}\) 是上一时隙实际发送节点比例；
- 时隙 0 使用 0。

修改：

```python
BASE_GLOBAL_STATE_DIM = 5
GLOBAL_STATE_DIM = 8
```

### 7.2 Critic 编码

`encode_global_state` 输出顺序必须为：

\[
\mathbf s_t^{critic}
=
[
\log(1+\overline z_t),\,
\log(1+z_t^{max}),\,
\log(1+z_t^{p95}),\,
\log(1+z_t^{tail}),\,
C_{t-1},\,
\log(1+N),\,
u,\,
b
].
\]

即：

- 只对四个 VAoI 统计量和 \(N\) 使用 `np.log1p`；
- 不对发送比例、\(u\)、\(b\) 使用 `log1p`；
- 不除以 5；
- 不使用 \(z_{ref}\)、z-score、running mean/std 或 episode 内归一化；
- 对四个 VAoI 输入先验证非负；
- 将 \(C_{t-1},u,b\) 验证或裁剪在 \([0,1]\)。

Critic MLP 保持：

\[
8\rightarrow64\rightarrow64\rightarrow V_\phi.
\]

Critic 仍只在训练阶段使用。

---

## 8. VAoI 奖励、GAE 和末状态

即时奖励保持：

\[
r_t=-\overline{\mathrm{VAoI}}_t.
\]

训练中使用：

\[
\widetilde r_t=0.1r_t.
\]

配置 `vaoi_reward_scale` 默认和正式单场景配置都设为 `0.1`。不要做 episode 内奖励标准化。

现有 `_gae` 暂时保留零末状态：

\[
V(s_{T+1})=0.
\]

因此：

\[
\delta_t
=
\widetilde r_t+
\gamma V_{\phi_{old}}(s_{t+1})
-V_{\phi_{old}}(s_t),
\]

最后一个时隙使用 next value 0。模型在整个 rollout 结束后才更新，所以 rollout 中保存的 `value_steps` 都来自同一个旧 Critic，满足 PPO/GAE 的旧值函数要求。

对 VAoI GAE 只在整个 rollout 的时间维上标准化一次：

\[
A_t^{VAoI,norm}
=
\frac{\widehat A_t^{VAoI}-\mu_A}
{\sigma_A+\epsilon}.
\]

然后把同一时隙的该全局优势复制给该时隙全部 \(N\) 个节点。

不要把预算项加入 Critic reward 或 return。

---

## 9. 删除债务优势并改写预算优势

删除当前 `_node_constraint_advantages` 中：

- `broadcast_debts`；
- `local_debt_penalty`；
- `budget_weight`；
- `debt_weight`；
- `constraint_advantage_scale`；
- 预算项和债务项内部的 `tanh`。

改为一个简单函数，例如：

```python
def _budget_advantages(actions, old_probabilities, multiplier, multiplier_max):
    ...
```

精确定义：

\[
\bar\mu_k
=
\operatorname{clip}(\mu_k,0,\mu_{max}),
\]

\[
A_{i,t}^{budget}
=
-\bar\mu_k
\left(
a_{i,t}-q_{i,t}^{old}
\right).
\]

要求：

- `actions` 和 `old_probabilities` 形状完全相同；
- `actions` 必须为 0/1；
- `old_probabilities` 必须在 \([0,1]\)；
- `multiplier_max > 0`；
- 返回 `float32`；
- 不额外乘固定的 0.15。

因为 \(a-q^{old}\in[-1,1]\)，所以：

\[
|A_{i,t}^{budget}|\le\mu_{max}.
\]

Actor 最终优势为：

\[
A_{i,t}
=
A_t^{VAoI,norm}
+
A_{i,t}^{budget}.
\]

不再包含任何债务项。

---

## 10. 拉格朗日乘子更新

### 10.1 更新频率

每个完整 rollout/episode 只更新一次乘子，不能按时隙快速更新。预算优势使用 rollout 开始时冻结的 \(\mu_k\)，rollout 结束后才得到 \(\mu_{k+1}\)。

### 10.2 广播率 EMA

令该 rollout 的实际平均发送比例为：

\[
\overline C_k
=
\frac1T\sum_{t=1}^{T}C_t.
\]

维护：

\[
\widehat C_k
=
\beta\widehat C_{k-1}
+
(1-\beta)\overline C_k.
\]

这里 \(\beta\) 是“旧 EMA 的权重”。配置名建议直接写成：

```yaml
training:
  tx_ratio_ema_beta: 0.9
```

初始化建议：

\[
\widehat C_{-1}=b,
\]

这样第一轮之前不会人为制造预算偏差。

### 10.3 相对预算误差、死区与截断

\[
e_k
=
\frac{\widehat C_k-b}{b+\epsilon}.
\]

配置：

```yaml
training:
  budget_relative_tolerance: 0.05
  multiplier_error_clip: 1.0
```

定义：

\[
\widetilde e_k=
\begin{cases}
0,&|e_k|\le\delta_b,\\
\operatorname{clip}(e_k,-e_{max},e_{max}),&\text{其他}.
\end{cases}
\]

这里的容忍度是相对误差。例如 `0.05` 表示广播率 EMA 在预算上下 5% 内不更新乘子。

### 10.4 显式有界更新

\[
\mu_{k+1}
=
\operatorname{clip}
\left[
\mu_k+\eta_\mu\widetilde e_k,\,
0,\,
\mu_{max}
\right].
\]

建议配置初值：

```yaml
training:
  initial_multiplier: 0.0
  multiplier_learning_rate: 0.01
  multiplier_max: 0.15
```

这些是第一轮实验起点，不是最终理论常数。由于改用了相对误差，不能直接沿用当前 `lagrange_learning_rate: 0.5` 或 `1.0`，否则乘子很可能一轮就触顶。

代码中统一使用 `multiplier` 或数学符号 \(\mu\)，不要继续混用 `lagrange`、`lambda` 和 `mu` 三套名称。

当前单场景配置只允许一个 \(N\)、一个 \(u\)、一个 \(b\)。可以保留现有训练 case 接口，但正式 V3 单场景配置必须各只有一个值。若继续保留多 case 代码路径，乘子和 EMA 至少要按完整 case key `(N,u,b)` 隔离，不能只按 `target` 隔离；否则不同负载和规模会污染同一个乘子状态。

---

## 11. 训练日志

每个 episode 至少保存以下字段：

- `episode`；
- `mean_VAoI`；
- `avg_tx_ratio`；
- `tx_ratio_ema`；
- `target_tx_ratio`；
- `budget_error`：\(\widehat C_k-b\)；
- `budget_relative_error`：\(e_k\)；
- `budget_error_after_deadzone`：\(\widetilde e_k\)；
- `mean_action_probability`；
- `multiplier_used`：本 rollout 预算优势使用的 \(\mu_k\)；
- `multiplier_next`：rollout 后更新得到的 \(\mu_{k+1}\)；
- `multiplier_at_max`：是否达到上限；
- `actor_loss`、`critic_loss`、`entropy`；
- `mean_vaoi_advantage`、`std_vaoi_advantage`、`max_abs_vaoi_advantage`；
- `mean_budget_advantage`、`std_budget_advantage`、`max_abs_budget_advantage`；
- `vaoi_policy_gradient_norm`；
- `budget_policy_gradient_norm`；
- `budget_to_vaoi_gradient_ratio`；
- `n_nodes`、`update_probability`。

策略梯度范数建议这样实现：

1. 在 `ctde_ppo.py` 中基于不含 entropy 的 PPO surrogate actor loss 创建一次 `tf.gradients` 和 `tf.global_norm`；
2. 每轮 PPO 更新前，用同一个旧策略 batch 分别 feed：
   - `actor_vaoi_advantages`；
   - `budget_advantages`；
3. 得到两个更新前梯度范数；
4. 最终训练仍 feed 两项之和；
5. 比率定义为：

   \[
   \frac{\|g_{budget}\|}
   {\|g_{VAoI}\|+\epsilon}.
   \]

不要通过分别执行两个 optimizer step 来测量贡献；只评估梯度，不更新参数。

删除 V3 历史中的：

- `mean_broadcast_debt`；
- `mean_abs_debt_advantage`；
- 旧 `lagrange` 字段。

旧的 `max_node_activity_ratio` 和 `node_cap_violation_fraction` 可以作为诊断指标继续记录，但论文中不能把它们解释为 V3 训练显式优化的公平性约束。

---

## 12. 配置文件

新建独立配置，例如：

```text
configs/nn_ctde_v3_single.yaml
configs/nn_ctde_v3_smoke.yaml
```

不要覆盖旧 V2 配置和旧结果目录。建议核心配置为：

```yaml
experiment:
  id: nn_ctde_v3_n100_u0.05_b0.10

constraints:
  target_tx_ratio: 0.10
  per_node_cap_multiplier: 1.5

training:
  episodes: 300
  rollout_slots: 500

  target_tx_ratios: [0.10]
  node_counts: [100]
  update_probabilities: [0.05]

  learning_rate: 0.0003
  clip_ratio: 0.2
  entropy_coefficient: 0.01
  gamma: 0.99
  gae_lambda: 0.95
  ppo_epochs: 8

  vaoi_reward_scale: 0.1
  consecutive_tx_scale: 3.0
  neighbor_confidence_time_constant: 20.0

  initial_multiplier: 0.0
  multiplier_learning_rate: 0.01
  multiplier_max: 0.15
  tx_ratio_ema_beta: 0.9
  budget_relative_tolerance: 0.05
  multiplier_error_clip: 1.0
```

删除 V3 配置中的：

```yaml
initial_lagrange
lagrange_learning_rate
local_debt_penalty
budget_advantage_weight
debt_advantage_weight
constraint_advantage_scale
```

在代码中对全部新增参数做范围校验。

---

## 13. 测试要求

### 13.1 本地知识测试

新增测试：

1. 从未发送时 `has_transmitted=False` 且上次发送快照为全零；
2. 节点发送后保存的是本时隙冻结包快照；
3. 未发送节点的快照不变；
4. 碰撞导致无人解码时，发送节点自己的快照仍更新；
5. `neighbor_last_decode_slot` 只在成功解码时更新；
6. 未成功解码、非接收端和非邻居均不更新；
7. 下一时隙观测 age 等于 1，保证无同一时隙信息泄漏。

### 13.2 Actor 特征测试

构造小型确定性拓扑和人工缓存，逐项断言 14 维特征的精确值：

- 度归一化；
- \(b\tau\) 时间缩放；
- 连续发送缩放；
- 瓶颈 max/mean；
- \(U_i\)；
- 每条边的 \(g_{ij}\)；
- \(G^{mean}\)、\(G^{max}\)、\(G^{bottle}\)；
- \(C^{nbr}\)；
- \(\log(1+N),u,b\) 的位置。

必须覆盖：

- 所有邻居估计未知；
- 一个有效、一个无效邻居；
- 有效估计随 age 增长而衰减；
- 所有 \(b_{ij}=0\)；
- 节点从未发送；
- 邻居顺序重排后编码完全不变；
- 编码值全部有限。

### 13.3 Actor 输出测试

保留并更新：

- 零初始化 Actor 对任意输入输出 \(q_i=b\)；
- 修改 residual bias 后概率可高于或低于 \(b\)；
- 不存在固定 `q_max`；
- `TARGET_TX_RATIO_INDEX == 13`。

### 13.4 Critic 编码测试

给定人工 5 维基础状态，断言输出严格为：

```python
[
    np.log1p(mean),
    np.log1p(maximum),
    np.log1p(p95),
    np.log1p(tail),
    last_tx_ratio,
    np.log1p(n_nodes),
    update_probability,
    target_tx_ratio,
]
```

并断言形状为 `(8,)`、dtype 为 `float32`。

### 13.5 预算优势测试

精确断言：

\[
A^{budget}=-clip(\mu,0,\mu_{max})(a-q).
\]

覆盖：

- \(a=1\) 时为负；
- \(a=0\) 时为正；
- \(\mu=0\) 时全零；
- \(\mu>\mu_{max}\) 时按上限裁剪；
- 最大绝对值不超过 \(\mu_{max}\)；
- 不再返回 debt advantage。

### 13.6 乘子更新测试

把 EMA 和乘子更新拆成纯函数并测试：

- 广播率等于预算时不更新；
- 相对误差位于死区内时不更新；
- 超预算时上升；
- 低于预算时下降但不小于 0；
- 大误差按 `multiplier_error_clip` 截断；
- 乘子不超过 `multiplier_max`；
- 不同 case 的 EMA 和乘子状态互不污染。

### 13.7 集成测试

运行 V3 smoke 配置至少 2 个短 episode，断言：

- rollout、PPO 更新、保存 checkpoint 全流程完成；
- Actor batch 为 `[T*N,14]`；
- Critic batch 为 `[T,8]`；
- 历史文件包含所有 V3 字段且没有债务优势字段；
- 所有 loss、优势统计、梯度范数、EMA 和乘子均为有限值；
- 保存后的 V3 checkpoint 可由 `NeuralCTDEPolicy` 恢复并推理；
- 旧 V2 checkpoint 不作为兼容目标。

---

## 14. 验证命令

在仓库根目录执行：

```bash
pytest -q
python scripts/train_nn_ctde.py --config configs/nn_ctde_v3_smoke.yaml
```

如已有针对神经模块的子集测试，也先单独运行：

```bash
pytest -q tests/test_local_knowledge.py tests/test_nn_local_af3.py
```

不要用修改测试期望值的方式掩盖实现错误。特别检查：

- Freshness 是否错误读取全局真实邻居缓存；
- 上次发送快照是否错误保存了接收合并后的缓存；
- 未知邻居是否被当作缓存全零；
- `G^{bottle}` 零分母是否产生 NaN；
- Critic 上下文顺序是否仍残留旧版 `[b,u,N]`；
- Actor 是否仍从旧索引 8 读取预算；
- 预算优势是否仍残留固定 `budget_advantage_weight` 或内部 `tanh`；
- 乘子是否使用当前 rollout 更新后的值反向计算同一 rollout 的优势。

---

## 15. 完成标准

实现完成后请提供：

1. 修改文件列表；
2. 14 维 Actor 输入的最终索引表；
3. 8 维 Critic 输入的最终索引表；
4. 新增本地状态的逐时隙更新顺序；
5. 预算优势与乘子更新的最终公式；
6. 配置参数及默认值；
7. 测试命令和结果；
8. 明确说明旧 V2 checkpoint 不兼容；
9. 不要启动 300 episode 正式训练，只运行单元测试和 smoke test。

