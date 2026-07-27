# Curvature-Gossip 两层 Actor 分阶段修改规格

> 目标仓库：`yamazakifei/Curvature`  
> 基线分支：`NN`  
> 建议开发分支：`NN_2layers`  
> 主要代码目录：`curvature-gossip/`

## 0. 文档目的

本文件用于指导 Codex 将当前 V3 单层 Actor 改造成可分阶段训练、可消融验证的两层分布式 Actor。

整体研究问题为：

1. 只依靠静态局部曲率先验，能否学习出优于均匀广播的节点基础概率？
2. 在静态曲率优先级之上，本地动态信息、无线干扰状态和非曲率结构信息能否进一步改善 VAoI？
3. 第二层是否需要显式读取第一层给出的基础概率？

必须按阶段实现，不要直接用一个通用 MLP 混合所有特征。核心结构为：

\[
q_i^{(1)}
=
\sigma\!\left[
\operatorname{logit}(b)
+
\alpha_\kappa(s_i^\kappa-c_\kappa)
\right],
\]

\[
\delta_i(t)
=
\delta_{\max}\tanh
g_\theta\!\left(\mathbf x_i^{(2)}(t)\right),
\]

\[
q_i(t)
=
\sigma\!\left[
\operatorname{logit}(b)
+
\alpha_\kappa(s_i^\kappa-c_\kappa)
+
\delta_i(t)
\right].
\]

其中：

- 第一层只负责静态曲率优先级；
- 第二层是本地上下文残差 MLP；
- 每个节点每个时隙只在最终 \(q_i(t)\) 上采样一次动作；
- Actor 在线推断只能使用节点本地可获得的信息；
- Critic 仍可在训练阶段使用全局状态，即 CTDE。

---

## 1. 实施边界与兼容要求

### 1.1 分支和现有代码保护

1. 从 `NN` 创建或切换到 `NN_2layers`。
2. 不要直接修改 `NN` 分支。
3. 修改前检查工作树，保留用户已有的未提交修改。
4. 现有 V3 Actor checkpoint 不要求与新 Actor 兼容。
5. 若集中式 Critic 的结构和 8 维输入不变，应尽量允许复用旧 Critic 权重。
6. 在 checkpoint 中保存 Actor 架构版本、阶段、输入特征名称及顺序，禁止只保存裸 `state_dict` 而无法判断特征语义。

### 1.2 不在本轮引入的内容

本轮主模型不引入：

- 正式广播预算约束；
- broadcast-debt advantage；
- budget advantage；
- 拉格朗日乘子更新；
- 节点独立的 \(\alpha_i\)；
- 根据曲率输出 \(\alpha_i\) 的 MLP；
- 全局在线概率校准；
- 在线全网曲率均值；
- Actor 对真实全局 VAoI、全网缓存或全局发送率的访问。

`target_tx_ratio=b` 在阶段1和阶段2中是基础 logit 的锚点和场景参数，不表示训练过程严格满足平均广播率约束。必须记录实际发送率，公平比较时再使用 matched-rate random。

---

## 2. 当前 V3 中需要保留的事实

### 2.1 现有 14 维本地特征

当前 `learning/features.py` 中的特征顺序为：

| 索引 | 特征 | 当前编码 |
|---:|---|---|
| 0 | 归一化节点度 | \(\log(1+d_i)/\log(1+N)\) |
| 1 | 距离上次广播的时间 | \(\tanh(b\,\tau_i^{tx})\) |
| 2 | 连续广播尝试次数 | \(\tanh(n_i^{con}/s_{con})\) |
| 3 | 拥塞 EWMA | \(\tanh(c_i/5)\) |
| 4 | incident bottleneck max | \(\max_j b_{ij}\) |
| 5 | incident bottleneck mean | \(\operatorname{mean}_j b_{ij}\) |
| 6 | 自身信息增量 | \(U_i\) |
| 7 | 邻居新鲜度需求均值 | \(G_i^{mean}\) |
| 8 | 邻居新鲜度需求最大值 | \(G_i^{max}\) |
| 9 | 曲率加权新鲜度 | \(G_i^{bottle}\) |
| 10 | 邻居估计平均置信度 | \(C_i^{nbr}\) |
| 11 | 网络规模 | \(\log(1+N)\) |
| 12 | 源更新概率 | \(u\) |
| 13 | 目标发送率 | \(b\) |

两层版本需要复用这些特征的已有定义和本地可观测实现，但必须重新分配它们在第一层、第二层和消融模型中的位置。

### 2.2 集中式 Critic

继续沿用现有 8 维全局状态：

\[
\left[
\log(1+\overline{\mathrm{VAoI}}),
\log(1+\mathrm{VAoI}_{max}),
\log(1+\mathrm{VAoI}_{95}),
\log(1+\mathrm{VAoI}_{tail}),
r_{tx}(t-1),
\log(1+N),
u,
b
\right].
\]

Critic 输出：

\[
V_\phi(S_t).
\]

部署时不需要 Critic。Actor 不能读取上述全局状态。

---

## 3. 本地 AF3 曲率与节点分数

### 3.1 边瓶颈重要度

主版本继续使用完全局部且对称的 AF3 瓶颈重要度：

\[
\mathrm{AF3}_{ij}
=
4-d_i-d_j+3|\mathcal N_i\cap\mathcal N_j|,
\]

\[
b_{ij}
=
\frac{\max(-\mathrm{AF3}_{ij},0)}
{\max(1,d_i+d_j-4)}.
\]

要求：

- \(b_{ij}=b_{ji}\)；
- \(b_{ij}\in[0,1]\)；
- 不使用 `global_negative_max`；
- 不需要当前拓扑的全网最大曲率。

### 3.2 节点曲率分数

阶段1使用：

\[
s_i^\kappa
=
\max_{j\in\mathcal N_i}b_{ij}.
\]

若节点没有邻居，定义：

\[
s_i^\kappa=0.
\]

`incident bottleneck mean` 不进入阶段1主公式，只作为后续增强消融的候选输入。

### 3.3 固定中心 \(c_\kappa\)

使用：

\[
\widetilde s_i^\kappa=s_i^\kappa-c_\kappa.
\]

\(c_\kappa\)必须是训练前由训练拓扑统计得到的固定常数，例如所有训练拓扑、所有有效节点的 \(s_i^\kappa\) 均值。

要求：

1. 训练开始前计算并冻结；
2. 写入实验配置或 checkpoint；
3. 评估和部署时直接加载；
4. 不允许在每个时隙或每个测试拓扑上重新计算全网均值；
5. 不引入运行时全局通信。

若只在单个固定训练拓扑上训练，可以由该训练拓扑离线统计 \(c_\kappa\)，但仍应显式保存数值。

---

## 4. 阶段0：基线与代码通路

阶段0不训练新 Actor，只保证下列策略能够通过统一配置运行和评估：

1. `uniform_random`：所有节点使用同一发送概率；
2. `matched_rate_random`：发送概率匹配待比较策略的实际平均发送率；
3. `curvature_fixed_alpha`：固定 \(\alpha_\kappa\)，关闭第二层；
4. `alpha_override=0`：严格退化为 \(q_i=b\)。

matched-rate 比较必须同时报告：

- 目标 \(b\)；
- 平均概率；
- 实际广播尝试率；
- 平均 VAoI；
- 拓扑数和随机种子。

不要把“使用相同配置中的 \(b\)”自动称为 matched-rate；只有实际平均发送率匹配后才能这样命名。

---

## 5. 阶段1：无 MLP 的单参数曲率 Actor

### 5.1 策略形式

\[
\ell_i^{(1)}
=
\operatorname{logit}(b)
+
\alpha_\kappa(s_i^\kappa-c_\kappa),
\]

\[
q_i^{(1)}=\sigma(\ell_i^{(1)}).
\]

其中：

\[
\operatorname{logit}(b)
=
\log\frac{b}{1-b}.
\]

对数值实现，先将 \(b\) 限制到安全区间，例如：

\[
b_{\mathrm{safe}}
=
\operatorname{clip}(b,\epsilon,1-\epsilon).
\]

### 5.2 唯一 Actor 训练参数

使用原始标量 \(a_\kappa\)，并参数化为：

\[
\alpha_\kappa=\operatorname{softplus}(a_\kappa)\ge0.
\]

阶段1的 Actor 是一个无隐藏层、所有节点共享的单参数可微策略模块，不是用于预测 \(\alpha_\kappa\) 的神经网络。

每个节点唯一随节点变化的输入是 \(s_i^\kappa\)：

\[
s_i^\kappa
\rightarrow
s_i^\kappa-c_\kappa
\rightarrow
\alpha_\kappa(s_i^\kappa-c_\kappa)
\rightarrow
+\operatorname{logit}(b)
\rightarrow
\sigma
\rightarrow
q_i^{(1)}.
\]

明确禁止：

- 节点独立的 \(\alpha_i\)；
- `Dense/MLP(s_i^\kappa)`；
- 通用 64–64 曲率网络；
- 曲率与动态状态在阶段1混合；
- 可学习的节点级 bias。

### 5.3 `alpha=0` 的严格退化测试

有限 \(a_\kappa\) 经过 softplus 后不能严格等于 0。因此：

- 正常训练使用 \(\alpha_\kappa=\operatorname{softplus}(a_\kappa)\)；
- 严格退化测试使用 `alpha_override=0.0`；
- 不要通过设置某个有限的 `alpha_raw` 假装得到精确 0。

在 `alpha_override=0.0` 时必须逐节点满足：

\[
q_i^{(1)}=b,
\]

允许的误差只来自浮点数计算。

### 5.4 阶段1的 CTDE 训练

| 模块 | 输入 | 输出 | 是否学习 |
|---|---|---|---:|
| AF3 计算 | 本地邻居和局部结构 | \(b_{ij}\) | 否 |
| 节点聚合 | incident \(b_{ij}\) | \(s_i^\kappa\) | 否 |
| 曲率 Actor | \(s_i^\kappa,b,c_\kappa\) | \(q_i^{(1)}\) | 只学习共享 \(a_\kappa\) |
| 集中式 Critic | 全局状态 \(S_t\) | \(V_\phi(S_t)\) | 是 |
| 第二层 MLP | 不存在 | 不存在 | 否 |

奖励继续使用：

\[
r_t=-\overline{\mathrm{VAoI}}(t).
\]

Critic 只学习 VAoI 回报。阶段1默认：

```yaml
training:
  use_budget_advantage: false
  update_multiplier: false
  use_debt_advantage: false
  entropy_coefficient: 0.0
```

PPO 的 Bernoulli ratio 按节点和时隙计算，再对有效的 `time × node` 样本取平均；所有节点可以共享同一全局 GAE advantage，但不要先把所有节点 log-prob 求和后构造随 \(N\) 指数缩放的 joint ratio。

### 5.5 阶段1验收含义

\[
\alpha_\kappa\approx0
\]

表示曲率没有提供稳定收益，策略近似 uniform；

\[
\alpha_\kappa>0
\]

表示瓶颈程度更高的节点应获得更高基础广播概率。

必须记录训练过程中：

- `alpha_raw`；
- \(\alpha_\kappa\)；
- \(q^{(1)}\) 的 mean/std/min/max；
- \(q^{(1)}\) 与 \(s^\kappa\) 的相关性；
- 实际广播率；
- mean/max/p95 VAoI。

---

## 6. 阶段2：本地上下文残差 MLP

### 6.1 角色定义

阶段2不应再称为“只含动态信息的 MLP”。准确名称为：

> 本地上下文残差 MLP（local-context residual MLP）

它可以同时输入：

- 本地动态状态；
- 非曲率静态结构；
- 可选场景上下文；
- 第一层基础概率的只读参考。

主模型中不重复输入第一层已经使用的原始曲率 max 特征。

### 6.2 最终策略

第一层：

\[
\ell_i^{(1)}
=
\operatorname{logit}(b)
+
\alpha_\kappa(s_i^\kappa-c_\kappa),
\]

\[
q_i^{(1)}=\sigma(\ell_i^{(1)}).
\]

第二层：

\[
\delta_i(t)
=
\delta_{\max}
\tanh
g_\theta\!\left(
\mathbf x_i^{(2)}(t)
\right).
\]

最终概率：

\[
q_i(t)
=
\sigma\!\left[
\ell_i^{(1)}+\delta_i(t)
\right].
\]

每个时隙只执行：

\[
a_i(t)\sim\operatorname{Bernoulli}(q_i(t)).
\]

禁止先按 \(q_i^{(1)}\) 采样，再由第二层决定是否覆盖或重采样。

### 6.3 第二层主模型输入

固定 \(N,u,b\) 的单场景训练，主模型输入为：

\[
\boxed{
\mathbf x_i^{(2)}(t)=
\left[
\widetilde d_i,\,
\widetilde\tau_i^{tx},\,
\widetilde n_i^{con},\,
\widetilde c_i^{EWMA},\,
U_i,\,
G_i^{mean},\,
G_i^{max},\,
C_i^{nbr},\,
\operatorname{sg}(q_i^{(1)})
\right]
}
\]

共 9 维。

各项定义如下。

#### 1. 归一化节点度

\[
\widetilde d_i
=
\frac{\log(1+d_i)}{\log(1+N)}.
\]

它表示潜在竞争规模和局部结构密度。节点度与干扰 EWMA 相关，但二者不可互相替代。

#### 2. 距离上次广播的时间

\[
\widetilde\tau_i^{tx}
=
\tanh\!\left(b\,\tau_i^{tx}\right).
\]

#### 3. 连续广播尝试次数

\[
\widetilde n_i^{con}
=
\tanh\!\left(
\frac{n_i^{con}}{s_{con}}
\right).
\]

当前代码中的 `consecutive_tx_attempts` 统计广播尝试，而不是成功解码。

#### 4. 残余干扰压力 EWMA

\[
\widetilde c_i^{EWMA}
=
\tanh\!\left(
\frac{c_i}{s_c}
\right),
\qquad s_c=5.0\ \text{为当前默认值}.
\]

完整语义和迁移要求见第 7 节。

#### 5. 自身信息增量

设 \(v_{i,m}(t)\) 为节点 \(i\) 当前缓存版本，\(v_{i,m}^{last\_tx}\) 为节点上次尝试广播时冻结的数据包版本：

\[
U_i(t)
=
\frac{1}{N}
\sum_{m=1}^{N}
\mathbb I[
v_{i,m}(t)>v_{i,m}^{last\_tx}
].
\]

#### 6. 邻居新鲜度需求均值

先定义节点 \(i\) 对邻居 \(j\) 的本地估计增益：

\[
g_{ij}(t)
=
\frac1N
\sum_m
\mathbb I[
v_{i,m}(t)>
\hat v_{j,m}^{(i)}(t)
].
\]

置信度权重为：

\[
w_{ij}(t)
=
\mathbb I_{ij}^{valid}
\exp\!\left(
-\frac{\operatorname{age}_{ij}(t)}{\tau_c}
\right).
\]

则：

\[
G_i^{mean}
=
\frac{\sum_jw_{ij}g_{ij}}
{\sum_jw_{ij}},
\]

分母为 0 时取 0。

#### 7. 邻居新鲜度需求最大值

\[
G_i^{max}
=
\max_{j:\,valid}g_{ij}.
\]

没有有效邻居估计时取 0。

#### 8. 邻居估计平均置信度

\[
C_i^{nbr}
=
\frac{1}{|\mathcal N_i|}
\sum_{j\in\mathcal N_i}w_{ij}.
\]

节点无邻居时取 0。

#### 9. 第一层基础概率参考

\[
q_{i,\mathrm{ref}}^{(1)}
=
\operatorname{sg}(q_i^{(1)}),
\]

其中 `sg` 表示 `stop_gradient` 或 PyTorch 中的 `.detach()`。

### 6.4 跨场景输入

若同一模型需要跨 \(N,u,b\) 训练，再加入：

\[
\left[
\log(1+N),u,b
\right].
\]

此时第二层输入为 12 维。

要求配置显式区分：

```yaml
actor:
  stage: 2
  include_scenario_context: false  # 固定 N,u,b 时为 false
```

固定 \(N,u,b\) 的主实验不要把三个常数特征保留在输入中后声称网络利用了它们。

### 6.5 MLP 结构

默认结构：

\[
9\rightarrow64\rightarrow64\rightarrow1
\]

或跨场景时：

\[
12\rightarrow64\rightarrow64\rightarrow1.
\]

隐藏层使用 ReLU。最后一层权重和 bias 必须零初始化，从而阶段2训练开始时：

\[
\delta_i(t)=0,
\]

\[
q_i(t)=q_i^{(1)}.
\]

`delta_max` 必须写入 YAML 和 checkpoint，不要只作为代码中的隐藏常数。

### 6.6 为什么输入 `stop_gradient(q_base)`

即使 MLP 不输入 \(q_i^{(1)}\)，最终加法已经利用第一层：

\[
\frac{q_i(t)}{1-q_i(t)}
=
\frac{q_i^{(1)}}{1-q_i^{(1)}}
\exp[\delta_i(t)].
\]

但若 MLP 看不到 \(q_i^{(1)}\)，具有相同上下文特征的节点会得到相同的 log-odds 修正，即使它们的基础概率不同。

主模型因此输入：

\[
\operatorname{sg}(q_i^{(1)}).
\]

这样第二层可以根据“当前基础概率已经多高”调整残差，同时避免通过参考输入产生一条额外的间接梯度路径。

注意：

- `.detach()` 只作用于作为 MLP 输入的 \(q_i^{(1)}\)；
- 不要 detach 最终加法中的 \(\ell_i^{(1)}\)；
- 因此 \(\alpha_\kappa\)仍能从

\[
\ell_i^{(1)}
\rightarrow
\ell_i^{(1)}+\delta_i
\rightarrow
q_i
\]

这条直接路径获得梯度。

### 6.7 阶段2的第一层训练方式

主配置：

1. 从已训练的阶段1 checkpoint 初始化 \(a_\kappa\) 和 Critic；
2. MLP 最后一层零初始化；
3. 阶段2中默认允许 \(a_\kappa\)和 MLP 联合训练；
4. `q_base` 参考输入保持 detach；
5. 另做 `freeze_stage1=true` 消融，以区分“固定曲率基础上的动态修正”与“联合适配”的收益。

---

## 7. 拥塞 EWMA：当前代码语义与新版本要求

### 7.1 当前实际测量量

当前 `state/local_knowledge.py` 对本时隙静默节点 \(i\) 计算：

\[
m_i(t)
=
\log\left[
1+
\max\left(
\frac{J_i(t)-N_0}{N_0},
0
\right)
\right],
\]

其中：

- \(J_i(t)\) 来自 `decode.interference_plus_noise[i]`；
- \(N_0\) 是噪声功率；
- 括号中是非负 INR；
- `log1p` 已在这里完成一次长尾压缩。

随后更新：

\[
\boxed{
c_i(t)
=
\beta_c c_i(t-1)
+
(1-\beta_c)m_i(t)
}
\]

当前默认：

\[
\beta_c=0.8.
\]

即：

\[
c_i(t)=0.8c_i(t-1)+0.2m_i(t).
\]

这里的 0.8 是旧值权重，不是新样本权重。

### 7.2 哪些节点更新

只有静默节点更新：

\[
a_i(t)=0
\Longrightarrow
c_i(t)\text{ 更新}.
\]

发送节点在半双工假设下不能可靠测量：

\[
a_i(t)=1
\Longrightarrow
c_i(t)=c_i(t-1).
\]

该状态在 `complete_step()` 中更新，因此时隙 \(t\) 得到的测量只能进入时隙 \(t+1\) 的 Actor 观测，禁止同槽泄漏。

### 7.3 `interference_plus_noise` 的准确含义

当前 `StrongestSignalDecoder`：

1. 对静默接收节点寻找正在发送的物理邻居；
2. 若存在候选邻居，将最强候选作为期望信号；
3. 从分母中扣除该期望信号功率；
4. 剩余活跃发送功率与噪声构成 `interference_plus_noise`；
5. 若不存在候选邻居，则所有活跃发送功率均进入干扰项。

所以该特征不是“发送邻居数量”，也不是简单的 channel-busy 二值量。应称为：

> local residual-interference-pressure EWMA  
> 本地残余干扰压力 EWMA

它同时受以下因素影响：

- 并发发送节点数量；
- 节点距离和拓扑密度；
- 路径损耗；
- 阴影衰落；
- 快衰落；
- 最强候选信号选择。

### 7.4 新版本保留的数值形式

继续保留：

\[
\frac{I_i}{N_0}
\rightarrow
\log\left(1+\frac{I_i}{N_0}\right)
\rightarrow
\mathrm{EWMA}_{\beta_c}
\rightarrow
\tanh(c_i/s_c).
\]

明确禁止：

- 在 `features.py` 再做一次 `log1p`；
- 用 \(c_i/d_i\) 或 \(c_i/\max(d_i,1)\) 取代绝对干扰压力；
- 因为输入了节点度就删除 EWMA；
- 因为输入了 EWMA 就删除节点度。

节点度表示潜在结构竞争规模，EWMA 表示近期真实无线干扰压力，应同时进入第二层。

### 7.5 配置归属迁移

当前 `engine.py` 使用：

```python
getattr(self.policy, "congestion_ewma_alpha", 0.8)
```

这会把观测状态参数错误地挂到 policy 上，并在 NN 训练中依赖隐藏默认值。

新版本改为显式配置：

```yaml
observation:
  congestion_ewma_beta: 0.8
  congestion_feature_scale: 5.0
```

实现要求：

1. `congestion_ewma_beta` 明确表示旧值权重；
2. `congestion_feature_scale` 控制 \(\tanh(c_i/s_c)\)；
3. 两者由模拟器/观测配置读取，不从 policy 对象读取；
4. `SimulationParameters` 或等价配置对象保存并校验两者；
5. \(\beta_c\in[0,1]\)；
6. \(s_c>0\)；
7. 避免继续使用 `alpha` 命名，以免与 \(\alpha_\kappa\) 混淆。

建议修改位置：

- `src/curvature_gossip/state/local_knowledge.py`
- `src/curvature_gossip/simulator/engine.py`
- `src/curvature_gossip/learning/features.py`
- YAML 配置解析和默认配置文件
- 对应单元测试

如果为旧代码保留参数别名，只能作为有明确 warning 的临时兼容层；新配置和日志统一使用 `beta`。

---

## 8. 第二层不使用和可选使用的特征

### 8.1 主模型删除的重复特征

#### 上一时隙动作

当前：

\[
n_i^{con}(t)=
\begin{cases}
n_i^{con}(t-1)+1,&a_i(t-1)=1,\\
0,&a_i(t-1)=0.
\end{cases}
\]

因此：

\[
a_i(t-1)=\mathbb I[n_i^{con}(t)>0].
\]

当前 `consecutive_tx_attempts` 与上一时隙广播尝试严格对应，所以第二层不再加入 `previous_action`。

若未来把连续次数改成“连续成功解码次数”，才需要重新评估二者是否重复。

#### 广播债务

主模型关闭 `broadcast_debt`，因为当前阶段没有正式预算约束。债务只允许在后续预算扩展中使用。

### 8.2 主模型不重复输入的曲率特征

主模型的 MLP 不输入：

- \(s_i^\kappa=\max_j b_{ij}\)；
- `incident_bottleneck_max`；
- 第一层 logit；
- \(\alpha_\kappa\)；
- 与 \(q_i^{(1)}\) 同时重复的其他等价量。

第一层参考只保留一个：

\[
\operatorname{sg}(q_i^{(1)}).
\]

### 8.3 曲率增强消融

以下特征不进入 `Hierarchical-clean` 主模型，但保留为消融：

1. `incident_bottleneck_mean`

\[
\overline b_i
=
\operatorname{mean}_{j\in\mathcal N_i}b_{ij}.
\]

2. `fresh_bottle`

\[
G_i^{bottle}
=
\frac{
\sum_jw_{ij}b_{ij}g_{ij}
}{
\sum_jw_{ij}b_{ij}
}.
\]

建议配置：

```yaml
actor:
  include_curvature_mean: false
  include_curvature_freshness_interaction: false
```

实验命名：

- `Hierarchical-clean`：9/12 维主输入；
- `Hierarchical+mean-curv`：额外加入 incident mean；
- `Hierarchical+interaction`：再加入 \(G_i^{bottle}\)。

不要同时打开多个特征却仍把结果标记为 clean 主模型。

---

## 9. 建议的配置结构

下面的字段名可以按现有配置系统小幅调整，但语义和显式性必须保留：

```yaml
policy:
  type: hierarchical_curvature_actor

actor:
  stage: 2                         # 1 or 2

  curvature:
    score: incident_bottleneck_max
    center: 0.0                    # 由训练拓扑离线统计后写入
    alpha_parameterization: softplus
    alpha_init: 0.1
    alpha_override: null           # 严格 alpha=0 测试时设为 0.0

  residual:
    enabled: true
    hidden_dims: [64, 64]
    activation: relu
    delta_max: 1.0
    zero_init_output: true
    use_stage1_reference: true
    detach_stage1_reference: true
    include_scenario_context: false
    include_curvature_mean: false
    include_curvature_freshness_interaction: false
    freeze_stage1: false

observation:
  consecutive_tx_scale: 3.0
  neighbor_confidence_time_constant: 20.0
  congestion_ewma_beta: 0.8
  congestion_feature_scale: 5.0

training:
  reward: negative_mean_vaoi
  use_budget_advantage: false
  update_multiplier: false
  use_debt_advantage: false
  entropy_coefficient: 0.0
```

配置校验要求：

- `stage=1` 时 residual 必须被禁用或完全不实例化；
- `stage=2` 时 residual 必须实例化；
- `use_stage1_reference=true` 时只加入 detached \(q_i^{(1)}\)；
- `detach_stage1_reference=false` 不作为主实验配置；
- `alpha_override` 非空时不更新 `alpha_raw`；
- `include_scenario_context=false` 时输入维数为 9；
- `include_scenario_context=true` 时输入维数为 12；
- 曲率增强特征每开启一项，维数增加 1，并写入 checkpoint 元数据。

---

## 10. 代码组织建议

不要继续让一个 `encode_observations()` 同时承担所有架构语义。建议拆分为：

```python
encode_curvature_score(observations) -> [N, 1]
encode_stage2_context(
    observations,
    target_tx_ratio,
    update_probability,
    ...,
) -> [N, 8] or [N, 11]
```

随后由 Actor 组装第一层参考：

```python
curvature_score = encode_curvature_score(observations)
base_logit, q_base = stage1(curvature_score, b, c_kappa)

context = encode_stage2_context(...)
stage1_reference = q_base.detach()
mlp_input = concat([context, stage1_reference], dim=-1)

delta = delta_max * tanh(residual_mlp(mlp_input))
q_final = sigmoid(base_logit + delta)
```

其中：

- 固定场景 `context` 为 8 维，拼接 `q_base` 后为 9 维；
- 跨场景 `context` 为 11 维，拼接 `q_base` 后为 12 维；
- Stage1 不调用 `encode_stage2_context()`；
- 编码器必须返回特征名称/顺序，便于 checkpoint 和日志核对。

建议新建清晰的 Actor 类，例如：

```python
class CurvatureBaseActor(nn.Module):
    ...

class HierarchicalResidualActor(nn.Module):
    ...
```

不要把阶段1伪装成 `Linear(1,1)` 后又加入可学习 bias。它只有共享的 `alpha_raw`。

---

## 11. PPO 与梯度实现要求

### 11.1 Bernoulli 概率

对数概率使用数值稳定的 Bernoulli 实现。优先让网络输出最终 logits：

\[
\ell_i(t)=\ell_i^{(1)}+\delta_i(t),
\]

再构造：

```python
dist = torch.distributions.Bernoulli(logits=final_logits)
```

不要先 sigmoid 后手工计算：

```python
log(q)
log(1 - q)
```

以免在概率接近 0 或 1 时产生数值问题。

### 11.2 Stage1 reference 的梯度测试

需要验证两件事同时成立：

1. `q_base` 作为 MLP 输入时已经 detach；
2. `base_logit` 在最终加法中未 detach。

换言之，\(\alpha_\kappa\)不应通过：

\[
q_i^{(1)}
\rightarrow
g_\theta
\rightarrow
\delta_i
\]

这条路径收到梯度，但仍应通过：

\[
\ell_i^{(1)}
\rightarrow
\ell_i^{(1)}+\delta_i
\rightarrow
\log\pi
\]

收到梯度。

### 11.3 优势和损失

保持：

- Critic 学习 VAoI return；
- GAE advantage 按 batch 标准化；
- 所有节点共享全局时隙 advantage；
- Actor loss 对 `time × node` 样本平均；
- 不加入 debt/budget correction；
- 记录 clip fraction、entropy、approx KL 和 value loss。

---

## 12. 必须添加的测试

### 12.1 曲率与阶段1

1. AF3 的 \(b_{ij}\) 对称且位于 \([0,1]\)。
2. 无邻居节点得到 \(s_i^\kappa=0\)。
3. `alpha_override=0.0` 时所有节点严格得到 \(q_i=b\)。
4. \(\alpha_\kappa>0\) 时，固定其他量，\(q_i^{(1)}\) 对 \(s_i^\kappa\) 单调不减。
5. 阶段1 Actor 的可训练 Actor 参数只有一个 `alpha_raw`。
6. 阶段1不实例化 residual MLP。

### 12.2 EWMA

1. 静默节点按

\[
c_t=\beta_cc_{t-1}+(1-\beta_c)\log1p(\mathrm{INR})
\]

更新。
2. 发送节点保持旧值。
3. `beta=0.8` 时数值与当前实现一致。
4. `congestion_feature_scale=5.0` 时特征与当前实现一致。
5. 不发生第二次 `log1p`。
6. 不除以节点度。
7. YAML 中修改 beta 或 scale 后确实影响结果。
8. EWMA 参数不再从 policy 对象读取。
9. 时隙 \(t\) 的测量不进入时隙 \(t\) 的动作。

### 12.3 阶段2输入

1. 固定场景 clean 输入严格为 9 维。
2. 跨场景 clean 输入严格为 12 维。
3. `previous_action` 不在特征列表。
4. `broadcast_debt` 不在 clean 特征列表。
5. `bottleneck_max` 不重复进入 MLP。
6. `q_base` 只出现一次且已 detach。
7. 开启 mean-curvature 或 interaction 时维数和 checkpoint 元数据同步变化。

### 12.4 初始化和梯度

1. 第二层输出层零初始化后：

\[
\delta_i=0,\qquad q_i=q_i^{(1)}.
\]

2. `q_base` 参考路径不对 `alpha_raw` 反传。
3. 最终 base-logit 路径仍能对 `alpha_raw` 反传。
4. `freeze_stage1=true` 时 `alpha_raw.grad` 为空或为零。
5. `freeze_stage1=false` 时在非退化 batch 上 `alpha_raw` 能更新。

### 12.5 端到端 smoke test

至少完成：

1. 小图阶段1训练若干 rollout；
2. Critic 参数更新；
3. `alpha_raw` 参数更新且无 NaN；
4. 保存并重新加载 checkpoint 后概率一致；
5. 从阶段1 checkpoint 启动阶段2；
6. 初始阶段2概率与阶段1一致；
7. 训练后 residual MLP 参数发生变化；
8. 全流程概率有限且位于 \((0,1)\)；
9. Actor 运行不读取 centralized state。

---

## 13. 实验与消融顺序

### 13.1 阶段1

至少比较：

1. `Uniform-b`：\(\alpha=0\)；
2. `Fixed-alpha`：人工设置若干 \(\alpha>0\)；
3. `Learned-alpha`：PPO 学习共享 \(\alpha_\kappa\)；
4. `Matched-rate random`：匹配 learned policy 的实际广播率。

### 13.2 阶段2主消融

从同一个阶段1 checkpoint 初始化：

1. `Stage1 only`；
2. `Stage2-no-base-ref`：MLP 不读取 \(q_i^{(1)}\)；
3. `Stage2-base-ref`：读取 \(\operatorname{sg}(q_i^{(1)})\)，主模型；
4. `Stage2-base-ref-frozen`：读取基础概率但冻结第一层；
5. `Stage2-base-ref-joint`：读取基础概率并联合训练第一层，主配置。

### 13.3 曲率增强消融

在主模型基础上依次比较：

1. `Hierarchical-clean`；
2. `Hierarchical+mean-curv`；
3. `Hierarchical+interaction`。

### 13.4 公平比较要求

所有比较保持：

- 相同固定拓扑集合；
- 相同更新事件种子；
- 相同信道/衰落种子；
- 相同评估时隙和 warmup；
- 报告均值、标准差和每拓扑结果；
- 同时报告 VAoI 与实际发送率；
- matched-rate baseline 使用被比较策略的实测速率，而不是只复用目标 \(b\)。

---

## 14. 诊断输出

训练历史和评估结果至少加入：

```text
alpha_raw
alpha_kappa
q_base_mean
q_base_std
q_base_min
q_base_max
delta_mean
delta_std
delta_abs_mean
delta_saturation_fraction
q_final_mean
q_final_std
q_final_min
q_final_max
actual_tx_ratio
mean_vaoi
max_vaoi
p95_vaoi
actor_loss
value_loss
entropy
approx_kl
clip_fraction
```

还应在离线评估中输出：

- `corr(s_kappa, q_base)`；
- `corr(s_kappa, q_final)`；
- `corr(congestion_ewma, delta)`；
- `corr(U_i, delta)`；
- `corr(G_i_mean, delta)`；
- 按 `q_base` 分位区间统计 \(\delta\)，检查第二层是否真正利用第一层参考；
- 各节点实际发送率分布。

---

## 15. 完成标准

只有同时满足以下条件，才算完成本轮修改：

1. 阶段1可通过 YAML 独立运行，Actor 只有共享 `alpha_raw`；
2. `alpha_override=0.0` 严格退化为 uniform-\(b\)；
3. 阶段2从阶段1 checkpoint 初始化；
4. 阶段2 clean 输入在固定场景下为 9 维；
5. 第二层同时包含节点度和现有残余干扰压力 EWMA；
6. EWMA 数值公式保留，不做度归一化，也不重复 `log1p`；
7. EWMA 的 beta 和 feature scale 从 observation/simulator 配置读取；
8. `previous_action` 和 `broadcast_debt` 不进入 clean 主模型；
9. 第一层参考使用 `.detach()` 的 \(q_i^{(1)}\)；
10. 最终 base-logit 直接路径不 detach，联合训练时 \(\alpha_\kappa\)仍可学习；
11. residual MLP 输出层零初始化，阶段2初始策略严格等于阶段1；
12. 阶段1、阶段2和 matched-rate random 均可在同一评估入口运行；
13. 所有新增测试通过；
14. README 或运行说明给出阶段1、阶段2和主要消融的命令；
15. 最终汇报修改文件、测试命令、测试结果及仍未运行的正式实验。

---

## 16. Codex 执行顺序

请严格按以下顺序实施：

1. 检查 `NN` 分支代码、现有配置和测试，不覆盖用户修改。
2. 建立 `NN_2layers` 开发分支。
3. 先迁移 EWMA 配置归属，添加等价性测试。
4. 拆分曲率分数编码和阶段2上下文编码。
5. 实现阶段1单参数 Actor。
6. 添加阶段1单元测试和小图 smoke test。
7. 实现阶段2 residual MLP 和 detached `q_base` 参考。
8. 添加维数、初始化和梯度路径测试。
9. 接入 checkpoint、日志和 YAML。
10. 添加 no-base-ref、freeze-stage1、mean-curv、interaction 消融配置。
11. 运行现有测试和新增测试。
12. 先做短 smoke experiment，不直接启动长时间正式训练。
13. 汇报代码修改、测试结果、短实验结果和正式实验建议。

在任何一步发现现有代码语义与本文件冲突时，先给出具体源码证据和影响，不要静默改变本文件的研究设计。
