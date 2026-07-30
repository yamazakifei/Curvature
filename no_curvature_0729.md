# 实现无曲率 Stage-2 消融实验

实现一个严格的“不使用曲率信息”的 Stage-2 消融实验。

## 1. 实验目的

当前两层策略为：

[
q_i
===

\sigma\left[
\operatorname{logit}(b)
+
\alpha_\kappa(s_{\kappa,i}-c_\kappa)
+
\Delta_\theta(x_i,q_i^{\mathrm{base}})
\right],
]

其中：

[
q_i^{\mathrm{base}}
===================

\sigma\left[
\operatorname{logit}(b)
+
\alpha_\kappa(s_{\kappa,i}-c_\kappa)
\right].
]

需要增加一个严格的无曲率消融策略：

[
q_i^{\mathrm{no_curvature}}
===========================

\sigma\left[
\operatorname{logit}(b)
+
\Delta_\theta(x_i)
\right].
]

该消融应满足：

1. 第一层等效设置为 (\alpha_\kappa=0)；
2. 第一层基准概率恒为：
   [
   q_i^{\mathrm{base}}=b;
   ]
3. 第二层 residual MLP 不接收第一层输出；
4. 第二层不接收任何曲率特征；
5. 最终概率仍以相同的 `logit(b)` 为 anchor；
6. Actor 的 PPO、Critic、奖励、训练轮数、验证场景和其他超参数保持不变；
7. 实验与当前曲率 Stage-2 的唯一本质差异应是是否使用曲率及 Stage-1 reference。

---

## 2. 不要只设置 `alpha_override: 0.0`

当前 Stage-2 实现即使设置：

```yaml
actor:
  curvature:
    alpha_override: 0.0
```

仍然会计算：

```python
q_base_reference = stop_gradient(base_probabilities)
residual_input = concat([residual_context, q_base_reference])
```

此时虽然 `q_base_reference=b`，它仍然是 residual MLP 的输入之一。

本消融要求完全删除该输入，不能仅令它成为常数。

最终 residual MLP 输入必须严格为：

```text
residual_context
```

而不是：

```text
concat(residual_context, detached_stage1_probability)
```

---

## 3. 建议的配置接口

扩展 Stage-2 residual 配置，允许：

```yaml
actor:
  stage: 2

  curvature:
    score: incident_bottleneck_max
    center: auto
    alpha_parameterization: softplus
    alpha_init: 1.5
    alpha_override: 0.0

  residual:
    enabled: true
    hidden_dims: [64, 64]
    activation: relu
    delta_max: 1.0
    zero_init_output: true

    use_stage1_reference: false
    detach_stage1_reference: true

    include_scenario_context: false
    include_curvature_mean: false
    include_curvature_freshness_interaction: false
    freeze_stage1: true
```

其中：

* `alpha_override: 0.0` 明确表示第一层曲率系数为零；
* `use_stage1_reference: false` 表示 residual MLP 不接收 `q_base`；
* `detach_stage1_reference` 在 reference 关闭时不应产生作用；
* 当前曲率 Stage-2 配置继续使用 `use_stage1_reference: true`，保持现有行为。

更推荐在 metadata 中明确记录：

```yaml
actor:
  architecture_version: no_curvature_stage2_residual_v1
  base_mode: uniform_logit_anchor
```

以避免该 checkpoint 与原始 `curvature_stage2_residual_v1` 混淆。

---

## 4. 修改配置校验

修改：

```text
curvature-gossip/src/curvature_gossip/learning/trainer.py
```

当前 `_stage1_actor_config()` 强制要求：

```python
use_stage1_reference == True
detach_stage1_reference == True
```

请改为支持两种合法模式。

### 模式 A：原始曲率 Stage 2

```yaml
alpha_override: 1.5
use_stage1_reference: true
```

Actor 输入特征名：

```text
[
    dynamic context features,
    detached_stage1_probability
]
```

架构版本：

```text
curvature_stage2_residual_v1
```

### 模式 B：无曲率消融 Stage 2

```yaml
alpha_override: 0.0
use_stage1_reference: false
```

Actor 输入特征名只能包含：

```text
dynamic context features
```

架构版本：

```text
no_curvature_stage2_residual_v1
```

为保证实验定义严格，建议增加以下校验：

```python
if not use_stage1_reference:
    if float(curvature.get("alpha_override", -1.0)) != 0.0:
        raise ValueError(
            "Stage-2 without Stage-1 reference requires "
            "actor.curvature.alpha_override=0.0"
        )
```

同时确保：

```python
include_curvature_mean == False
include_curvature_freshness_interaction == False
```

无曲率模式下如开启任何曲率增强项，应直接报错。

---

## 5. 修改 Actor 图结构

修改：

```text
curvature-gossip/src/curvature_gossip/learning/ctde_ppo.py
```

当前 Stage 2 固定使用：

```python
q_base_reference = tf.stop_gradient(
    tf.expand_dims(self.base_probabilities, axis=1)
)
self.residual_input = tf.concat(
    [self.residual_context, q_base_reference],
    axis=1,
)
```

请根据：

```python
use_stage1_reference = bool(
    residual.get("use_stage1_reference", True)
)
```

分别构造输入。

### 使用曲率 reference

保持现有实现：

```python
q_base_reference = tf.stop_gradient(
    tf.expand_dims(self.base_probabilities, axis=1),
    name="detached_stage1_reference",
)
self.residual_input = tf.concat(
    [self.residual_context, q_base_reference],
    axis=1,
    name="residual_input",
)
```

### 无曲率消融

直接使用：

```python
self.residual_input = tf.identity(
    self.residual_context,
    name="residual_input",
)
```

禁止创建或拼接 `q_base_reference`。

最终公式保持：

```python
self.logits = self.base_logits + self.residual_delta
self.probabilities = sigmoid(self.logits)
```

由于 `alpha_override=0.0`，应严格有：

```python
self.base_logits == logit(target_tx_ratio)
self.base_probabilities == target_tx_ratio
```

即：

[
q_i
===

\sigma\left[
\operatorname{logit}(b)+\Delta_\theta(x_i)
\right].
]

不要改成概率空间加法：

```python
q = clip(b + delta)
```

必须保持现有 logit-space residual 形式。

---

## 6. residual MLP 输入维数

当前动态 context：

* 不包含 scenario context 时为 8 维；
* 包含 scenario context 时为 11 维。

因此：

### 原始 Stage 2

```text
input_dim = context_width + 1
```

最后一维为：

```text
detached_stage1_probability
```

### 无曲率消融

```text
input_dim = context_width
```

不得为缺失的 reference 补零，也不得把常数 `b` 拼接进去。

原因是补零或拼接 `b` 都不是“删除第一层输入”，而只是替换第一层输入。

---

## 7. 特征编码

检查：

```text
curvature-gossip/src/curvature_gossip/learning/features.py
```

当前 `STAGE2_CONTEXT_FEATURE_NAMES` 为：

```text
normalized_degree
time_since_last_tx
consecutive_tx_attempts
congestion_ewma
self_information_increment
neighbor_freshness_mean
neighbor_freshness_max
neighbor_confidence_mean
```

这些特征可以保留。

注意：

* `normalized_degree` 是普通拓扑局部信息，不是曲率，可以保留；
* 不允许加入 `incident_bottleneck_max`；
* 不允许加入 `incident_bottleneck_mean`；
* 不允许加入 curvature-freshness interaction；
* 无曲率模式下 `input_feature_names` 中不得出现：

  * `incident_bottleneck_max`
  * `detached_stage1_probability`
  * 任何名称中带 `curvature` 或 `kappa` 的特征。

现有 `Stage2EncodedObservations` 可以暂时继续携带 `curvature_scores`，用于复用统一代码路径，但必须满足：

1. `alpha_override=0.0`；
2. 该值不会影响 `base_logits`；
3. 该值不会进入 residual MLP；
4. 该值不产生可训练梯度；
5. checkpoint metadata 明确标记为 no-curvature。

更彻底的实现可以为该消融增加独立编码数据类，使 Actor 路径完全不需要 `curvature_scores`。但不要为此大规模重构现有训练代码。

---

## 8. Alpha 参数处理

无曲率消融中：

```yaml
alpha_override: 0.0
freeze_stage1: true
```

必须确保：

* `effective_alpha` 严格为 0；
* `alpha_raw` 不参与优化；
* PPO 的 actor optimizer 只更新 residual MLP；
* 不允许训练过程中 alpha 偏离 0；
* 保存的 metadata 中记录：

  * `alpha_override: 0.0`
  * `effective_alpha: 0.0`
  * `stage1_reference_used: false`

即使 TensorFlow 图中为了 checkpoint 兼容仍创建 `alpha_raw`，该变量也必须被冻结且不影响前向结果。

---

## 9. 新增实验配置

新增正式配置，例如：

```text
curvature-gossip/configs/nn_2layers_stage2_no_curvature.yaml
```

建议从当前正式 Stage-2 配置复制，只修改 Actor 消融相关字段。

核心配置：

```yaml
experiment:
  id: nn_2layers_stage2_no_curvature

actor:
  stage: 2

  curvature:
    score: incident_bottleneck_max
    center: auto
    alpha_parameterization: softplus
    alpha_init: 1.5
    alpha_override: 0.0

  residual:
    enabled: true
    hidden_dims: [64, 64]
    activation: relu
    delta_max: 1.0
    zero_init_output: true
    use_stage1_reference: false
    detach_stage1_reference: true
    include_scenario_context: false
    include_curvature_mean: false
    include_curvature_freshness_interaction: false
    freeze_stage1: true
```

新增 smoke 配置：

```text
curvature-gossip/configs/nn_2layers_stage2_no_curvature_smoke.yaml
```

smoke 配置应使用较少 episodes 和 slots，但执行完整：

* rollout；
* PPO update；
* validation；
* checkpoint save；
* checkpoint restore；
* diagnostics 输出。

---

## 10. 公平对比要求

无曲率消融必须和原始曲率 Stage 2 使用完全相同的：

* topology seeds；
* source update seeds；
* shadowing seeds；
* fading seeds；
* action seeds；
* 节点数 (N)；
* 更新概率 (u)；
* 目标广播率 (b)；
* rollout slots；
* episodes；
* PPO epochs；
* learning rate；
* reward scale；
* Critic 结构；
* residual hidden dimensions；
* residual `delta_max`；
* residual 输出层零初始化；
* validation scenarios；
* checkpoint 选择标准。

原始曲率组：

[
q_i^{\mathrm{curv}}
===================

\sigma\left[
\operatorname{logit}(b)
+
\alpha_\kappa(s_{\kappa,i}-c_\kappa)
+
\Delta_\theta(x_i,q_i^{\mathrm{base}})
\right].
]

无曲率组：

[
q_i^{\mathrm{no_curv}}
======================

\sigma\left[
\operatorname{logit}(b)
+
\Delta_\theta(x_i)
\right].
]

不要同时修改网络容量、奖励或训练参数，否则无法归因曲率贡献。

需要注意，删除一个 reference 输入后，无曲率 residual MLP 的第一层参数量会略少。这是严格删除曲率信息的自然结果，不要通过加入无意义常数特征来补齐参数量。

---

## 11. 日志和诊断

在 Stage-2 diagnostics、training history 和 validation history 中增加或确认以下字段：

```text
actor_architecture_version
use_stage1_reference
effective_alpha
q_base_mean
q_base_std
delta_mean
delta_std
q_final_mean
q_final_std
actual_tx_ratio
mean_VAoI
```

无曲率模式必须满足：

```text
effective_alpha = 0
q_base_mean ≈ b
q_base_std ≈ 0
use_stage1_reference = false
```

建议增加：

```text
residual_input_dim
residual_input_feature_names
```

无 scenario context 时，无曲率消融应报告：

```text
residual_input_dim = 8
```

原始曲率 Stage 2 应报告：

```text
residual_input_dim = 9
```

---

## 12. 单元测试

至少增加以下测试。

### Test 1：alpha=0 时 base 恒为 b

对不同曲率分数：

```python
s_kappa = [0.0, 0.2, 0.7, 1.0]
b = 0.1
```

要求：

```python
q_base == [0.1, 0.1, 0.1, 0.1]
```

允许数值误差：

```text
atol <= 1e-6
```

### Test 2：无曲率 residual 输入不包含 Stage-1 reference

当：

```yaml
use_stage1_reference: false
include_scenario_context: false
```

要求：

```text
residual_input.shape[1] == 8
```

且 `input_feature_names` 中不包含：

```text
detached_stage1_probability
incident_bottleneck_max
```

### Test 3：改变曲率不改变输出

固定：

* residual context；
* target (b)；
* Actor 参数。

只改变：

```text
curvature_scores
```

例如从全 0 改成全 1。

无曲率模式下必须满足：

```python
q_final_before == q_final_after
```

误差：

```text
atol <= 1e-6
```

这是本消融最关键的功能测试。

### Test 4：零初始化输出

初始化后、训练前：

```text
delta = 0
q_final = b
```

对所有节点成立。

### Test 5：只有 residual 参数更新

完成一次 PPO update 后检查：

* residual MLP 至少一个参数发生改变；
* `alpha_raw` 不变；
* Critic 参数可以更新；
* `effective_alpha` 仍为 0。

### Test 6：原始 Stage-2 行为保持兼容

当：

```yaml
use_stage1_reference: true
```

要求：

* residual input 维数仍为 9 或 12；
* `detached_stage1_probability` 仍存在；
* 原有 Stage-2 tests 继续通过；
* 原始 Stage-2 checkpoint 的加载行为不被意外破坏。

如果输入维数变化导致 checkpoint 不兼容，应通过不同的 `architecture_version` 明确拒绝错误加载，并给出清晰报错，不要静默加载。

### Test 7：配置非法组合

以下配置应报错：

```yaml
alpha_override: 1.5
use_stage1_reference: false
```

因为它会让曲率仍然通过 base logit 影响最终输出，不能称为 no-curvature。

以下配置也应报错：

```yaml
alpha_override: 0.0
use_stage1_reference: false
include_curvature_mean: true
```

---

## 13. Smoke run

运行：

```powershell
cd D:\ZMF\2026Curvature\curvature-gossip

conda run --no-capture-output -n GRL_AoI_cpu37 python -m pytest -q

conda run --no-capture-output -n GRL_AoI_cpu37 python scripts/train_nn_ctde.py --config configs/nn_2layers_stage2_no_curvature_smoke.yaml
```

检查输出：

1. 测试全部通过；
2. 训练和验证可正常结束；
3. `q_base_mean` 等于配置中的 (b)；
4. `q_base_std` 接近零；
5. `residual_input_dim=8`；
6. `use_stage1_reference=false`；
7. `effective_alpha=0`；
8. checkpoint 可保存和恢复；
9. 改变 curvature score 不改变 Actor 输出。

---

## 14. 最终交付内容

完成后请给出：

1. 修改文件列表；
2. 每个文件的修改说明；
3. 新增配置文件；
4. 新增测试列表；
5. 最终概率公式；
6. no-curvature Actor 的准确输入维数和特征顺序；
7. smoke test 命令及结果；
8. `pytest` 结果；
9. 一组数值验证，展示：

   * 不同曲率分数下 `q_base` 均为 (b)；
   * 改变曲率分数不会改变 `q_final`；
10. 说明原始曲率 Stage-2 配置和 checkpoint 是否保持兼容。

不要只修改 YAML。必须修改配置校验、Actor 输入构造、metadata、diagnostics 和测试，使“不利用曲率”成为可验证的架构属性。
