# Codex 修改方案：Stage-1 基础概率搜索、Bmax 解耦与统一种子派生

## 0. 基线与目标

- 基线仓库：`yamazakifei/Curvature`
- 基线 commit：`71c42c9ee5cce55c9c8bb771ce9e4ca664eb58c9`
- 原训练入口保持不变：

```bash
python scripts/train_nn_ctde.py --config configs/GNN/mpnn_newC_ch1_search.yaml
```

本次修改需要完成：

1. 将全网平均广播概率上限 `Bmax` 与 Stage-1 基础概率 `b` 完全解耦；
2. 在原 Stage-2 PPO 训练前，自动执行 curvature-only 的 Stage-1 参数搜索；
3. `b` 使用“粗搜索 + 按 alpha 分别局部细化”，`alpha` 使用 YAML 给定列表；
4. 使用一个 `experiment.master_seed` 自动派生校准、搜索、训练和验证四类随机流；
5. 保存完整搜索、校准和种子清单；
6. 保持旧 YAML、显式验证场景和现有非 NN 实验兼容。

---

## 1. 参数语义拆分

### 1.1 全网概率上限

新增规范字段：

```yaml
constraints:
  max_tx_ratio: 0.20
```

定义：

\[
\mathbb E_{t,i}[q_{i,t}] \le B_{\max}.
\]

`max_tx_ratio` 用于：

- Stage-1 搜索候选范围；
- 搜索候选约束过滤；
- 预算乘子及预算日志；
- Critic/场景上下文中的预算信息；
- 验证场景与固定随机基线；
- `per_node_cap_multiplier` 的诊断阈值。

它不得直接作为 Stage-1 的基础 logit 中心。

### 1.2 Stage-1 基础工作点

新增字段：

```yaml
actor:
  curvature:
    base_tx_ratio: 0.10
```

定义：`base_tx_ratio=b` 是 Stage-1 curvature-only 在校准拓扑分布上的目标平均基础概率。

Stage-1 改为：

\[
q_i^{\mathrm{base}}
=
\sigma\left(\beta_0+\alpha_\kappa(s_i-c_\kappa)\right).
\]

其中：

- `c_kappa`：校准拓扑池上的公共曲率中心；
- `alpha_kappa`：曲率差异强度；
- `beta0`：针对候选 `(b, alpha)` 离线校准的公共截距；
- `beta0` 只能按候选和校准池确定，禁止逐拓扑在线校准。

### 1.3 兼容规则

配置解析后应统一生成内部规范字段：

- 新配置优先读取 `constraints.max_tx_ratio`；
- 若不存在，则回退到旧字段 `constraints.target_tx_ratio`；
- 新配置优先读取 `training.max_tx_ratios`；
- 若不存在，则回退到旧字段 `training.target_tx_ratios`；
- 新配置优先读取 `actor.curvature.base_tx_ratio`；
- 若不存在，则回退到规范化后的 `max_tx_ratio`，保持旧行为；
- 旧配置回退时打印一次清晰 warning，但不能报错。

旧 `target_tx_ratio` 在兼容路径中应被解释为 `Bmax`，不要再由新代码直接作为 Stage-1 中心。

---

## 2. 训练入口的整体流程

在 `train_ctde()` 正式创建 Stage-2 模型和开始 PPO 之前增加预处理：

```text
读取 YAML
  -> 规范化 Bmax / b / 兼容字段
  -> 构造统一 seed plan
  -> 若 base_search.enabled=true：
       生成 beta0 校准拓扑池
       计算公共 c_kappa
       生成 coarse b candidates
       读取 alpha_candidates
       为每个 (b, alpha) 校准 beta0
       curvature-only 粗搜索
       对每个 alpha 的最佳 b 分别局部细化
       在完整搜索池复核候选
       选择 (b*, alpha*, beta0*, c_kappa)
       保存搜索结果
     否则：
       使用 base_tx_ratio + alpha_override
       可按配置决定是否校准 beta0
  -> 将最终 Stage-1 参数写入 resolved config
  -> 冻结 Stage 1
  -> 按原流程创建 Actor/Critic
  -> 执行 Stage-2 PPO 训练
  -> 在固定验证池选择 checkpoint
```

搜索模块不要复制一套通信仿真逻辑；应复用现有 topology、curvature、channel、simulator 和 metrics 组件。

---

## 3. Stage-1 搜索设计

### 3.1 自动粗搜索 b

从 `Bmax` 自动生成等间隔候选：

```yaml
base_candidates:
  mode: coarse_to_fine
  min_fraction_of_max: 0.25
  max_fraction_of_max: 1.0
  coarse_count: 7
```

默认：

\[
b_k=B_{\max}\left[r_{\min}+k\frac{r_{\max}-r_{\min}}{K-1}\right].
\]

当 `Bmax=0.2` 时，默认得到：

```text
0.050, 0.075, 0.100, 0.125, 0.150, 0.175, 0.200
```

要求：

- 去重并排序；
- 所有候选必须满足 `0 < b <= Bmax`；
- 浮点比较使用容差；
- 生成结果写入搜索输出。

### 3.2 alpha 候选

只使用 YAML 显式列表：

```yaml
alpha_candidates: [0.0, 0.5, 1.0, 1.5, 2.0]
```

要求：

- 非空、有限、非负；
- 去重并保留确定性顺序；
- 若省略，默认只使用 `[actor.curvature.alpha_override]`；
- `alpha=0` 对应无曲率差异的均匀 Stage-1 内部基线。

### 3.3 每个 alpha 独立细化 b

粗搜索完成后，对每个 `alpha`：

1. 找到该 `alpha` 下最佳 coarse `b`；
2. 使用其左右相邻 coarse 点形成细化区间；
3. 在区间内生成 `refine_count` 个等间隔点；
4. 边界最优时使用 `[b_min, next]` 或 `[previous, Bmax]`；
5. 去掉已经运行过的 coarse 点；
6. 所有细化候选仍需重新校准自己的 `beta0`。

禁止只围绕全局最佳 alpha 的 b 做细化，因为不同 alpha 的最佳工作点可能不同。

### 3.4 beta0 离线校准

使用独立校准拓扑池，收集所有节点的曲率分数 `s_i`。先计算并冻结公共 `c_kappa`，然后针对每个 `(b, alpha)` 用二分法求：

\[
\frac{1}{M}\sum_{m=1}^{M}
\sigma\left(\beta_0+\alpha(s_m-c_\kappa)\right)=b.
\]

要求：

- 一个候选一个 `beta0`；
- 一个候选在所有拓扑上共用同一个 `beta0`；
- 禁止逐拓扑、逐时隙或推理期重新校准；
- 校准只需要生成拓扑并计算曲率，不运行 source/channel/action 仿真；
- 二分边界应自动扩展或使用足够宽的稳定区间；
- 保存目标均值、实际均值、误差、迭代次数和收敛状态。

### 3.5 搜索场景和公平比较

所有 `(b, alpha)` 候选必须复用完全相同的：

- topology realizations；
- source update realizations；
- shadowing/fading realizations；
- Bernoulli uniform random numbers。

候选参数不得进入随机种子派生。这样候选之间是配对比较。

建议搜索池：

- 总计 20 张 topology；
- 每张 topology 2 组动态随机重复；
- coarse 阶段先使用前 10 张 topology、每张 1 组动态重复；
- refine 和最终排序使用完整 20 x 2 场景。

### 3.6 约束与候选选择

用户约束是平均广播概率上限，因此候选可行性优先依据：

\[
\overline q_{\mathrm{search}}\le B_{\max}+\epsilon_q.
\]

不要用随机采样得到的 actual tx ratio 替代概率约束；actual tx ratio 只作为诊断。

候选排序：

1. 过滤平均策略概率超过 `Bmax + tolerance` 的候选；
2. 以搜索池 `mean_VAoI` 最小为主指标；
3. 若相对 VAoI 差异不超过 `tie_relative_tolerance`，优先选择更低的搜索池平均概率；
4. 再相同则优先更低的 `b`；
5. 再相同则按 alpha/YAML 顺序保证确定性。

---

## 4. 四个集合及默认规模

四个集合在逻辑上完全隔离：

| 集合 | 默认规模 | 用途 |
|---|---:|---|
| beta0 校准池 | 32 张 topology | 计算公共 `c_kappa`，为每个候选求 `beta0` |
| Stage-1 搜索池 | 20 topology x 2 动态重复 | curvature-only 选择 `(b*, alpha*)` |
| Stage-2 训练池 | 300 个独立 episode | PPO 梯度训练 |
| 固定验证池 | 20 个场景 | checkpoint 选择和泛化监控 |

默认四个池不共享随机流。由于仿真样本可生成，不需要为了省数据而重合。

注意：coarse 使用的 10 个场景是 Stage-1 搜索池的子集，不是额外验证集；refine 使用完整搜索池是允许的。

---

## 5. 统一种子设计

### 5.1 用户只配置一个根种子

```yaml
experiment:
  master_seed: 20260723
```

新自动模式下，不要求用户显式列出四个集合的种子，也不要求设置 `training.topology_seed_start`。

随机流统一按以下逻辑派生：

```text
seed stream = H(master_seed, pool_namespace, process_namespace, index...)
```

复用现有 `make_rng()`，不要自行使用不稳定的 Python `hash()`。

### 5.2 固定命名空间

建议固定如下命名空间，不允许候选参数进入命名空间：

```text
stage1_calibration / topology
stage1_search      / topology
stage1_search      / source_updates
stage1_search      / shadowing
stage1_search      / fading
stage1_search      / actions
stage2_training    / topology
stage2_training    / source_updates
stage2_training    / shadowing
stage2_training    / fading
stage2_training    / actions
fixed_validation   / topology
fixed_validation   / source_updates
fixed_validation   / shadowing
fixed_validation   / fading
fixed_validation   / actions
```

现有训练命名空间如 `nn_topology` 可以保留为兼容别名；新自动配置应统一使用明确的 `stage2_training` 命名空间，或者继续使用旧名称但在 seed manifest 中声明。关键要求是固定、确定、互不混用。

### 5.3 自动和显式模式

验证集支持：

- `scenario_generation.mode: auto`：由 `master_seed + namespace + scenario_index` 自动生成；
- `scenario_generation.mode: explicit`：继续读取当前 `validation.scenarios`；
- 旧 YAML 未写 `scenario_generation` 时，保持当前显式场景行为。

Stage-1 校准和搜索第一版只需实现自动模式；代码结构应允许以后扩展 explicit manifest。

### 5.4 必须保存 seed manifest

输出：

```text
resolved_seed_manifest.json
```

至少记录：

- master seed；
- seed derivation 版本；
- 每个 pool 的命名空间；
- topology/scenario/episode 索引；
- 自动生成的验证场景定义；
- 搜索中的 topology index 和 dynamic repeat index；
- 旧显式模式下的原始 seed 值。

修改 `master_seed` 必须联动改变四个自动集合；同一配置和 master seed 重跑必须得到完全相同 manifest。

---

## 6. 配置结构和新 YAML

创建：

```text
configs/GNN/mpnn_newC_ch1_search.yaml
```

其完整建议内容见本任务附带 YAML 文件。

同时在仓库根目录创建中文说明：

```text
GNN_CONFIG_GUIDE_ZH.md
```

说明新旧字段、搜索配置、随机种子、验证模式、输出文件及常见配置组合。

---

## 7. 代码组织建议

不要把所有逻辑堆进 `trainer.py`。建议新增独立模块，例如：

```text
src/curvature_gossip/learning/stage1_search.py
src/curvature_gossip/learning/seed_plan.py
```

建议职责：

### `seed_plan.py`

- 解析 `master_seed`；
- 定义固定命名空间常量；
- 生成 calibration/search/training/validation 的索引计划；
- 生成并保存 manifest；
- 检查同一 pool 内重复和跨 pool 命名空间误用。

### `stage1_search.py`

- 生成 coarse/refine b candidates；
- 校准公共 center 和候选 beta0；
- 构造复用的搜索场景；
- 运行 curvature-only 候选；
- 汇总约束与指标；
- 选择最佳候选；
- 保存结果。

### `trainer.py`

只负责：

- 规范化配置；
- 调用 seed plan；
- 可选调用 Stage-1 search；
- 将结果注入 resolved config；
- 继续原 Stage-2 PPO 流程。

### `ctde_ppo.py`

Stage-1 base 改为使用已解析的：

```text
calibrated_intercept + alpha * (score - center)
```

若没有 `calibrated_intercept`，兼容模式可使用 `logit(base_tx_ratio)`。

Stage-2 最终概率仍为：

\[
q_{i,t}=\sigma(\operatorname{logit}q_i^{\mathrm{base}}+\delta_{i,t}).
\]

---

## 8. 搜索和训练输出

在实验输出目录保存：

```text
stage1_calibration.json
stage1_search_results.csv
stage1_search_per_scenario.csv
stage1_search_summary.json
resolved_seed_manifest.json
resolved_training_config.yaml
```

### `stage1_calibration.json`

记录：

- calibration pool 配置和索引；
- `c_kappa`；
- 曲率分数统计；
- 每个 `(b, alpha)` 的 `beta0`、实际校准均值和误差。

### `stage1_search_results.csv`

每行一个候选，至少包括：

```text
phase
base_tx_ratio
alpha
calibrated_intercept
calibration_probability_mean
search_probability_mean
actual_tx_ratio
mean_VAoI
mean_tail_VAoI
constraint_feasible
rank
selected
```

### `stage1_search_per_scenario.csv`

每个候选、每个场景一行，保留配对统计所需字段。

### `stage1_search_summary.json`

记录最终：

```text
selected_base_tx_ratio
selected_alpha
selected_intercept
selected_center
selection_metric
constraint_status
```

### `resolved_training_config.yaml`

写入真正用于 Stage-2 的参数：

- `constraints.max_tx_ratio`；
- 最终 `base_tx_ratio`；
- `alpha_override=selected_alpha`；
- `center=selected_center`；
- `calibrated_intercept=selected_intercept`；
- `freeze_stage1=true`；
- 搜索摘要和 seed derivation 版本。

不要修改用户原始 YAML 文件。

---

## 9. 测试检查与调整

### 9.1 现有种子测试

已检查当前 `tests/test_reproducibility.py`：它测试传统实验中相同 `master_seed`、显式 topology/channel/update seed 及不同 policy 顺序应产生相同结果。

要求：

- 保留该测试和旧显式 seed 语义；
- 不要为了 NN 自动 seed 方案删除 `experiment.topology_seeds/channel_seeds/update_seeds` 的旧实验支持；
- 若重构 `make_rng()` 或 seed API，确保该测试继续通过。

### 9.2 当前固定验证

当前 `learning/validation.py`：

- 强制要求 `validation.scenarios` 非空；
- 已使用 `fixed_validation` 命名空间；
- policy action stream 还包含 policy name，以保证不同策略拥有独立但确定的动作流；
- 同一 checkpoint 验证不会修改模型变量。

需要调整：

- `validation_scenarios()` 支持 `auto` 和 `explicit`；
- auto 模式启动时只解析/生成一次固定场景定义；
- 同一训练过程每次验证复用完全相同场景；
- paired NN 与 Stage-1-only 仍使用相同动作流；
- matched/fixed random 的动作流保持确定性且不影响 NN 流；
- auto 模式生成的场景写入 seed manifest。

### 9.3 新增搜索测试

建议新增：

```text
tests/test_stage1_base_search.py
tests/test_seed_plan.py
```

至少覆盖：

1. `Bmax=0.2`、默认参数生成 7 个正确 coarse b；
2. b 候选不超过 Bmax，排序、去重和边界正确；
3. 每个 alpha 独立生成 refine 区间；
4. 边界最优时 refine 不越界；
5. beta0 二分校准后 pooled mean probability 达到目标误差；
6. beta0 是 pool-level 常数，不是 per-topology 值；
7. 同一 master seed 生成相同四池 manifest；
8. 修改 master seed 会改变四个自动池；
9. 不同 pool/process namespace 产生隔离随机流；
10. 候选 `(b, alpha)` 不影响搜索场景 seed；
11. 所有搜索候选使用相同 topology/update/channel/action realization；
12. search disabled 时直接使用 `base_tx_ratio/alpha_override`；
13. 新 `max_tx_ratio/base_tx_ratio` 语义正确；
14. 旧 `target_tx_ratio` YAML 保持可运行；
15. auto validation 在多次 checkpoint 评估中场景不变；
16. explicit validation 与旧行为一致；
17. tiny smoke test 能完成 calibration -> coarse -> refine -> Stage-2 初始化；
18. 输出 CSV/JSON/YAML 和 manifest 字段完整。

### 9.4 更新 Actor 测试

检查并按需更新：

```text
tests/test_stage2_actor.py
tests/test_stage2_mpnn_actor.py
```

新增断言：

- 改变 `max_tx_ratio` 不应在固定 `base_tx_ratio/intercept` 时改变 Stage-1 base probability；
- 改变 `base_tx_ratio/intercept` 应改变 Stage-1 base；
- `delta=0` 时 Stage-2 final probability 等于 Stage-1 base；
- 搜索结果注入后 Stage-1 参数被冻结；
- no-curvature ablation 仍不读取曲率字段。

### 9.5 测试运行要求

- 单元测试默认使用极小 topology count、slots 和候选数，避免 CI 变慢；
- 正式 YAML 的 32/20x2/300/20 规模不得用于常规单元测试；
- 增加一个标记为 slow/integration 的完整搜索测试可选；
- 测试运行产生的 `.pytest_codex_tmp` 和模型 checkpoint 不应提交到 Git。

---

## 10. 验收标准

1. 原命令可直接运行新 YAML；
2. 修改一个 `master_seed` 会联动改变四个自动集合；
3. 同配置同 master seed 完全可复现；
4. `Bmax=0.2` 时 Stage-1 可自动选择约 0.1 等低于上限的工作点；
5. `max_tx_ratio` 不再直接决定 Stage-1 base logit；
6. 每个 alpha 都完成独立 coarse-to-fine b 搜索；
7. 所有候选使用相同搜索随机场景；
8. beta0 是多拓扑校准得到的公共常数，推理不需要全局统计；
9. Stage-1 搜索池、Stage-2 训练池和验证池严格隔离；
10. 搜索结果、resolved config 和 seed manifest 完整保存；
11. 旧 YAML 和旧显式验证场景继续工作；
12. 现有测试通过，新增测试通过，且未引入分布式执行信息泄漏。

---

## 11. 完成后请报告

- 修改文件列表；
- 新增配置字段及兼容规则；
- Stage-1 搜索的实际流程；
- beta0 校准方法和数值容差；
- 四个集合的 namespace 与规模；
- seed manifest 示例；
- 新增输出文件示例；
- 更新/新增测试及结果；
- 是否发现旧配置、验证配对或随机流复现方面的潜在问题。
