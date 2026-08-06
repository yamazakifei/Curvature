# CODEX_STAGE1_BASE_SEARCH_MODIFICATION_PLAN_0805V3.2

## 0. 任务目标

请在当前 `GNN_Actor` 分支的实现基础上完成 Stage-1 curvature-only 基础概率搜索 V3.2。目标有三项：

1. 显著降低 Stage-1 搜索耗时；
2. 新版搜索使用无中心形式，删除 `c_kappa`：
   \[
   q_i^{\mathrm{base}}=\sigma(\beta_0+\alpha_\kappa s_i);
   \]
3. 保持已有带非零 `c_kappa` 的 YAML、已训练 checkpoint 和旧实验的兼容性。

同时新增配置文件：

```text
configs/GNN/mpnnV3.2_ch1_search.yaml
```

不要覆盖或直接修改现有：

```text
configs/GNN/mpnn_newC_ch1_search.yaml
```

训练入口保持不变：

```bash
python scripts/train_nn_ctde.py --config configs/GNN/mpnnV3.2_ch1_search.yaml
```

---

## 1. 当前实现与主要耗时来源

当前主要文件：

```text
src/curvature_gossip/learning/stage1_search.py
src/curvature_gossip/learning/trainer.py
src/curvature_gossip/learning/ctde_ppo.py
src/curvature_gossip/learning/seed_plan.py
```

当前 `stage1_search.py` 的搜索流程为：

1. 在 calibration pool 上生成多张拓扑并计算 AF3/importance/scores；
2. 对全部 coarse `(b, alpha)` 进行短集合评估；
3. 为每个 alpha 生成局部 refine `b`；
4. 将全部 coarse 候选和全部 refine 候选再次放到 full scenarios 上运行；
5. 每个候选、每个场景都会重新生成相同 topology、重新计算 AF3、importance 和传播模型。

当前示例配置下：

- coarse `b` 数量：7；
- alpha 数量：5；
- coarse scenarios：10 topology × 1 repeat；
- full scenarios：20 topology × 2 repeats；
- full slots：500。

因此原流程通常需要约百万量级的动态仿真时隙，并且同一 topology 的 AF3/importance 被每个候选重复计算。

`beta0` 的二分校准只是在静态 score 数组上进行 sigmoid/mean 运算，不是主要瓶颈。不要优先优化二分算法。

---

## 2. V3.2 数学定义：搜索时删除中心项

### 2.1 新版搜索公式

新版 V3.2 搜索使用：

\[
q_i^{\mathrm{base}}
=\sigma\left(\beta_0+\alpha_\kappa s_i\right),
\]

其中：

- `s_i`：当前 `incident_bottleneck_max`；
- `alpha_kappa`：来自 YAML 的显式候选列表；
- `beta0`：针对每个 `(b, alpha)` 在 calibration pool 上单独二分校准；
- 不再计算或使用 `c_kappa`。

每个候选的校准目标保持：

\[
\frac{1}{M}
\sum_{m=1}^{M}
\sigma\left(\beta_0+\alpha_\kappa s_m\right)
=b.
\]

### 2.2 为什么可以删除中心项

旧公式为：

\[
\sigma\left(\beta_0+\alpha(s-c)\right)
=\sigma\left((\beta_0-\alpha c)+\alpha s\right).
\]

因此，在 `beta0` 会重新校准的搜索模式下，`c_kappa` 与截距完全冗余。删除中心项不降低第一层表达能力，只会改变校准后截距的数值。

---

## 3. `c_kappa` 兼容策略

不能直接删除 `actor.curvature.center` 字段，也不能修改已有 TensorFlow variable 名称或 checkpoint shape。

### 3.1 新增显式字段

支持：

```yaml
actor:
  curvature:
    center_mode: none
    center: 0.0
```

允许值：

```text
none
fixed
calibration_mean
```

语义：

- `none`：解析后的 `center = 0.0`；
- `fixed`：要求 `center` 为有限数值并直接使用；
- `calibration_mean`：使用 calibration/training topology scores 的 pooled mean；这是旧 `auto` 行为。

### 3.2 旧配置回退规则

当 YAML 没有 `center_mode` 时，必须保持旧行为：

1. `center` 是有限数值：等价于 `center_mode: fixed`；
2. `center: auto`、`center: null` 或字段缺省：等价于 `center_mode: calibration_mean`。

这样旧 YAML 使用原文件重新运行时仍能获得原来的中心处理。

### 3.3 新 V3.2 配置规则

新文件 `mpnnV3.2_ch1_search.yaml` 必须显式设置：

```yaml
center_mode: none
center: 0.0
```

搜索结果中仍可以保存：

```json
{
  "selected_center_mode": "none",
  "selected_center": 0.0
}
```

保留 `center: 0.0` 是为了让现有 Actor 图构建代码继续接收数值，不是重新引入中心项。

### 3.4 checkpoint 兼容要求

必须满足：

- 不修改 `actor/alpha_raw` 等已有 trainable variable 名称；
- 不修改已有 MPNN/MLP variable shape；
- `center_mode` 只参与配置解析，传入 `CTDEPPO` 前仍将 `actor.curvature.center` 解析为有限数值；
- 旧 checkpoint 配合其原始 `training_config.yaml` 或原始 actor metadata 加载时，前向概率保持不变；
- 旧配置中的非零 `center` 仍参与：
  \[
  \beta_0+\alpha(s-c).
  \]

旧 checkpoint 如果以 `restore_actor_only: true` 加载到新 V3.2 YAML，使用新 YAML 的 `center=0` 是有意的迁移初始化，不应伪装成严格 resume。

若项目已有 full-resume 语义，建议增加检查：

- full resume 时 checkpoint actor metadata 的 `center`/`center_mode` 必须和当前配置一致；
- 不一致时抛出清晰错误；
- actor-only restore 允许不一致，但输出 warning。

---

## 4. 搜索速度优化：四个优先级全部实现

## 4.1 优先级一：缓存静态 topology/scenario 数据

### 4.1.1 不得缓存可变 simulator

`GossipSimulator` 内含可变对象：

- `VersionState`；
- `LocalKnowledge`；
- metrics/tracker；
- update/fading/action RNG；
- pending step state。

因此不能跨候选复用同一个 simulator 实例。

### 4.1.2 新增不可变缓存结构

建议在 `stage1_search.py` 增加：

```python
@dataclass(frozen=True)
class SearchTopologyCache:
    topology_index: int
    topology: Any
    curvature: Any
    importance: Mapping
    scores: np.ndarray


@dataclass(frozen=True)
class SearchScenarioCache:
    topology_index: int
    dynamic_repeat_index: int
    topology_cache: SearchTopologyCache
    propagation: PropagationModel
```

或者使用等价命名。

缓存内容至少包括：

- topology；
- curvature result；
- bottleneck importance；
- node score vector；
- positions；
- `PropagationModel`，包括固定 shadowing 和 `mean_rx_power_mw`。

`PropagationModel.received_power()` 只读取静态矩阵并使用外部 `fading_rng`，因此可在串行候选之间复用同一个 propagation 对象。

### 4.1.3 候选评估时只重建动态状态

新增类似：

```python
def build_simulator_from_cache(
    raw,
    scenario_cache,
    max_tx_ratio,
    slots,
):
    ...
```

每次调用必须创建全新的：

- `GossipSimulator`；
- source update RNG；
- fading RNG；
- action RNG；
- state/knowledge/metrics/tracker。

RNG 仍使用：

```python
make_rng(master_seed, "stage1_search", process, topology_index, repeat)
```

候选 `(b, alpha)` 绝不能进入随机种子。

这样不同候选仍共享相同外生 realization 和相同 uniform action random sequence，保持 paired comparison。

### 4.1.4 缓存构建次数

串行模式下，一次搜索中：

- calibration topology：每张只计算一次 AF3；
- search topology：每张只计算一次 AF3；
- 每个 `(topology, repeat)` 只构造一次固定 shadowing/PropagationModel。

不得随着候选数量增加而重复计算 AF3。

---

## 4.2 优先级二：减少进入 full evaluation 的候选

当前所有 coarse 候选都会再次进入 full evaluation。修改为：

1. 在 coarse scenarios 上评估全部 coarse `(b, alpha)`；
2. 对每个 alpha 单独排序 coarse 候选；
3. 每个 alpha 只保留 coarse top-K，默认 `K=2`；
4. 每个 alpha 仍围绕其 coarse 最优 `b` 生成 refine 候选；
5. full candidate set 为：
   - 每个 alpha 的 coarse top-K；
   - 该 alpha 所有去重后的 refine 候选；
6. 其他 coarse 候选标记为 `pruned_after_coarse=true`，不运行 full scenarios。

新增配置：

```yaml
candidate_selection:
  coarse_keep_top_k_per_alpha: 2
```

约束：

- K 必须为正整数；
- K 大于 coarse candidate 数量时自动截断；
- coarse 排序仍遵循现有可行性优先和 VAoI 规则；
- refine 中与保留 coarse 点重复的 `(b, alpha)` 必须去重；
- 最终最优候选只能从 full-evaluated candidates 中选择。

输出 CSV 中增加：

```text
coarse_rank_within_alpha
evaluated_on_full
pruned_after_coarse
```

`stage1_search_results.csv` 必须同时保留 coarse 结果和 full 结果，不能因为剪枝丢失可审计性。

---

## 4.3 优先级三：粗搜索使用更短 slots

新增配置：

```yaml
evaluation:
  coarse_slots: 200
  full_slots: 500
```

兼容规则：

- `coarse_slots` 缺省时回退到旧 `evaluation.slots`；
- `full_slots` 缺省时回退到旧 `evaluation.slots`；
- 若旧 `evaluation.slots` 也缺省，则回退到 `experiment.slots`。

coarse 和 full 必须从相同 seed 的时隙 0 开始：

- coarse 使用前 200 个随机数序列；
- full 重新初始化相同 seed，并使用前 500 个随机数序列。

不得从 coarse simulator 的末状态继续 full，否则不同候选的 full comparison 不再是独立、可复现的完整 rollout。

输出文件中记录：

```text
phase_slots
```

---

## 4.4 优先级四：进程级并行候选评估

### 4.4.1 配置

新增：

```yaml
parallel:
  enabled: true
  backend: process
  workers: 0
  chunksize: 1
  deterministic_order: true
```

语义：

- `workers: 0`：自动选择，建议 `min(os.cpu_count(), candidate_count)`，并允许实现设置合理上限；
- `workers: 1` 或 `enabled: false`：严格串行；
- 只支持 `process`，不要使用线程池作为 CPU-bound 默认实现。

### 4.4.2 任务粒度

使用“每个候选一个任务”，不要使用“每个候选 × 每个场景一个任务”。

每个 worker：

1. 在 initializer 或首次任务时建立完整 search scenario cache；
2. 后续在该 worker 接收的多个候选间复用 topology/AF3/propagation cache；
3. 每个候选仍创建独立 simulator 和 RNG。

Windows 使用 spawn 时，worker 初始化函数、候选评估函数和 dataclass 必须定义在模块顶层，不能使用 lambda、闭包或局部函数。

### 4.4.3 确定性

必须保证：

- 串行和并行使用完全相同的 RNG namespace/index；
- candidate ID 不进入 RNG；
- worker PID、执行顺序和完成顺序不进入 RNG；
- CSV/JSON 由父进程统一写入；
- 结果按显式 `candidate_id` 或输入顺序稳定排序；
- 相同 `master_seed` 下，串行和并行的数值结果完全一致，至少在浮点容差内一致。

建议使用 `ProcessPoolExecutor.map()` 或对 futures 结果做显式排序。

### 4.4.4 异常处理

worker 异常必须在父进程重新抛出，并附带：

```text
phase
candidate_id
b
alpha
```

不要静默跳过失败候选。

---

## 5. 建议的 V3.2 搜索流程

伪代码：

```python
def run_stage1_search(raw, output_directory):
    config = parse_search_config(raw)
    center_mode = resolve_center_mode(raw["actor"]["curvature"])

    calibration_cache = build_calibration_topology_cache(...)
    scores = concatenate(cache.scores for cache in calibration_cache)
    center = resolve_center_value(center_mode, scores)  # V3.2 = 0.0

    coarse_candidates = []
    for alpha in alpha_candidates:
        for b in coarse_b_candidates:
            beta0 = calibrate_intercept(scores, b, alpha, center)
            coarse_candidates.append(...)

    full_scenario_cache = build_search_scenario_cache(
        topology_count=full_topology_count,
        dynamic_repeats=full_dynamic_repeats,
    )
    coarse_scenarios = subset(full_scenario_cache, coarse counts)

    coarse_results = evaluate_candidates(
        coarse_candidates,
        coarse_scenarios,
        slots=coarse_slots,
        parallel=parallel_config,
    )

    kept_coarse = keep_top_k_per_alpha(coarse_results, K)
    refine_candidates = build_refine_candidates_from_best_per_alpha(...)
    full_candidates = unique(kept_coarse + refine_candidates)

    full_results = evaluate_candidates(
        full_candidates,
        full_scenario_cache,
        slots=full_slots,
        parallel=parallel_config,
    )

    selected = select_candidate(full_results, ...)
    write_outputs(...)
    return resolved_actor_fields(...)
```

注意：若并行实现采用 worker-local cache，父进程不必把大对象 cache 逐任务 pickle；但同一个 worker 不能为每个候选重新构建 cache。

---

## 6. 配置解析与默认值

新 V3.2 YAML 使用：

```yaml
actor:
  curvature:
    center_mode: none
    center: 0.0
    base_search:
      schema_version: v3_2
      enabled: true
      evaluation:
        coarse_slots: 200
        full_slots: 500
        candidate_selection:
          coarse_keep_top_k_per_alpha: 2
        cache:
          enabled: true
        parallel:
          enabled: true
          backend: process
          workers: 0
          chunksize: 1
          deterministic_order: true
```

默认建议：

```text
center_mode = legacy resolver when omitted
cache.enabled = true
coarse_keep_top_k_per_alpha = 2
parallel.enabled = false for old configs
parallel.workers = 1 when disabled
coarse_slots/full_slots = legacy evaluation.slots
```

旧 YAML 不包含新字段时，应能继续运行且结果语义不变。

---

## 7. `trainer.py` 修改要求

当前流程在 `run_stage1_search()` 返回结果后，还会调用 `_freeze_stage1_center()`。

修改时要避免：

- V3.2 搜索已经解析 `center=0` 后，训练器再次计算 auto center；
- 搜索返回的 `center_mode` 被丢失；
- 旧非搜索模型无法使用原来的 auto/fixed center。

推荐逻辑：

```python
search_result = run_stage1_search(...)
actor_config = _stage1_actor_config(raw)

if search_result:
    merge search_result into actor.curvature
    # search_result 必须包含已解析的有限 center 数值
    # 不再运行 _freeze_stage1_center
elif curvature enabled:
    center = resolve/freeze legacy or explicit center
```

即：

```text
搜索启用且返回结果 -> 使用搜索已经解析的 center
搜索关闭 -> 使用 _freeze_stage1_center 的兼容逻辑
```

建议将 `_freeze_stage1_center()` 改名为更准确的：

```python
_resolve_stage1_center_for_training(...)
```

但若重命名会影响测试，可以保留函数名并更新实现。

---

## 8. `ctde_ppo.py` 修改要求

`CTDEPPO._build_curvature_base()` 继续只接受数值：

```python
center = float(curvature.get("center", 0.0))
```

不要让 TensorFlow 图直接解析字符串 `none/auto`。

公式继续支持两种已解析配置：

旧模型：

\[
\mathrm{base\_logit}+\alpha(s-c).
\]

新 V3.2：

\[
\mathrm{calibrated\_intercept}+\alpha s,
\]

其中 `center=0.0`。

不要删除 `center` 相关前向代码，因为旧 checkpoint 仍需要它。

---

## 9. 输出文件调整

继续生成：

```text
stage1_calibration.json
stage1_search_results.csv
stage1_search_per_scenario.csv
stage1_search_summary.json
resolved_training_config.yaml
resolved_seed_manifest.json
```

新增：

```text
stage1_search_runtime.json
```

建议内容：

```json
{
  "schema_version": "v3_2",
  "center_mode": "none",
  "calibration_topology_count": 32,
  "search_topology_count": 20,
  "dynamic_repeats": 2,
  "coarse_candidate_count": 35,
  "coarse_slots": 200,
  "coarse_simulation_count": 350,
  "full_candidate_count": 20,
  "full_slots": 500,
  "full_simulation_count": 800,
  "cache_enabled": true,
  "parallel_enabled": true,
  "parallel_workers": 4,
  "timing_seconds": {
    "calibration_cache": 0.0,
    "search_cache": 0.0,
    "coarse_evaluation": 0.0,
    "full_evaluation": 0.0,
    "total": 0.0
  }
}
```

数值以真实运行结果填写。

`stage1_search_summary.json` 增加：

```text
schema_version
selected_center_mode
selected_center
coarse_candidate_count
full_candidate_count
```

对于 V3.2：

```text
selected_center_mode = none
selected_center = 0.0
```

---

## 10. 测试要求

新增测试文件建议：

```text
tests/test_stage1_search_v32.py
```

并按需要更新已有 Stage-1/Trainer 测试。

### 10.1 无中心搜索测试

验证：

```yaml
center_mode: none
center: 0.0
```

得到：

```text
search_result["center"] == 0.0
search_result["center_mode"] == "none"
```

候选概率严格等于：

```python
sigmoid(beta0 + alpha * scores)
```

### 10.2 旧数值中心兼容测试

构造旧 actor config：

```yaml
center: 0.4
```

不提供 `center_mode`，验证前向结果仍为：

```python
sigmoid(base_logit + alpha * (scores - 0.4))
```

### 10.3 旧 auto 兼容测试

不提供 `center_mode`，使用：

```yaml
center: auto
```

验证仍解析为 calibration mean，而不是 0。

### 10.4 参数化等价测试

在同一 score 数组上：

- 旧中心校准得到 `(beta_old, c)`；
- 新无中心校准得到 `beta_new`。

验证：

\[
\beta_{new}\approx\beta_{old}-\alpha c
\]

以及所有节点概率相等。

### 10.5 缓存调用次数测试

monkeypatch curvature provider 的 `compute()` 计数。

串行搜索中，搜索 topology 的 curvature compute 次数必须与唯一 topology 数量成正比，而不是与候选数量相乘。

### 10.6 candidate pruning 测试

例如 3 个 alpha、7 个 coarse b、K=2：

- coarse 评估 21 个候选；
- 每个 alpha 只有 top-2 coarse 进入 full；
- refine 候选正常加入并去重；
- 被剪枝候选具有 `pruned_after_coarse=true`。

### 10.7 coarse/full slots 测试

验证 coarse simulator 使用 `coarse_slots`，full simulator 使用 `full_slots`，并且二者分别从相同 seed 的 slot 0 开始。

### 10.8 串行/并行确定性测试

同一个小型配置分别运行：

```yaml
parallel.enabled: false
```

和：

```yaml
parallel.enabled: true
parallel.workers: 2
```

验证：

- selected `(b, alpha, beta0)` 相同；
- 每个候选每个场景的 mean VAoI、actual tx ratio 相同；
- 输出排序相同；
- seed manifest 不因 workers 改变。

### 10.9 checkpoint 结构兼容测试

验证修改前后 trainable variable 名称和 shape 不因 `center_mode` 增加而改变。

用带数值 center 的旧测试 checkpoint/config 恢复后，概率与修改前一致。

### 10.10 错误配置测试

以下配置必须明确报错：

```text
center_mode: fixed 但 center 缺失/非数值
未知 center_mode
coarse_slots <= 0
full_slots <= 0
coarse_keep_top_k_per_alpha <= 0
parallel.backend != process
workers < 0
```

---

## 11. 性能验收标准

以新 YAML 默认候选为例：

原动态仿真规模约为：

```text
coarse: 35 × 10 × 500 = 175000 slots
full:   约45~50 × 40 × 500 = 900000~1000000 slots
```

V3.2 预期约为：

```text
coarse: 35 × 10 × 200 = 70000 slots
full:   约20 × 40 × 500 = 400000 slots
```

总动态 slots 预期下降到约 47 万，约为原来的 40%–45%。

此外，串行缓存后 AF3 计算应从“候选数 × topology 数”下降为：

```text
calibration_topology_count + search_topology_count
```

并行 worker-local cache 允许每个 worker 各构建一次 cache，但禁止每个候选重复构建。

验收时至少报告：

- 修改前后 wall-clock；
- coarse/full candidate 数；
- simulator run 数；
- AF3 compute 调用数；
- cache build 次数；
- workers 数；
- 最终 selected 参数是否与关闭优化、使用等价搜索集合时一致。

性能优化不得改变随机流、公平配对规则和最终评价公式。

---

## 12. 新 YAML 文件要求

新增：

```text
configs/GNN/mpnnV3.2_ch1_search.yaml
```

该文件应基于当前 `mpnn_newC_ch1_search.yaml` 的 topology、channel、Actor、Critic 和 PPO 配置创建，但修改：

1. experiment id/description 改为 V3.2；
2. `center_mode: none`；
3. `center: 0.0`；
4. 搜索 schema 标记为 `v3_2`；
5. `coarse_slots: 200`；
6. `full_slots: 500`；
7. `coarse_keep_top_k_per_alpha: 2`；
8. cache 开启；
9. process parallel 开启，`workers: 0` 自动；
10. 不覆盖旧 YAML。

---

## 13. 完成定义

Codex 完成后必须满足：

- 新 YAML 可被 `yaml.safe_load` 正常解析；
- 新命令可运行完整 Stage-1 搜索后继续 Stage-2 PPO；
- V3.2 搜索输出 `center=0.0`；
- `beta0` 对每个 `(b, alpha)` 正常校准；
- 搜索场景静态数据不再按候选重复计算；
- full 阶段不再重新评估所有 coarse 候选；
- coarse/full slots 分离；
- 串行/并行结果可复现且一致；
- 旧 YAML 与旧 checkpoint 仍可按原始中心配置运行；
- 全部现有测试和新增测试通过；
- 运行时输出足够的信息证明优化实际生效。
