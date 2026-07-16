# Curvature-Aware Distributed Gossip Simulator: Codex Implementation Plan

## 0. How Codex should use this document

Implement this plan as a new Python project. Work milestone by milestone, run the relevant tests after each milestone, and do not introduce neural networks, PyTorch, reinforcement learning, mobility, power control, or broadcast-budget optimization in version 1.

When a design choice is not explicitly specified, choose the simplest deterministic implementation that preserves:

1. fixed node positions and fixed distance-based adjacency;
2. distributed per-node broadcast decisions;
3. full-cache gossip and causal slot updates;
4. reproducibility under fixed random seeds;
5. clean interfaces for future topology, curvature, channel, and policy extensions.

The final implementation must include source code, tests, example YAML configurations, a README, and one command that reproduces the first experiment.

---

## 1. Version-1 objective

Build a configurable slotted simulator for a fixed wireless multi-source gossip network:

- There are `N` fixed wireless nodes with 2-D positions.
- The fixed physical communication graph is a unit-disk graph: an undirected edge exists iff the Euclidean distance is no larger than the configured communication radius.
- Every node is also an information source. In every slot, source `s` independently generates a new version with probability `p_update`.
- Every node stores the freshest version it knows for all `N` sources.
- A node independently decides whether to broadcast its complete cache in each slot.
- Nodes are half-duplex. A transmitting node cannot receive in the same slot.
- A silent receiver attempts to decode at most one in-range transmitter. Reception succeeds only when the strongest candidate's SINR exceeds the threshold.
- Successful receivers merge the decoded cache componentwise by maximum version number.
- The primary performance metric is Version Age of Information (VAoI).
- Compute Ollivier-Ricci curvature globally once at initialization. Runtime policies may use only incident edge curvatures and local observations.
- Provide interfaces for future distributed/local curvature estimators, but do not simulate curvature-control messages in version 1.

### Explicit non-goals for version 1

- No neural-network training or inference.
- No centralized scheduler.
- No mobility or topology changes.
- No per-slot curvature recomputation.
- No optimization constraint on the total number of broadcasts.
- No SIC, multi-packet reception, retransmission queue, routing table, or packet fragmentation.
- No assumption that the distributed policy knows the true current versions at remote sources.

---

## 2. Mathematical state and slot chronology

### 2.1 Source versions and node caches

Let `V_s[t]` be source `s`'s current version. At the beginning of slot `t`:

```text
G_s[t] ~ Bernoulli(p_update)
V_s[t] <- V_s[t-1] + G_s[t]
```

Node `i` stores:

```text
cache_versions[i, s]   # freshest version of source s known by node i
cache_gen_slots[i, s]  # generation slot associated with that version
```

The source always immediately knows its own update:

```text
cache_versions[s, s] = V_s
cache_gen_slots[s, s] = t when G_s[t] = 1
```

### 2.2 Version age

For ordered source-holder pair `(i, s)`:

```text
version_age[i, s] = source_versions[s] - cache_versions[i, s]
```

Exclude diagonal pairs `i == s` from network metrics. The diagonal must always be zero.

Also compute time AoI as a secondary diagnostic:

```text
time_age[i, s] = t - cache_gen_slots[i, s]
```

Do not pass `source_versions` or the true `version_age` matrix into a distributed policy. They are simulator/evaluator state, not generally locally observable state.

### 2.3 Required slot order

Use exactly this order:

1. Generate independent source updates and update each source's diagonal cache entry.
2. Freeze a read-only snapshot of all caches that will be carried in this slot's packets.
3. Build one local observation per node from locally available state.
4. Each node independently computes a transmission probability and samples a binary action.
5. Generate the slot fading matrix independently of the actions.
6. Resolve strongest-signal reception and SINR for each silent receiver.
7. Merge decoded packet snapshots into receiver caches simultaneously.
8. Update locally maintained neighbor-cache estimates from overheard packet snapshots.
9. Update metrics and dissemination-delay trackers.

The simultaneous merge in step 7 is mandatory. Information received in slot `t` cannot be forwarded again in the same slot.

---

## 3. Wireless channel and decoding model

### 3.1 Received power

Implement a generic log-distance path-loss model with optional fixed shadowing and per-slot Rayleigh fading.

For transmitter `i` and receiver `j`:

```text
PL_ij_dB = PL0_dB + 10 * pathloss_exponent * log10(max(d_ij, d0) / d0)
           + shadowing_ij_dB
rx_power_ij_mW[t] = tx_power_mW * 10^(-PL_ij_dB / 10) * fading_ij[t]
```

where Rayleigh power fading is `fading_ij[t] ~ Exp(1)`. Shadowing is sampled once per topology/channel realization and held fixed for the whole run. Support an option for reciprocal shadowing.

Noise power:

```text
noise_dBm = noise_psd_dBm_per_Hz
            + 10*log10(bandwidth_Hz)
            + noise_figure_dB
```

### 3.2 Receiver rule

For silent receiver `j`:

1. Candidate transmitters are active nodes `i` satisfying `(i, j)` in the fixed communication graph.
2. Select the candidate with maximum instantaneous received power.
3. Interference is generated by all other active nodes, including active nodes outside `j`'s communication radius.
4. Decode the strongest candidate iff its SINR is at least `sinr_threshold`.
5. A receiver decodes at most one packet per slot.

```text
SINR_i_to_j = rx_power[i, j] /
              (noise_power + sum_{k active, k != i} rx_power[k, j])
```

If `j` transmits, it receives nothing. Do not add ACKs in version 1.

### 3.3 Implementation constraints

- Vectorize received-power and interference calculations with NumPy.
- Generate a full `N x N` fading matrix every slot, regardless of which nodes transmit. This keeps channel randomness aligned across policy comparisons.
- Set diagonal received powers to zero.
- Do not store all fading matrices over time.
- Allow fading and shadowing to be disabled through configuration for deterministic tests.
- Initially accept a direct `sinr_threshold_db`. Leave an optional helper for deriving it from packet length, bandwidth, and slot duration, but this helper need not be the default.

---

## 4. Distributed local knowledge and observations

The simulator has global state, but the policy interface must only receive a `NodeObservation` containing local information.

Each node `i` may know:

- its own complete cache vector `cache_versions[i, :]`;
- its fixed neighbor IDs;
- its incident curvature values `kappa[i, j]`;
- its own time since last attempted transmission;
- the last packet-cache snapshot it successfully overheard from each neighbor;
- whether it transmitted in the previous slot;
- optionally, its previous-slot measured interference/noise-plus-interference.

Each node must maintain:

```text
neighbor_cache_estimate[j, s]
neighbor_estimate_valid[j]
last_tx_slot
previous_interference_power
```

Initialize an unknown neighbor cache estimate to all zeros and mark it invalid. The gain calculation may treat an invalid estimate as zeros, but the observation must retain the validity flag for future policies.

Do not expose any of the following to the heuristic policy:

- other nodes' true current caches;
- the true source-version vector;
- the global VAoI matrix;
- current-slot fading or current-slot actions of other nodes;
- graph-wide curvature statistics at runtime, except normalization constants precomputed at initialization.

---

## 5. Curvature subsystem

### 5.1 Common result and provider interface

Create a common interface similar to:

```python
@dataclass(frozen=True)
class CurvatureResult:
    method: str
    edge_values: dict[tuple[int, int], float]
    node_values: np.ndarray
    metadata: dict[str, Any]

class CurvatureProvider(ABC):
    @abstractmethod
    def compute(self, topology: Topology) -> CurvatureResult:
        ...
```

Always store an undirected edge under canonical key `(min(i, j), max(i, j))`.

The simulator and policies must depend only on `CurvatureResult`, not on a concrete ORC class.

### 5.2 Global Ollivier-Ricci curvature: required implementation

Implement `GlobalORCCurvature` with configurable idleness `alpha`, default `0.5`.

For node `i`:

```text
m_i(i) = alpha
m_i(k) = (1-alpha) / degree(i), k in neighbors(i)
```

For each graph edge `(i, j)`:

1. Build supports `S_i = {i} union N_i` and `S_j = {j} union N_j`.
2. Use unweighted all-pairs graph shortest-path distances as transport costs.
3. Solve the Earth Mover linear program using `scipy.optimize.linprog(method="highs")`.
4. Compute:

```text
kappa_ij = 1 - W1(m_i, m_j) / d_G(i, j)
```

For physical graph edges, `d_G(i, j) = 1`, but retain the denominator in code.

Compute all-pairs shortest-path lengths once per topology. ORC is computed once before simulation slots begin.

Store node curvature as the mean of incident edge curvatures for diagnostics only. The heuristic policy should use edge curvature.

Validate that the topology is connected and that every curvature is finite.

### 5.3 Curvature importance transform

Use the standard negative-curvature bottleneck importance:

```text
raw_b_ij = max(-kappa_ij, 0)
b_ij = raw_b_ij / max_e(raw_b_e) if max_e(raw_b_e) > 0 else 0
```

Make normalization strategy configurable:

- `global_negative_max` (default in version 1);
- `incident_negative_max` (future distributed option);
- `none`.

### 5.4 Required extension interfaces

Add importable but not necessarily operational distributed interfaces:

```python
@dataclass(frozen=True)
class CurvatureMessage:
    sender: int
    receiver: int | None
    kind: str
    payload: Mapping[str, Any]

class DistributedCurvatureAgent(ABC):
    @abstractmethod
    def initialize(self, node_id: int, local_view: LocalTopologyView) -> None: ...
    @abstractmethod
    def create_messages(self) -> list[CurvatureMessage]: ...
    @abstractmethod
    def receive(self, message: CurvatureMessage) -> None: ...
    @abstractmethod
    def is_ready(self) -> bool: ...
    @abstractmethod
    def incident_curvatures(self) -> Mapping[int, float]: ...
```

Add placeholder classes for:

- `DistributedORCAgent`;
- `DistributedAF3Agent`.

They may raise `NotImplementedError` with a clear message in version 1. Do not integrate their control messages into the data channel yet.

### 5.5 Optional low-cost implementation

If time permits after all required milestones, implement global/offline AF3 behind the same `CurvatureProvider` interface:

```text
AF3(i,j) = 4 - degree(i) - degree(j) + 3 * |N_i intersect N_j|
```

This is not the default method, but it verifies that curvature backends are replaceable.

---

## 6. Heuristic distributed policies

### 6.1 Policy interface

Use a node-local probability interface:

```python
class DistributedBroadcastPolicy(ABC):
    @abstractmethod
    def transmission_probability(self, obs: NodeObservation) -> float:
        ...
```

The simulator loops through nodes, obtains one probability per node, clips it to `[0, 1]`, then samples independent Bernoulli actions using a policy-specific RNG stream.

### 6.2 Local edge innovation gain

For node `i` and neighbor `j`, calculate the positive version advantage:

```text
delta_ij[s] = max(cache_i[s] - estimated_cache_j[s], 0)
```

Support configurable aggregation across sources:

- `max` (default, aligned with worst-case VAoI);
- `mean`;
- `top_fraction`, with configurable fraction and at least one selected entry.

Call the result `gain_ij`.

This is a locally estimated potential freshness reduction, not the true reduction at node `j`.

### 6.3 Policies required in version 1

#### A. Uniform random policy

```text
q_i = configured random_tx_probability
```

#### B. Freshness-only policy

```text
edge_score_ij = gain_ij
node_score_i = max_j edge_score_ij + age_weight * time_since_last_tx_i
q_i = q_min + (q_max-q_min) * sigmoid((node_score_i-threshold)/temperature)
```

#### C. Curvature-aware freshness policy

```text
edge_score_ij = (1 + curvature_weight * b_ij) * gain_ij
node_score_i = max_j edge_score_ij + age_weight * time_since_last_tx_i
q_i = q_min + (q_max-q_min) * sigmoid((node_score_i-threshold)/temperature)
```

Use the curvature as a multiplicative gate on innovation. Do not add a standalone positive curvature/bottleneck term that causes a bridge node to broadcast repeatedly when it has no newer information.

Use identical `q_min`, `q_max`, `threshold`, `temperature`, and `age_weight` for B and C unless a configuration explicitly overrides them.

### 6.4 Broadcast count in version 1

Do not enforce an activity budget or add a Lagrange penalty. Always measure and report:

- transmissions per slot;
- per-node activity ratio;
- successful decodes per transmission;
- innovative cache entries delivered per transmission.

Document that a later matched-activity comparison will be required before attributing all performance differences specifically to curvature.

---

## 7. Topology subsystem

### 7.1 Extensible interface and registry

Create:

```python
@dataclass(frozen=True)
class Topology:
    positions: np.ndarray          # shape [N, 2]
    graph: nx.Graph
    communication_radius: float
    node_labels: np.ndarray | None # cluster/region labels
    metadata: dict[str, Any]

class TopologyGenerator(ABC):
    @abstractmethod
    def generate(self, rng: np.random.Generator, params: Mapping[str, Any]) -> Topology:
        ...
```

Use a registry/factory keyed by configuration string. Adding a topology must require only:

1. a new generator class;
2. one registry decorator or registration call;
3. a configuration block.

The simulator and experiment runner must not contain topology-type conditionals.

### 7.2 Universal topology validation

Every generator must validate:

- unique finite positions;
- graph node IDs exactly `0, ..., N-1`;
- graph is undirected, simple, and connected;
- for every pair `(i,j)`, an edge exists iff `distance(i,j) <= communication_radius`, up to a numerical tolerance;
- no self-loops;
- configured minimum/maximum degree constraints, if provided.

If rejection sampling reaches `max_attempts`, raise a descriptive error containing the failed constraints and parameters.

### 7.3 Required topology types

#### A. `random_geometric`

- Uniform positions in a configurable rectangle.
- Edges strictly by distance radius.
- Resample until connected and optional mean-degree range is satisfied.

#### B. `two_cluster_bridge`

- Two spatially separated dense clusters.
- Configurable total node count, cluster sizes, cluster radius/spread, communication radius, center separation, and number of bridge gateway pairs/corridors.
- Non-gateway nodes must be constrained away from the inter-cluster gap so unintended cross-cluster edges are avoided.
- Place gateway pairs deterministically near the inner boundaries, then sample remaining nodes.
- Validate the actual number of inter-cluster edges. Support either exact `bridge_edge_count` or a configured acceptable range.
- Ensure every gateway connects to its own cluster and the full graph remains connected.
- Save binary cluster labels in `node_labels` and list the actual inter-cluster edges in metadata.

Prefer a robust deterministic construction plus validation over fragile unconstrained Gaussian sampling.

#### C. `grid_2d`

- Configurable rows, columns, and spacing.
- Choose or validate a radius that connects horizontal/vertical nearest neighbors. By default exclude diagonal edges.

#### D. `ring`

- Equally spaced positions on a circle.
- Derive or validate a radius that connects the two immediate circular neighbors but not next-nearest neighbors.

### 7.4 Future topology support

Leave room for, but do not implement unless trivial:

- multi-cluster geometric graphs;
- imported positions from CSV;
- imported fixed adjacency with a distance-consistency check;
- UAV corridor or formation topologies.

---

## 8. Metrics and experiment outputs

### 8.1 Per-slot scalar metrics

After the simultaneous cache update, compute over all ordered off-diagonal pairs:

- mean version age;
- maximum version age;
- 95th and 99th percentiles;
- top-5%-mean version age, interpreted as an empirical tail/CVaR-style metric;
- mean time AoI;
- number of transmitters;
- number of successful receiver decodes;
- number of transmitters decoded by at least one receiver;
- number of improved cache entries;
- total version improvement, i.e. sum of positive version increments delivered.

If cluster labels are present, also compute directional cross-cluster mean and maximum VAoI.

### 8.2 Dissemination delay

For every generated `(source, version)`, record its generation slot. A version is complete when all nodes hold that version or a newer one:

```text
min_i cache_versions[i, source] >= version
```

Record completion delay. Later versions are allowed to supersede an older version and thereby complete it.

At run end, report:

- completed-update fraction;
- mean, median, 95th percentile, and maximum delay among completed updates;
- number of right-censored updates that did not complete.

Do not silently treat censored updates as zero delay.

### 8.3 Online aggregation and storage

- Do not store the full `T x N x N` age history.
- Maintain online sums and compact per-slot scalar traces.
- Allow trace sampling/downsampling through configuration.
- Save:

```text
results/<experiment_id>/<topology_seed>/<policy_name>/
    resolved_config.yaml
    summary.json
    per_slot.csv                 # optional/downsampled
    dissemination_delays.csv
    edge_curvature.csv
    topology_nodes.csv
    topology_edges.csv
```

- Never overwrite an existing run directory unless `--overwrite` is explicitly passed.

### 8.4 Required plots

Provide plotting functions or a plotting CLI for:

1. topology with edges colored by ORC and bridge edges highlighted;
2. histogram of edge curvature;
3. smoothed maximum-VAoI trace, with raw data retained and smoothing window stated;
4. empirical CDF of instantaneous maximum VAoI;
5. comparison of mean/max/tail VAoI across policies;
6. VAoI versus average transmissions per slot.

Plotting must be separate from simulation logic.

---

## 9. Reproducibility and fair policy comparison

Use `numpy.random.SeedSequence` to create independent RNG streams for:

- topology positions;
- static shadowing;
- source updates;
- per-slot fading;
- each policy's Bernoulli action sampling.

For a fixed topology/channel/update seed tuple:

- every policy receives the same topology and shadowing realization;
- every policy receives the same source-update realization;
- every policy receives the same full fading matrix in each slot;
- only policy-action randomness differs.

Do not make random-number consumption depend on the number of active nodes. Generate all `N` source-update draws and the full `N x N` fading matrix every slot.

The experiment runner should run all configured policies on paired exogenous seeds and aggregate across seeds. Report mean and 95% confidence intervals across independent run/topology seeds, not across correlated time slots.

---

## 10. Proposed project structure

```text
curvature-gossip/
├── pyproject.toml
├── README.md
├── configs/
│   ├── first_experiment.yaml
│   ├── smoke_test.yaml
│   ├── random_geometric.yaml
│   └── topology_sweep.yaml
├── src/curvature_gossip/
│   ├── __init__.py
│   ├── config.py
│   ├── random_streams.py
│   ├── topology/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── registry.py
│   │   ├── validation.py
│   │   ├── random_geometric.py
│   │   ├── two_cluster_bridge.py
│   │   ├── grid.py
│   │   └── ring.py
│   ├── curvature/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── global_orc.py
│   │   ├── af3.py
│   │   └── distributed/
│   │       ├── __init__.py
│   │       ├── base.py
│   │       ├── orc_agent.py
│   │       └── af3_agent.py
│   ├── channel/
│   │   ├── __init__.py
│   │   ├── propagation.py
│   │   └── decoder.py
│   ├── state/
│   │   ├── __init__.py
│   │   ├── version_state.py
│   │   └── local_knowledge.py
│   ├── policies/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── random_policy.py
│   │   ├── freshness_policy.py
│   │   └── curvature_freshness_policy.py
│   ├── simulator/
│   │   ├── __init__.py
│   │   ├── observations.py
│   │   └── engine.py
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── collector.py
│   │   ├── dissemination.py
│   │   └── aggregation.py
│   ├── experiments/
│   │   ├── __init__.py
│   │   ├── runner.py
│   │   └── sweep.py
│   ├── plotting/
│   │   ├── __init__.py
│   │   └── figures.py
│   └── cli.py
└── tests/
    ├── test_topologies.py
    ├── test_orc.py
    ├── test_channel.py
    ├── test_version_state.py
    ├── test_policy_locality.py
    ├── test_slot_causality.py
    ├── test_reproducibility.py
    └── test_smoke_experiment.py
```

Keep modules small and avoid circular imports. Use dataclasses and type hints. Avoid a framework-heavy dependency-injection system.

### Dependencies

Use a minimal set:

- Python `>=3.11`;
- NumPy;
- SciPy;
- NetworkX;
- pandas;
- PyYAML;
- matplotlib;
- seaborn;
- pytest.

Do not require `GraphRicciCurvature`, POT, PyTorch, Gymnasium, Ray, or an RL library in version 1.

---

## 11. Example configuration schema

Create a working `configs/first_experiment.yaml` similar to:

```yaml
experiment:
  id: first_orc_heuristic
  slots: 100000
  warmup_slots: 10000
  trace_stride: 10
  master_seed: 20260713
  topology_seeds: [0, 1, 2, 3, 4]
  channel_seeds: [0]
  update_seeds: [0]

topology:
  type: two_cluster_bridge
  params:
    n_nodes: 40
    area_width_m: 500.0
    area_height_m: 500.0
    communication_radius_m: 110.0
    bridge_edge_count: 1
    max_attempts: 500

source:
  update_probability: 0.05

channel:
  tx_power_dbm: 20.0
  pathloss_reference_db: 40.0
  reference_distance_m: 1.0
  pathloss_exponent: 3.0
  shadowing_std_db: 4.0
  reciprocal_shadowing: true
  rayleigh_fading: true
  noise_psd_dbm_per_hz: -174.0
  bandwidth_hz: 5000000.0
  noise_figure_db: 7.0
  sinr_threshold_db: 0.0

curvature:
  method: global_orc
  alpha: 0.5
  normalization: global_negative_max

policies:
  - name: random
    type: random
    params:
      tx_probability: 0.1

  - name: freshness_only
    type: freshness
    params:
      source_aggregation: max
      q_min: 0.01
      q_max: 0.5
      threshold: 1.0
      temperature: 1.0
      age_weight: 0.02

  - name: orc_freshness
    type: curvature_freshness
    params:
      source_aggregation: max
      curvature_weight: 1.0
      q_min: 0.01
      q_max: 0.5
      threshold: 1.0
      temperature: 1.0
      age_weight: 0.02

output:
  root: results
  save_per_slot: true
  make_plots: true
```

If this precise geometry cannot realize one bridge edge under the implemented deterministic generator, adjust the default cluster-layout parameters in the final working config rather than weakening topology validation.

Also provide a fast smoke configuration with approximately `N=12`, `slots=200`, no fading, and one seed.

---

## 12. CLI requirements

At minimum support:

```bash
python -m curvature_gossip.cli run --config configs/smoke_test.yaml
python -m curvature_gossip.cli run --config configs/first_experiment.yaml
python -m curvature_gossip.cli plot --results results/first_orc_heuristic
```

The run command must:

1. validate and resolve the configuration;
2. generate each topology once per topology seed;
3. compute curvature once per topology;
4. run all policies with paired exogenous randomness;
5. save per-run outputs;
6. write an aggregate policy-comparison CSV/JSON;
7. optionally generate plots.

Print concise progress and a final table containing at least:

```text
policy | mean_VAoI | mean_max_VAoI | p95_max_VAoI |
avg_tx_per_slot | successful_decodes_per_tx
```

---

## 13. Milestones and implementation order

### Milestone 1: skeleton, configuration, and topology registry

- Create package, dependencies, CLI skeleton, dataclasses, YAML loading, and validation.
- Implement topology base class, registry, universal validator, and all four required generators.
- Add topology plots and topology tests.

Exit condition: every topology is reproducible, connected, and exactly distance-consistent.

### Milestone 2: version state and deterministic cache propagation

- Implement source updates, cache snapshots, componentwise merge, local neighbor-cache estimates, and dissemination tracking.
- Add tests for diagonal freshness, monotonic cache versions, supersession, and no same-slot multi-hop cascade.

Exit condition: a deterministic small graph produces hand-verifiable cache trajectories.

### Milestone 3: channel and receiver

- Implement path loss, shadowing, fading, noise conversion, half-duplex, strongest-candidate selection, SINR, and at-most-one decode.
- Add zero-interference, collision, half-duplex, range, and reproducibility tests.

Exit condition: channel tests cover all receiver rules without using the simulator engine.

### Milestone 4: global ORC and curvature interface

- Implement `CurvatureProvider`, `CurvatureResult`, exact global ORC, negative-part normalization, CSV export, and topology curvature plot.
- Add distributed curvature interface stubs.
- Optionally implement AF3 only after ORC tests pass.

Exit condition: ORC values are finite and symmetric; bridge edges in a controlled two-cluster graph are among the most negative edges.

### Milestone 5: local observations and heuristic policies

- Implement local observation builder with an explicit whitelist of fields.
- Implement random, freshness-only, and ORC-freshness policies.
- Add tests proving policies do not receive true global versions/VAoI.

Exit condition: each node chooses from only its local observation and incident curvature.

### Milestone 6: simulator engine and online metrics

- Integrate the exact slot chronology.
- Add per-slot metrics, online summaries, dissemination delay, and cluster-directional metrics.
- Avoid storing full state history.

Exit condition: smoke simulation runs end-to-end and all invariant tests pass.

### Milestone 7: paired multi-policy experiment runner

- Implement independent RNG streams and paired exogenous randomness.
- Run all policies over configured seeds.
- Save resolved configs, summaries, traces, topology, and curvature.
- Aggregate means and 95% CIs across independent seeds.

Exit condition: rerunning the same command produces identical numerical outputs.

### Milestone 8: first executable experiment and documentation

- Finalize `first_experiment.yaml` for `N=40`, `p_update=0.05`, and a two-cluster one-bridge topology.
- Add a sweep configuration over bridge counts `{1, 2, 4}`, update probabilities `{0.01, 0.05, 0.1, 0.2}`, and topology types.
- Document model equations, slot order, locality assumptions, configuration options, output schema, and extension instructions.

Exit condition: README commands reproduce the first comparison and plots.

---

## 14. Required tests and invariants

### Topology

- Every returned edge satisfies `distance <= R + tolerance`.
- Every non-edge satisfies `distance > R - tolerance`.
- Graph is connected and simple.
- Same seed and config return identical positions and edges.
- `two_cluster_bridge` satisfies its actual inter-cluster edge constraint.

### State and age

- Cache versions never decrease.
- `cache_versions[i, s] <= source_versions[s]` always.
- Diagonal version age is always zero.
- All off-diagonal version ages are nonnegative.
- With `p_update=0`, VAoI remains zero from the all-zero initialization.
- A later version can complete dissemination of an earlier version.

### Slot causality

Use a three-node line. If only node 0 initially has a new version and nodes 0 and 1 transmit in the same slot, node 2 must not receive node 0's version via node 1 in that slot.

### Channel

- Transmitting receivers decode nothing.
- No in-range active transmitter means no decode.
- With fading/shadowing disabled and no interference, success agrees with the SNR threshold.
- A strong interferer can cause decoding failure.
- At most one transmitter is decoded per receiver.
- Out-of-range transmitters cannot be decoded but still contribute interference.

### ORC

- Edge-key canonicalization is correct.
- Curvature is finite for every edge.
- Reversing edge endpoint order gives the same result.
- A controlled bridge edge is more negative than representative dense intra-cluster edges.
- Normalized bottleneck importance lies in `[0, 1]`.

### Policy locality

- `NodeObservation` has no global source-version or global age field.
- A policy cannot access mutable simulator arrays through an observation reference.
- ORC-freshness with `curvature_weight=0` returns the same probabilities as freshness-only under identical observations.
- If all gains and `age_weight` are zero, probabilities reduce to the configured sigmoid baseline.

### Reproducibility

- Same full seed tuple produces byte-identical or numerically identical summaries.
- Changing only the policy RNG does not change topology, source updates, shadowing, or fading.
- Policy ordering in the configuration does not change a policy's exogenous realization.

---

## 15. Definition of done

Version 1 is complete only when all of the following hold:

- `pytest` passes.
- The smoke command completes without manual edits.
- The first experiment runs all three policies on the same topology/update/channel realizations.
- Global ORC is computed only once per topology, before slot simulation.
- Runtime policy decisions are node-local and independent.
- Cache delivery is simultaneous and causal.
- Primary VAoI, tail VAoI, dissemination delay, and transmission statistics are saved.
- At least random geometric, two-cluster bridge, grid, and ring topologies work through the same configuration/registry interface.
- Distributed curvature interfaces exist without affecting version-1 execution.
- README clearly states that version 1 has no broadcast-count constraint and therefore reports, but does not match, activity across policies.
- Adding a new topology or curvature backend does not require modifying the simulator engine.

---

## 16. Deferred follow-up work

Do not implement these in version 1, but preserve compatibility:

1. matched-broadcast-load calibration between policies;
2. PRR-weighted or distance-weighted ORC;
3. distributed AF3/Balanced-Forman/ORC control-message simulation;
4. slow-timescale curvature updates;
5. ACK or neighbor digest protocols;
6. learned parameter-shared local actor with centralized training;
7. variable packet size and Shannon-derived SINR threshold as `N` scales;
8. topology mobility and dynamic neighbor discovery;
9. multi-channel access, power control, and SIC;
10. imported UAV trajectories or measurement traces.

