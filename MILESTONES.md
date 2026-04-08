# Throughput Predictor Refactoring - Milestones

**Goal**: Redesign `ssir/pathfinder/rl/` into a three-stage supervised learning pipeline:
1. **Data Collection**: Generate episodes with ground-truth candidate evaluations (offline)
2. **Train NN**: Supervised throughput prediction model (offline)
3. **Inference**: Use trained model to replace expensive `compute_network_throughput()` calls (online)

**Key Design Constraints**:
- No caching (graph-dependent, skip complexity)
- Pickle for dataset serialization
- NN predicts throughput (scalar) directly
- Wide dynamic range: Mbps (leaf users, few downstream) → Kbps (near-source, many users)
- No RL structure; pure supervised learning

---

## Phase 1: Data Collection for Throughput Predictor

### M1.1 - Data Collection Module Structure ✅ COMPLETE
**Status**: Done

Define core data schema for training data.

**Deliverables**:
- `DataEntry` dataclass: episode metadata, graph state, candidate routes, true throughputs, selected route index
- `EpisodeDataset` dataclass: episode wrapper with stats (min/max/mean throughput)
- Save/load functions (pickle-based)
- Input validation for consistency

**Files**:
- `ssir/pathfinder/data_collection/__init__.py`
- `ssir/pathfinder/data_collection/data_schema.py`
- `ssir/pathfinder/data_collection/episode_generator.py` (includes M1.2 & M1.3 helpers)
- `ssir/pathfinder/data_collection/test_basic.py`

**Tests**: ✓ Real graph (41 users), throughput range 1e3–1.6e4 bps, save/load verified

---

### M1.2 - Candidate Route Generation
**Status**: Pending (partially implemented in M1.1)

Generate diverse candidate paths for each user.

**Requirements**:
- Use A* with multiple metrics ("hop", "distance", "random" variants)
- Remove duplicates
- Return unique paths

**Implementation**: `_generate_candidate_paths()` in `episode_generator.py`

---

### M1.3 - Candidate Evaluation & Ranking
**Status**: Pending (partially implemented in M1.1)

Evaluate ground-truth throughput for all candidates.

**Requirements**:
- Evaluate each candidate on partial graph (fresh copy)
- Handle NaN/inf gracefully
- Rank by throughput
- Epsilon-greedy selection:
  - With prob `epsilon`: pick highest-throughput route
  - Else: random from top 5%

**Implementation**: 
- `_evaluate_candidate()` — computes throughput
- `_select_route_epsilon_greedy()` — picks route
- Both in `episode_generator.py`

---

### M1.4 - Episode Generation (Single-Threaded)
**Status**: Pending

Wrap candidate generation/evaluation into full episode loop.

**Requirements**:
- For each user in episode:
  - Generate candidates
  - Evaluate all candidates
  - Select via epsilon-greedy
  - Record DataEntry
  - Apply selected route to partial graph
- Return EpisodeDataset with statistics

**Implementation**: `generate_episode()` in `episode_generator.py` (done, needs testing at scale)

**Metrics to Track**:
- Throughput range (min/max/mean)
- Candidates per user (diversity)
- Selection statistics (exploitation vs. exploration)

---

### M1.5 - Multi-processing Episode Generation
**Status**: Pending

Parallelize episode generation across multiple workers.

**Requirements**:
- Worker pool: each worker generates 1-10 episodes independently
- Main process: coordinates workers, writes sharded pickle files
- Track: total episodes generated, throughput statistics, candidate diversity
- Support early stopping / resume from checkpoint

**File to Create**: `ssir/pathfinder/data_collection/collector.py`

**Responsibilities**:
```python
def collect_episodes(
    num_episodes: int,
    num_workers: int,
    output_dir: str,
    config: dict,  # SPSC, density ranges, candidate count, epsilon
) -> None:
    """Collect episodes in parallel, write to sharded files."""
```

**Output Structure**:
```
data/train/
├── episode_0000.pkl
├── episode_0001.pkl
├── ...
├── metadata.json  # summary stats
└── config.yaml    # collection config
```

---

## Phase 2: Feature Engineering & Dataset

### M2.1 - Graph Encoding Functions
**Status**: Pending

Extract graph features for NN input.

**Requirements**:
- Node features: type, position, hops, power capacity, antenna gain
- Edge features: SNR (if computable), distance, link capacity
- Graph-level features: SPSC, noise density, eavesdropper density
- Output: PyTorch geometric `Data` object

**File to Create**: `ssir/pathfinder/data_collection/graph_features.py`

---

### M2.2 - Candidate Encoding
**Status**: Pending

Convert candidate routes into learnable representations.

**Requirements**:
- Create node/edge masks for candidate path
- Extract subgraph features along route
- Compute projected load (bandwidth sharing impact)
- Output: `CandidateFeatures(node_mask, edge_mask, subgraph_x, load_projection)`

**File to Create**: `ssir/pathfinder/data_collection/candidate_features.py`

---

### M2.3 - Dataset Loader
**Status**: Pending

PyTorch dataset wrapper for training.

**Requirements**:
- Load episodes from pickle shards
- Yield `(graph_features, candidate_features_list, true_throughputs)` tuples
- Support train/val splits
- Optional in-memory feature caching
- Batching: collate multiple candidates per graph

**File to Create**: `ssir/pathfinder/data_collection/dataset.py`

---

### M2.4 - Throughput Normalization
**Status**: Pending

Handle wide throughput range (Mbps → Kbps).

**Requirements**:
- Analyze throughput distribution across dataset
- Choose normalization strategy:
  - **Option A**: Predict log-throughput (targets = log(throughput))
  - **Option B**: Batch-wise normalization + denormalize output
  - **Option C**: Quantile-based loss (e.g., Huber) with wide bounds
- Apply to all training targets

**Decision**: TBD (recommend Option A: log-throughput)

---

## Phase 3: Train Throughput Predictor

### M3.1 - NN Architecture (Throughput Scorer)
**Status**: Pending

Design and implement the throughput prediction model.

**Architecture**:
```
Input: full graph + candidate route mask
├─ Graph Encoder (GNN): produces node/edge embeddings
├─ Candidate Pooler: aggregate embeddings along masked route + load projection
├─ Throughput Head: FC layers (embedding → normalized throughput logit)
Output: predicted throughput (scalar per candidate)
```

**Requirements**:
- Reuse existing GNN from `ssir/pathfinder/rl/network.py` if possible
- Support batched scoring: (B graphs, K candidates) → (B, K) predictions
- Handle variable graph sizes

**File to Create**: `ssir/pathfinder/train_nn/model.py`

---

### M3.2 - Loss Function & Metrics
**Status**: Pending

Define training objectives and evaluation metrics.

**Requirements**:
- Loss: MSE on normalized throughput, or Huber for robustness
- Metrics:
  - MAE (mean absolute error)
  - Ranking accuracy (% top-1, top-5 correct)
  - Relative error: |pred - true| / true
- Track separately: high-throughput (leaf) vs. low-throughput (near-source) regimes

**File to Create**: `ssir/pathfinder/train_nn/loss.py`

---

### M3.3 - Training Loop
**Status**: Pending

Implement supervised training procedure.

**Requirements**:
- Load dataset batches
- Forward: predict throughputs for all candidates
- Backward: compute loss, optimize
- Early stopping on validation ranking accuracy
- Checkpoint best model (by validation loss)
- Hyperparameters: lr, batch size, weight decay, GNN depth, embedding dim

**File to Create**: `ssir/pathfinder/train_nn/train.py`

**Config Keys**:
```yaml
training:
  epochs: 100
  batch_size: 128
  embedding_dim: 128
  loss_fn: mse  # or huber
  lr: 1e-4
  weight_decay: 1e-5
  scheduler: cosine
  early_stopping_patience: 10
  checkpoint_dir: models/
```

---

### M3.4 - Validation & Analysis
**Status**: Pending

Evaluate model performance and identify failure modes.

**Requirements**:
- Plot actual vs. predicted throughput (log scale, separate by regime)
- Confusion matrix for top-5 ranking accuracy
- Analyze failure cases
- Generate calibration curves
- Benchmark: inference time vs. ground truth

**File to Create**: `ssir/pathfinder/train_nn/evaluate.py`

---

## Phase 4: Replace Node-wise Compute with Predictor

### M4.1 - Inference Module
**Status**: Pending

Wrapper for using trained model at inference time.

**Requirements**:
- Load trained model, set to eval mode
- Forward pass (no grad)
- Denormalize output back to original throughput scale
- Return predicted throughput

**Implementation**: `ssir/pathfinder/inference/predictor.py`

---

### M4.2 - Candidate Scoring via NN
**Status**: Pending

Replace expensive `compute_network_throughput()` with NN predictions.

**Requirements**:
- Rewrite candidate evaluation to use NN instead of exact computation
- Profile: measure speedup (expect 10-100×)

**Implementation**: Update `ssir/pathfinder/inference/scorer.py`

---

### M4.3 - Inference Episode Generation
**Status**: Pending

Generate episodes using the trained model instead of ground truth.

**Requirements**:
- For each user:
  - Generate candidates
  - Score with NN (fast)
  - Apply best route
- Compare final episode throughput vs. ground truth (Monte Carlo reference)
- Return metrics: throughput ratio, runtime savings

**Implementation**: `ssir/pathfinder/inference/episode_generator.py`

---

### M4.4 - Speedup Analysis
**Status**: Pending

Benchmark and report inference efficiency gains.

**Requirements**:
- Time breakdown: candidate scoring, model inference, graph updates
- Compare NN-based vs. brute-force (exact) inference
- Expected: 10-100× speedup

**Output**: Benchmark report in `results/inference_benchmark.json`

---

## Phase 5: Integration & Cleanup

### M5.1 - Unified Config & CLI
**Status**: Pending

Create single configuration file for all three phases.

**Config Structure**:
```yaml
data_collection:
  num_episodes: 10000
  candidates_per_user: 50
  epsilon: 0.1
  num_workers: 8
  output_dir: data/train

training:
  epochs: 100
  batch_size: 128
  embedding_dim: 128
  lr: 1e-4
  model_dir: models/
  model_name: best_model.pth

inference:
  model_path: models/best_model.pth
  num_inference_episodes: 100
  batch_size: 32
```

**File to Create**: `configs/throughput_predictor.yaml`

---

### M5.2 - CLI Entrypoints
**Status**: Pending

Create command-line interfaces for three phases.

**Files to Create**:
- `ssir/pathfinder/data_collection/__main__.py`
- `ssir/pathfinder/train_nn/__main__.py`
- `ssir/pathfinder/inference/__main__.py`

**Usage**:
```bash
# Data collection
python -m ssir.pathfinder.data_collection --config configs/throughput_predictor.yaml

# Training
python -m ssir.pathfinder.train_nn --config configs/throughput_predictor.yaml

# Inference
python -m ssir.pathfinder.inference --config configs/throughput_predictor.yaml
```

---

### M5.3 - Archive Old Code
**Status**: Pending

Clean up deprecated RL code.

**Requirements**:
- Move old online RL code to `ssir/pathfinder/rl_archive/`:
  - `agent.py`
  - Old `train.py`
  - `network.py` (if not reused)
- Update imports across all modules
- Clean up `ssir/pathfinder/__init__.py`

---

## Dependency Order

```
M1.1 ✅
  ↓
M1.2 → M1.3 → M1.4 → M1.5
        ↓
M2.1 → M2.2 → M2.3 → M2.4
                 ↓
M3.1 → M3.2 → M3.3 → M3.4
                      ↓
                    M4.1 → M4.2 → M4.3 → M4.4
                                          ↓
                                        M5.1 → M5.2 → M5.3
```

---

## Progress Summary

| Milestone | Status | Notes |
|-----------|--------|-------|
| M1.1 | ✅ COMPLETE | Data schema (DataEntry, EpisodeDataset), I/O |
| M1.2 | ✅ COMPLETE | Candidate generation (multi-metric A*) |
| M1.3 | ✅ COMPLETE | Candidate evaluation + epsilon-greedy selection |
| M1.4 | ✅ COMPLETE | Single-threaded collector (1.9s/episode) |
| M1.5 | ✅ COMPLETE | Parallel collector (1.84x speedup, 2 workers) |
| M2.1 | ✅ COMPLETE | Graph features (17D node, 1D edge, 2D global) |
| M2.2 | ✅ COMPLETE | Candidate masks + load projections |
| M2.3 | ✅ COMPLETE | PyTorch dataset + loader with train/val split |
| M2.4 | ✅ COMPLETE | Log-scale normalization (1e3-1.6e4 bps range) |
| M3.1 | ✅ COMPLETE | GAT encoder + candidate pooling + throughput head |
| M3.2 | ✅ COMPLETE | MSE loss + MAE/RMSE/ranking metrics + regime-specific |
| M3.3 | ✅ COMPLETE | Training loop, optimizer, scheduler, early stopping |
| M3.4 | ✅ COMPLETE | Evaluation, metrics, visualizations, report generation |
| M4.1 | ⏳ PENDING | Inference module |
| M4.2 | ⏳ PENDING | Candidate scoring via NN |
| M4.3 | ⏳ PENDING | Inference episode generation |
| M4.4 | ⏳ PENDING | Speedup analysis |
| M5.1 | ⏳ PENDING | Unified config |
| M5.2 | ⏳ PENDING | CLI entrypoints |
| M5.3 | ⏳ PENDING | Archive old code |

---

## Critical Design Decisions

1. **Throughput Range Handling** (M2.4):
   - Problem: 1e3 bps (Kbps, near-source) to 1.6e4 bps (Mbps, leaf users)
   - Recommendation: Log-scale prediction (targets = log(throughput))
   - Validation: separate metrics for high/low throughput regimes

2. **Feature Importance**:
   - Number of downstream users (bandwidth sharing) — most critical
   - Candidate path length (hops) and bottleneck node position
   - Model must learn interaction between graph state + route choice

3. **Baseline Comparisons**:
   - Start: random route selection (establish floor)
   - Then: greedy (max throughput, expensive ground truth)
   - Finally: NN-based (fast, should match or exceed greedy)

---

## Success Criteria

- ✅ Data collection: generate 10K+ episodes, throughput range validated
- ✅ NN training: >90% ranking accuracy (top-5), <10% relative error
- ✅ Inference: 10-100× speedup vs. ground-truth compute
- ✅ End-to-end: inference episode throughput within 5% of Monte Carlo baseline
