# Phase 1: Data Collection - COMPLETE ✅

**Completed**: M1.1 through M1.5 (all data collection milestones)

## Summary

Built a complete data collection pipeline for generating training episodes for a throughput predictor neural network. The pipeline supports both single-threaded and multi-process collection modes.

### What Was Built

**Core Data Schema (M1.1)**
- `DataEntry`: Individual training example (graph state, candidates, ground-truth throughputs, selected route)
- `EpisodeDataset`: Complete episode wrapper with aggregated statistics
- Pickle-based serialization for efficiency

**Episode Generation (M1.2-M1.3)**
- Multi-metric A* candidate generation (hop, distance, random variants)
- Ground-truth throughput evaluation via network simulation
- Epsilon-greedy route selection (exploit vs explore)
- Integrated into single `generate_episode()` function

**Collection Infrastructure (M1.4-M1.5)**
- **M1.4**: Single-threaded `EpisodeCollector` (baseline, good for testing)
- **M1.5**: Parallel `ParallelEpisodeCollector` (production-ready, 1.84x speedup with 2 workers)
- Both support configuration sampling (SPSC thresholds, eavesdropper densities)
- Statistics aggregation across all episodes
- CLI support with `--num-workers` flag

### Files Created

```
ssir/pathfinder/data_collection/
├── __init__.py              # Module exports
├── __main__.py              # CLI interface
├── data_schema.py           # DataEntry, EpisodeDataset
├── episode_generator.py     # Core generation logic
├── collector.py             # Single-threaded collector
├── parallel_collector.py    # Multi-process collector
├── test_basic.py            # M1.1 tests
├── test_collector.py        # M1.4 tests
├── test_parallel_collector.py  # M1.5 tests
├── M1_1_SUMMARY.md          # M1.1 documentation
├── M1_4_SUMMARY.md          # M1.4 documentation
└── M1_5_SUMMARY.md          # M1.5 documentation
```

### Test Results

**M1.1 Tests** (5 episodes):
- ✅ Data schema validation
- ✅ Episode generation (41 users/episode)
- ✅ Throughput range: 1.03e3 - 1.59e4 bps
- ✅ Save/load integrity

**M1.4 Tests** (11 episodes total):
- ✅ Single config collection (5 episodes, ~10.5s)
- ✅ Multiple config sampling (6 episodes, ~11s)
- ✅ File integrity verification

**M1.5 Tests** (14 episodes total):
- ✅ 2-worker parallel (4 episodes, 4.07s)
- ✅ 4-worker parallel (8 episodes, 3.77s)
- ✅ File integrity verification
- ✅ Speedup analysis: 1.84x with 2 workers, 92% efficiency

### Performance Characteristics

```
Single Episode Generation:
  - 41 users, ~30 candidates each
  - Time: ~1.9-2.0 seconds
  - Output: ~1.6MB pickle file

Scaling:
  - Sequential: O(n) linear with episodes
  - Parallel (2 workers): 1.84x speedup
  - Parallel (4 workers): ~3.0x expected (multiprocessing overhead ~20-30%)

Throughput Range (Validated):
  - Leaf users (few downstream): ~15 Kbps
  - Intermediate nodes (many users): ~1-2 Kbps
  - Range: 1e3 - 1.6e4 bps (wide dynamic range)
```

### Usage

**Command Line (Single-threaded)**:
```bash
python -m ssir.pathfinder.data_collection \
  --graph /fast/hslyu/train/exp_000/graph.pkl \
  --num-episodes 1000 \
  --output-dir data/train \
  --candidates-per-user 50 \
  --epsilon 0.1
```

**Command Line (Parallel, 4 workers)**:
```bash
python -m ssir.pathfinder.data_collection \
  --graph /fast/hslyu/train/exp_000/graph.pkl \
  --num-episodes 10000 \
  --output-dir data/train \
  --candidates-per-user 50 \
  --epsilon 0.1 \
  --num-workers 4 \
  --spsc-thresholds 0.99 0.9999 \
  --eavesdropper-densities 1e-3 1e-4 \
  --verbose
```

**Python API**:
```python
from ssir.pathfinder.data_collection import (
    ParallelEpisodeCollector,
    CollectionConfig,
)
import ssir.basestations as bs

graph = bs.IABRelayGraph()
graph.load_graph("path/to/graph.pkl")

config = CollectionConfig(
    num_episodes=10000,
    candidates_per_user=50,
    epsilon=0.1,
    output_dir="data/train",
    spsc_thresholds=[0.99, 0.9999],
    eavesdropper_densities=[1e-3, 1e-4],
)

collector = ParallelEpisodeCollector(config, num_workers=4)
stats = collector.collect(graph)

print(f"Generated {stats.num_episodes} episodes")
print(f"Throughput range: {stats.throughput_global_min:.2e} - {stats.throughput_global_max:.2e}")
```

### Key Design Decisions

1. **No Graph Caching**: Evaluations are fresh each time (requested simplification)
2. **Pickle Serialization**: Simple, efficient for this use case
3. **Epsilon-Greedy Sampling**:
   - Exploit (ε=0.1): pick highest throughput route
   - Explore (1-ε=0.9): sample randomly from top 5%
4. **Multi-Process Architecture**:
   - Worker pool for embarrassingly parallel workload
   - Independent RNG per worker (seed + worker_id)
   - Aggregated statistics at end (no inter-process sync needed)

### Integration with Existing Code

- Uses existing `IABRelayGraph`, `BaseStation`, `User` classes
- Leverages `utils.get_aborescence_graph()` for route application
- Reuses `compute_network_throughput()` for ground truth
- No modifications to `ssir.basestations` or `ssir.pathfinder.utils` required

### Output Format

**Episode File (episode_XXXXXX.pkl)**:
```python
EpisodeDataset(
    episode_id=0,
    spsc_threshold=0.9999,
    eavesdropper_density=1e-3,
    num_users=41,
    entries=[
        DataEntry(
            episode_id=0,
            user_index=0,
            spsc_threshold=0.9999,
            eavesdropper_density=1e-3,
            master_graph=IABRelayGraph(...),
            partial_graph=IABRelayGraph(...),  # State before this user
            candidate_routes=[UserRouteCandidate(...), ...],
            true_throughputs=[2000.5, 2100.3, ..., 1900.2],  # Ground truth
            selected_candidate_idx=0,  # Epsilon-greedy pick
        ),
        ...
    ],
    throughput_stats={
        'min': 1030.4,
        'max': 15909.1,
        'mean': 2291.9,
        'count': 1230,
        'sum': 2818832.3,
    }
)
```

**Stats File (collection_stats.json)**:
```json
{
  "num_episodes": 1000,
  "total_users": 41000,
  "total_candidates": 1230000,
  "throughput_global_min": 1030.44,
  "throughput_global_max": 15909.08,
  "throughput_global_mean": 2291.88,
  "candidates_per_user_mean": 30.0,
  "candidates_per_user_min": 6,
  "candidates_per_user_max": 50,
  "episodes_by_config": {
    "(0.9900, 1.00e-03)": 250,
    "(0.9900, 1.00e-04)": 250,
    "(0.9999, 1.00e-03)": 250,
    "(0.9999, 1.00e-04)": 250
  },
  "runtime_seconds": 1234.56
}
```

## Next Phase: Feature Engineering & Training (M2.1-M2.4)

Phase 2 requires:
1. **M2.1**: Graph encoding (node/edge features)
2. **M2.2**: Candidate encoding (route masks + load projections)
3. **M2.3**: PyTorch dataset loader with batching
4. **M2.4**: Throughput normalization (recommend log-scale for 1e3-1.6e4 range)

These will transform raw episodes into learnable representations for the NN.

## Recommendations for Large-Scale Collection

**10K+ Episodes**:
```bash
# Use 8-16 workers depending on system cores
# Run on machine with 32GB+ RAM (each worker loads ~100MB graph)

python -m ssir.pathfinder.data_collection \
  --graph graph.pkl \
  --num-episodes 100000 \
  --output-dir data/train_large \
  --candidates-per-user 50 \
  --epsilon 0.1 \
  --num-workers 16 \
  --spsc-thresholds 0.9 0.95 0.99 0.999 0.9999 \
  --eavesdropper-densities 1e-4 1e-3 1e-2 \
  --verbose
```

Expected runtime: ~16-20 hours (100K episodes / 16 workers / 1.9s per episode ≈ 1.2 worker-hours).

## Documentation

- `ssir/pathfinder/data_collection/M1_1_SUMMARY.md`: Data schema details
- `ssir/pathfinder/data_collection/M1_4_SUMMARY.md`: Single-threaded collector
- `ssir/pathfinder/data_collection/M1_5_SUMMARY.md`: Parallel collector
- `MILESTONES.md`: Full project roadmap with all 19 milestones
