# M1.4: Single-Threaded Episode Collection - COMPLETE

## Overview
Implemented a single-threaded episode collector that generates multiple episodes with progress tracking and aggregated statistics.

## Key Files Created

### 1. `collector.py`
Core collection infrastructure:

- **`CollectionConfig`**: Configuration dataclass with:
  - `num_episodes`: Total episodes to generate
  - `candidates_per_user`: Routes per user (default 50)
  - `epsilon`: Exploitation probability (default 0.1)
  - `output_dir`: Save location
  - `spsc_thresholds`: List of SPSC values to sample
  - `eavesdropper_densities`: List of densities to sample
  - `seed`: Random seed for reproducibility
  - `verbose`: Progress reporting

- **`CollectionStats`**: Aggregated statistics with:
  - Episode count, user count, candidate count
  - Global throughput range (min/max/mean)
  - Candidates per user statistics
  - Configuration sampling distribution
  - Runtime in seconds
  - JSON serialization for persistence

- **`EpisodeCollector`**: Main collector class
  - `__init__(config)`: Initialize with configuration
  - `collect(graph)` → `CollectionStats`: Generate all episodes
  - Tracks progress via logging
  - Saves episodes to disk with metadata
  - Aggregates statistics across all episodes

### 2. `__main__.py`
Command-line interface for episode collection:

```bash
python -m ssir.pathfinder.data_collection \
    --graph /path/to/graph.pkl \
    --num-episodes 1000 \
    --output-dir data/train \
    --candidates-per-user 50 \
    --epsilon 0.1 \
    --spsc-thresholds 0.99 0.9999 \
    --eavesdropper-densities 1e-3 1e-4 \
    --seed 42 \
    --verbose
```

**Output Files**:
- `episode_XXXXXX.pkl`: Serialized EpisodeDataset (one per episode)
- `collection_stats.json`: Aggregated statistics in JSON format
- `collection.log`: Collection progress and errors

### 3. `test_collector.py`
Comprehensive tests:
- Single configuration collection (5 episodes)
- Multiple configuration sampling (6 episodes, mixed SPSC/density)
- Episode file integrity verification

## Design Decisions

1. **Configuration Sampling**: 
   - Episodes randomly sample from SPSC and eavesdropper density lists
   - Allows evaluating model across environmental conditions
   - Tracked in `episodes_by_config` dict

2. **Progress Tracking**:
   - File logging to `collection.log`
   - Periodic console output (every 100 episodes if verbose)
   - Statistics updated incrementally
   - Handles errors gracefully with logging

3. **Reproducibility**:
   - Seed control for deterministic episode ordering
   - Stored in config for reference
   - Allows resuming collections with same sequence

4. **Statistics Aggregation**:
   - Tracks all throughput values across all candidates
   - Computes global min/max/mean
   - Tracks candidate diversity (min/max/mean per user)
   - Monitors configuration sampling distribution

## Test Results

```
Single Config (5 episodes):
  - Total users: 205
  - Total candidates: 4844
  - Throughput range: 1.03e+03 - 1.59e+04 bps
  - Mean throughput: 2.30e+03 bps
  - Runtime: 10.50s

Multiple Configs (6 episodes):
  - Sampled 4 different (SPSC, density) pairs
  - Total users: 246
  - Runtime: 11.04s

CLI Test (3 episodes):
  - Episodes collected: 3
  - Total users: 123
  - Total candidates: 2908
  - Throughput range: 1.03e+03 - 1.59e+04 bps
  - Runtime: 6.38s
```

## Performance Notes

- **Single episode**: ~2s per 41-user graph
- **Scaling**: Linear in number of episodes (embarrassingly parallel)
- **Memory**: Each episode ~1.6MB (full graph state stored)
- **Throughput range**: Consistent 1e3-1.6e4 bps (Kbps-Mbps)

## Integration with Previous Milestones

- **M1.2**: Candidate generation (via `_generate_candidate_paths()`)
- **M1.3**: Candidate evaluation (via `_evaluate_candidate()` + epsilon-greedy)
- **M1.1**: Data schema (DataEntry, EpisodeDataset)

All utilities from M1.1-M1.3 are used directly in the single-threaded loop.

## Next Steps (M1.5)

- Multi-process wrapper: parallel episode generation
- Worker pool management
- Sharded file writing
- Checkpoint/resume support
- Distributed progress tracking

## Usage Examples

### Basic Collection
```python
from ssir.pathfinder.data_collection import EpisodeCollector, CollectionConfig
import ssir.basestations as bs

graph = bs.IABRelayGraph()
graph.load_graph("path/to/graph.pkl")

config = CollectionConfig(
    num_episodes=1000,
    output_dir="data/train",
)

collector = EpisodeCollector(config)
stats = collector.collect(graph)

print(f"Collected {stats.num_episodes} episodes")
print(f"Throughput range: {stats.throughput_global_min:.2e} - {stats.throughput_global_max:.2e}")
```

### Multi-Configuration Collection
```python
config = CollectionConfig(
    num_episodes=10000,
    candidates_per_user=50,
    epsilon=0.1,
    spsc_thresholds=[0.99, 0.999, 0.9999],
    eavesdropper_densities=[1e-4, 1e-3, 1e-2],
    seed=42,
    verbose=True,
)
```

### CLI Collection
```bash
python -m ssir.pathfinder.data_collection \
    --graph /fast/hslyu/train/exp_000/graph.pkl \
    --num-episodes 10000 \
    --output-dir data/train_large \
    --candidates-per-user 50 \
    --spsc-thresholds 0.99 0.9999 \
    --eavesdropper-densities 1e-3 1e-4 \
    --verbose
```
