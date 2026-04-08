# M1.1: Data Collection Module Structure - COMPLETE

## Overview
Implemented the foundational data schema and infrastructure for collecting training data for the throughput predictor.

## Key Files Created

### 1. `data_schema.py`
Defines core data structures:
- **`DataEntry`**: Single training example containing:
  - Episode metadata (episode_id, user_index, SPSC threshold, eavesdropper density)
  - Graph state (master_graph, partial_graph)
  - Candidate routes and their ground-truth throughputs
  - Selected route index (via epsilon-greedy)
  - Optional metadata for analysis
  - Validation ensures consistency between candidates and throughputs

- **`EpisodeDataset`**: Complete episode wrapper with:
  - Episode-level metadata
  - List of DataEntry objects (one per user)
  - Throughput statistics (min, max, mean)
  - Validation ensures all entries belong to the same episode

- **I/O Functions**:
  - `save_episode()` / `load_episode()`: Serialize/deserialize EpisodeDataset
  - `save_data_entry()` / `load_data_entry()`: Individual entry I/O

### 2. `episode_generator.py`
Core episode generation logic (implements M1.2 & M1.3):
- **`_generate_candidate_paths()`**: Creates diverse candidate paths using:
  - Multiple A* metrics (hop, distance, random variants)
  - Returns unique paths
  
- **`_paths_to_candidates()`**: Converts paths to `UserRouteCandidate` objects

- **`_evaluate_candidate()`**: Computes ground-truth throughput:
  - Applies candidate route to partial graph
  - Calls `compute_network_throughput()`
  - Handles NaN/inf gracefully (returns 0.0)

- **`_select_route_epsilon_greedy()`**: Route selection:
  - With probability `epsilon`: select highest-throughput route
  - Otherwise: randomly from top 5%

- **`generate_episode()`**: Main entry point:
  - Iteratively assigns users to routes
  - Evaluates all candidates for each user
  - Selects via epsilon-greedy
  - Records ground-truth throughputs
  - Applies selected route to partial graph before next user
  - Returns `EpisodeDataset` with statistics

### 3. `test_basic.py`
Test suite validating:
- DataEntry validation (catches mismatches between candidates and throughputs)
- Episode generation with real graph (41 users, Kbps-Mbps range)
- Save/load integrity

## Design Decisions

1. **No Caching**: Skipped graph state caching as requested—evaluations are fresh each time
2. **Pickle Format**: Episodes serialized as `.pkl` files for simplicity
3. **Throughput Range**: Real data spans 1e3-1.6e4 (Kbps-Mbps); model will need log-scale handling
4. **Epsilon-Greedy Exploration**: 
   - Exploitation (epsilon=0.1): pick highest throughput
   - Exploration (1-epsilon): sample from top 5% for diversity

## Test Results

```
Generated episode with 41 users
  - Throughput range: 1.03e+03 - 1.59e+04 bps
  - Mean throughput: 2.13e+03 bps
  - Save/load integrity verified
```

## Next Steps (M1.4 & M1.5)

- **M1.4**: Wrap `generate_episode()` with episode recording and I/O
- **M1.5**: Multi-processing wrapper for embarrassingly parallel episode generation

## Assumptions & Constraints

- User assignment order: distance-based (farthest first) by default, but customizable
- Graph state: immutable master_graph, mutable partial_graph for each step
- Throughput evaluation: always on fresh copy of partial_graph
- SPSC/eavesdropper settings: fixed per episode (passed as parameters)
