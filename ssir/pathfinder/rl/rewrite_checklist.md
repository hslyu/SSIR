## Trajectory-Candidate Rewrite Checklist

1. Candidate precomputation
- Reuse Monte Carlo-style predecessor/path candidates.
- Precompute per-user feasible route candidates on the master graph.
- Keep the Monte Carlo user-by-user procedure: update partial graph, score route batch, pick best route, continue.

2. Graph embedding
- Build node features containing all variables needed by the throughput equation.
- Build edge features containing all variables needed to recover per-link SNR / spectral efficiency behavior.
- Encode the graph with a message-passing model that produces node and edge embeddings.

3. Candidate-aware pooling
- Represent each user-route candidate with node/edge masks over the master graph.
- Add candidate-specific projected load features needed by the throughput equation.
- Pool node and edge embeddings only along the candidate route.

4. Candidate throughput network
- Predict candidate throughput from the pooled candidate representation.
- Support batched candidate scoring for one graph with many candidates.

5. Training path
- Build targets from exact throughput after applying one user-route candidate to the current partial graph.
- Train the candidate scorer as a supervised ranking / regression model over per-user route batches.
- Replace the current online step-by-step action scorer with the user-by-user candidate scorer after validation.
