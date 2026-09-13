The full graph path of `_directed_triangles_and_degree_iter` rebuilds the predecessor and successor sets for every neighbor of every node. For node `i` it builds `ipreds = set(preds) - {i}` and `isuccs = set(succs) - {i}`. Then, in the `j` loop, it builds `set(G._pred[j]) - {j}` and `set(G._succ[j]) - {j}` for each `j` in `chain(ipreds, isuccs)`. A node `j` with several incident edges has those sets reconstructed once for each such edge, and the intersection count is paid again through a Python generator, `sum(1 for k in chain(...))`.

This change adds a fast path for the `nodes is None` case. It builds `preds`, `succs`, `preds | succs`, and `preds & succs` once for each node and stores them in `node_data`. For each incident neighbor it then counts the four intersections with four `len` calls. The identity `|P ∩ Pj| + |P ∩ Sj| + |S ∩ Pj| + |S ∩ Sj| = |(P ∪ S) ∩ (Pj ∪ Sj)| + |(P ∩ S) ∩ (Pj ∪ Sj)| + |(P ∪ S) ∩ (Pj ∩ Sj)| + |(P ∩ S) ∩ (Pj ∩ Sj)|` lets it do that without separating `jpreds` and `jsuccs`. Reciprocal neighbors are doubled once, matching the duplicate `j` in `chain(ipreds, isuccs)`.

The existing streaming loop remains for explicit `nodes` and for `nodes` not `None`, so subset and singleton calls keep their previous behavior. The full graph path now holds each node's four neighbor sets for the duration of the iteration; this replaces the repeated per incident edge set construction with one set construction per node. Self loops are still discarded from each node's sets before the union and intersection are stored.

Measured against commit `c94928ed94899033126c9d47f797a1f698584b20`. Each input was run 6 times on the base and 6 times with this change; the numbers are derived from the medians.

| input | base | proposed improvement | speedup |
|---|---|---|---|
| `G = nx.erdos_renyi_graph(1000, 0.05, seed=42, directed=True)` | 1.372 s | 350.2 ms | 3.91x |
| `G = nx.erdos_renyi_graph(1000, 0.01, seed=42, directed=True)` | 77.9 ms | 21.3 ms | 3.63x |
| `G = nx.erdos_renyi_graph(1000, 0.002, seed=42, directed=True)` | 8.8 ms | 3.3 ms | 2.67x |
| `G = nx.gn_graph(800, seed=3)` | 3.7 ms | 1.5 ms | 2.48x |
| `G = nx.erdos_renyi_graph(1600, 0.02, seed=42, directed=True)` | 883.8 ms | 231.2 ms | 3.83x |
