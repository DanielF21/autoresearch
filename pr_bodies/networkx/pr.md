# ENH: Speed up directed clustering in nx.clustering

The motivation for this change is to speed up `nx.clustering` on directed graphs without weights. The hot path is `_directed_triangles_and_degree_iter`. For each node it builds `ipreds` and `isuccs` as sets, then for every neighbor `j` in `chain(ipreds, isuccs)` it rebuilds `set(G._pred[j]) - {j}` and `set(G._succ[j]) - {j}` and runs four Python set intersections. A node of degree `d` therefore has its adjacency reconstructed once for every incident edge, so the work includes repeated set construction on the order of `m * d`, not just the intersections on the original two sets.

This change keeps that original loop as the fallback and adds an integer mask path for the whole graph case. When `nodes` is None and `len(G)` is at most 4096, each node is mapped to an index once and its in and out neighbors are stored as Python integer masks with self loops removed. The four set intersections per neighbor become four bitwise ANDs followed by `int.bit_count`, and `dtotal` and `dbidirectional` are computed with `bit_count` on the predecessor mask, its successor mask, and their intersection. Single node and node subset calls, and graphs with more than 4096 nodes, still use the existing set based code. The 4096 node bound keeps each mask at a bounded width and avoids big integer shifts and popcounts on large sparse graphs, where the set based loop is cheaper.

Measured against commit `c94928ed94899033126c9d47f797a1f698584b20`. Each input was run 6 times on the base and 6 times with this change; the numbers are derived from the medians.

| input | base | proposed improvement | speedup |
|---|---|---|---|
| `G = nx.erdos_renyi_graph(1000, 0.05, seed=42, directed=True)` | 1.467 s | 75.1 ms | 19.51x |
| `G = nx.erdos_renyi_graph(1000, 0.01, seed=42, directed=True)` | 83.0 ms | 14.2 ms | 5.83x |
| `G = nx.erdos_renyi_graph(1000, 0.002, seed=42, directed=True)` | 9.4 ms | 3.3 ms | 2.80x |
| `G = nx.gn_graph(800, seed=3)` | 3.8 ms | 1.3 ms | 2.82x |
| `G = nx.erdos_renyi_graph(1600, 0.02, seed=42, directed=True)` | 952.9 ms | 90.8 ms | 10.47x |
