Measured against commit `c94928ed94899033126c9d47f797a1f698584b20`. Each input was run 6 times on the base and 6 times with this change; the numbers are derived from the medians.

| input | base | proposed improvement | speedup |
| --- | --- | --- | --- |
| `G = nx.erdos_renyi_graph(1000, 0.05, seed=42, directed=True)` | 1.372 s | 350.2 ms | 3.91x |
| `G = nx.erdos_renyi_graph(1000, 0.01, seed=42, directed=True)` | 77.9 ms | 21.3 ms | 3.63x |
| `G = nx.erdos_renyi_graph(1000, 0.002, seed=42, directed=True)` | 8.8 ms | 3.3 ms | 2.67x |
| `G = nx.gn_graph(800, seed=3)` | 3.7 ms | 1.5 ms | 2.48x |
| `G = nx.erdos_renyi_graph(1600, 0.02, seed=42, directed=True)` | 883.8 ms | 231.2 ms | 3.83x |

`clustering(G)` on a directed graph calls `_directed_triangles_and_degree_iter(G, None)`. On that path, the old code built `ipreds` and `isuccs` for node `i`, then for every `j` in `chain(ipreds, isuccs)` built `jpreds = set(G._pred[j]) - {j}` and `jsuccs = set(G._succ[j]) - {j}` from scratch. Each node is an in- or out-neighbor of every node that touches it, so those `_pred[j]` and `_succ[j]` views were converted into sets and discarded repeatedly. The four overlaps were then counted one element at a time by `sum(1 for k in chain((ipreds & jpreds), (ipreds & jsuccs), (isuccs & jpreds), (isuccs & jsuccs)))`.

This change keeps the existing `nodes is not None` path and adds a full-graph path that builds each node's `preds` and `succs` once. It stores the neighbor union `preds | succs` and the reciprocal-neighbor set `preds & succs`, then iterates the unique neighbor union instead of `chain(ipreds, isuccs)`. The four overlap counts are computed as four `len(set & set)` intersections using the inclusion-exclusion identity

`|ipreds & jpreds| + |ipreds & jsuccs| + |isuccs & jpreds| + |isuccs & jsuccs| = |ineighbors & jneighbors| + |reciprocal & jneighbors| + |ineighbors & jreciprocal| + |reciprocal & jreciprocal|`

where `ineighbors`/`reciprocal` belong to `i` and `jneighbors`/`jreciprocal` belong to `j`. A node in both `preds` and `succs` gets its count doubled once, which matches the old `chain` visiting that reciprocal neighbor twice. The full-graph path therefore keeps two cached sets per node until the iterator is exhausted, while explicit node subsets keep the old streaming loop.
