The motivation for this change is to speed up `nx.clustering` on directed graphs.

In `_directed_triangles_and_degree_iter`, the `nodes is None` path rebuilt `set(G._pred[j])` and `set(G._succ[j])` for every neighbor `j` in the inner loop, once for each node that has `j` as a neighbor, and counted the four intersections with a generator over `chain(ipreds, isuccs)`. This PR computes each node's predecessor/successor union and reciprocal sets once, stores them in a dict, and counts the four directed-triangle intersections with four `len(set & set)` operations. A neighbor is still counted twice when it is both a predecessor and successor of the current node, matching the duplicate from `chain(ipreds, isuccs)`.

Measured against base commit `c94928ed94899033126c9d47f797a1f698584b20`, with alternating base and patched runs on one machine. Each input produced the same result as the base.

- `nx.erdos_renyi_graph(1000, 0.05, seed=42, directed=True)`: 1.372 s -> 350.2 ms
- `nx.erdos_renyi_graph(1000, 0.01, seed=42, directed=True)`: 77.9 ms -> 21.3 ms
- `nx.erdos_renyi_graph(1000, 0.002, seed=42, directed=True)`: 8.8 ms -> 3.3 ms
- `nx.gn_graph(800, seed=3)`: 3.7 ms -> 1.5 ms
- `nx.erdos_renyi_graph(1600, 0.02, seed=42, directed=True)`: 883.8 ms -> 231.2 ms

The geometric mean speedup over the five inputs is 3.25x and no input was slower. The path where `nodes` is not `None` is unchanged. This stores one union set and one reciprocal set per node for the `nodes is None` path instead of rebuilding neighbor sets inside the inner loop.

`networkx/algorithms/tests/test_cluster.py`: 56 passed. Full test suite: 9090 passed.
