`_directed_triangles_and_degree_iter` is the loop behind directed `nx.clustering`. For every node `i`, the old loop builds `set(G._pred[j]) - {j}` and `set(G._succ[j]) - {j}` for every `j` in `chain(ipreds, isuccs)`. Those neighbor sets are thrown away at the end of each `j` iteration, so they are rebuilt for each adjacent `(i, j)` pair rather than once per node. The four intersections are then counted with:

```python
directed_triangles += sum(
    1
    for k in chain(
        (ipreds & jpreds),
        (ipreds & jsuccs),
        (isuccs & jpreds),
        (isuccs & jsuccs),
    )
)
```

which walks each element of each intersection through a Python generator.

This change caches each node's predecessor and successor sets once when `nodes is None`, with self-loops discarded, and stores their union and intersection. The inner loop visits each unique neighbor once and counts the same four intersections as `len(set & set)` terms. If `ineighbors` is the in-or-out set and `reciprocal` is the in-and-out set for a node, then:

```
|ipreds & jpreds| + |ipreds & jsuccs| + |isuccs & jpreds| + |isuccs & jsuccs|
    = |ineighbors & jneighbors|
    + |reciprocal & jneighbors|
    + |ineighbors & jreciprocal|
    + |reciprocal & jreciprocal|
```

`chain(ipreds, isuccs)` yields a reciprocal neighbor twice, so the cached path doubles the contribution when `j` is in `reciprocal`. The `nodes is not None` path is unchanged. The cached path stores three sets per node for the duration of the iterator; that is the memory-for-time trade in the all-nodes case.

Measured against base commit `c94928ed94899033126c9d47f797a1f698584b20`, with results compared against the base and identical on every input:

| input | base median | patched median | speedup |
|---|---|---|---|
| `nx.erdos_renyi_graph(1000, 0.05, seed=42, directed=True)` | 1.372 s | 350.2 ms | 3.91x |
| `nx.erdos_renyi_graph(1000, 0.01, seed=42, directed=True)` | 77.9 ms | 21.3 ms | 3.63x |
| `nx.erdos_renyi_graph(1000, 0.002, seed=42, directed=True)` | 8.8 ms | 3.3 ms | 2.67x |
| `nx.gn_graph(800, seed=3)` | 3.7 ms | 1.5 ms | 2.48x |
| `nx.erdos_renyi_graph(1600, 0.02, seed=42, directed=True)` | 883.8 ms | 231.2 ms | 3.83x |

Geometric mean over the five inputs is 3.25x. `networkx/algorithms/tests/test_cluster.py` passes (56 passed, 0 failed); the full suite passes (9090 passed, 0 failed).
