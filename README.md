# two_layer_hnsw_like_cpp

A C++/pybind11 implementation of the query-heavy parts of TLHL with additional structural optimizations.

## What moved to C++

- center-layer graph build
- base-layer graph build
- CSR conversion
- query routing on the center hyperplane tree
- center-level probing
- base-layer graph search
- top-k reranking

## Structural optimizations included

- CSR adjacency for query traversal
- visited-tag arrays in C++
- real-point anchor entries per center
- adaptive internal `ef` budget
- batch query API

## What remains in Python

- balanced clustering (`CPD_KMeans`) / standard `KMeans`
- flattening the Python router tree into arrays for the C++ core
- anchor extraction from cluster labels

## Install

```bash
pip install .
```

## Example

```python
from two_layer_hnsw_like_cpp import TwoLayerHNSWLikeIndexCPP

index = TwoLayerHNSWLikeIndexCPP(n_centers=512, m=12, ef_construction=100)
index.fit(X)
ans = index.search(q, k=100, ef=20, n_probe_centers=3)
```
