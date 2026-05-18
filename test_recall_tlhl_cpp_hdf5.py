from __future__ import annotations

import os
import time

import h5py
import numpy as np
import pytest

from two_layer_hnsw_like_cpp import TwoLayerHNSWLikeIndexCPP


def load_annb_hdf5_top100(
    *,
    hdf5_path: str,
    base_limit: int | None,
    query_limit: int | None,
    k_true: int,
    hdf5_distances_are_squared: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Directly load ANN-Benchmarks style HDF5 data.

    Expected datasets:
      - train:     base vectors
      - test:      query vectors
      - neighbors: ground-truth neighbor ids in train
      - distances: ground-truth distances

    Notes:
      - For sift-128-euclidean.hdf5, use train as base and test as query.
      - If your search implementation returns squared L2, keep
        hdf5_distances_are_squared=False so HDF5 Euclidean distances are squared
        before ANNB recall comparison.
    """
    with h5py.File(hdf5_path, "r") as f:
        train = f["train"]
        test = f["test"]
        neighbors = f["neighbors"]
        distances = f["distances"]

        n_base = train.shape[0] if base_limit is None else min(int(base_limit), train.shape[0])
        n_query = test.shape[0] if query_limit is None else min(int(query_limit), test.shape[0])

        if n_base <= 0:
            raise ValueError(f"base_limit must be > 0, got {base_limit}")
        if n_query <= 0:
            raise ValueError(f"query_limit must be > 0, got {query_limit}")
        if k_true > neighbors.shape[1]:
            raise ValueError(f"k_true={k_true} exceeds HDF5 neighbors width={neighbors.shape[1]}")

        base = np.ascontiguousarray(np.asarray(train[:n_base], dtype=np.float32))
        queries = np.ascontiguousarray(np.asarray(test[:n_query], dtype=np.float32))
        true_indices = np.ascontiguousarray(np.asarray(neighbors[:n_query, :k_true], dtype=np.int32))
        true_distances = np.ascontiguousarray(np.asarray(distances[:n_query, :k_true], dtype=np.float32))

    max_gt = int(true_indices.max()) if true_indices.size else -1
    if max_gt >= n_base:
        raise ValueError(
            f"HDF5 ground truth references id {max_gt}, but base_limit is only {n_base}. "
            "Use base_limit equal to the full train size, or regenerate subset-specific GT."
        )

    if not hdf5_distances_are_squared:
        true_distances = true_distances * true_distances

    return base, queries, true_indices, true_distances


def annb_recall_from_distances(
    *,
    true_distances: np.ndarray,
    run_distances: np.ndarray,
    k_eval: int,
    epsilon: float,
) -> float:
    thresholds = true_distances[:, k_eval - 1] + float(epsilon)
    actual = (run_distances[:, :k_eval] <= thresholds[:, None]).sum(axis=1)
    recalls = actual.astype(np.float32) / float(k_eval)
    return float(np.mean(recalls))


def overlap_recall(
    *,
    true_indices: np.ndarray,
    run_indices: np.ndarray,
    k_eval: int,
) -> float:
    recalls = np.empty(true_indices.shape[0], dtype=np.float32)
    for i in range(true_indices.shape[0]):
        gt = set(map(int, true_indices[i, :k_eval]))
        pred = set(map(int, run_indices[i, :k_eval]))
        recalls[i] = len(gt & pred) / float(k_eval)
    return float(np.mean(recalls))


def run_two_layer_index_cpp_queries(
    *,
    base: np.ndarray,
    queries: np.ndarray,
    k_eval: int,
    ef_search: int,
    n_centers: int,
    m: int,
    ef_construction: int,
    n_probe_centers: int,
    query_is_base_prefix: bool,
    cluster_method: str = "cpd_kmeans",
    anchor_k: int = 4,
    ef_extra: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    t1 = time.perf_counter()
    index = TwoLayerHNSWLikeIndexCPP(
        n_centers=n_centers,
        m=m,
        ef_construction=ef_construction,
        cluster_method=cluster_method,
        anchor_k=anchor_k,
        random_state=42,
    ).fit(base)
    t2 = time.perf_counter()

    _ = index.search(
        queries[0],
        k=min(10, k_eval),
        ef=ef_search,
    )

    internal_k = k_eval + 1 if query_is_base_prefix else k_eval

    t3 = time.perf_counter()
    if hasattr(index, "search_many"):
        run_indices, run_distances = index.search_many(
            queries,
            k=internal_k,
            ef=ef_search,
        )
        print("many")
        print(index.core.last_batch_stats())
        run_indices = np.asarray(run_indices, dtype=np.int32)
        run_distances = np.asarray(run_distances, dtype=np.float32)
    else:
        run_indices = np.empty((queries.shape[0], internal_k), dtype=np.int32)
        run_distances = np.empty((queries.shape[0], internal_k), dtype=np.float32)
        for i, q in enumerate(queries):
            ans = index.search(
                q,
                k=internal_k,
                ef=ef_search,
            )
            print("single")
            print(index.core.last_query_stats())
            ids = np.array([idx for idx, _ in ans], dtype=np.int32)
            dists = np.array([dist for _, dist in ans], dtype=np.float32)
            run_indices[i, :len(ids)] = ids
            run_distances[i, :len(dists)] = dists
    t4 = time.perf_counter()

    out_indices = np.empty((queries.shape[0], k_eval), dtype=np.int32)
    out_distances = np.empty((queries.shape[0], k_eval), dtype=np.float32)

    for i in range(queries.shape[0]):
        ids = np.asarray(run_indices[i], dtype=np.int32)
        dists = np.asarray(run_distances[i], dtype=np.float32)

        valid = ids >= 0
        ids = ids[valid]
        dists = dists[valid]

        if query_is_base_prefix:
            mask = ids != i
            ids = ids[mask]
            dists = dists[mask]

        ids = ids[:k_eval]
        dists = dists[:k_eval]

        if ids.shape[0] < k_eval:
            raise ValueError(
                f"Query {i} returned only {ids.shape[0]} results after filtering, expected {k_eval}"
            )

        out_indices[i] = ids
        out_distances[i] = dists

    print(f"index summary: {index.summary()}")
    print(f"create={(t2 - t1) * 1000:.3f} ms query={(t4 - t3) * 1000:.3f} ms")
    return out_indices, out_distances


def test_two_layer_index_cpp_recall() -> None:
    hdf5_path = "../../../Downloads/sift-128-euclidean.hdf5"

    # Full SIFT1M base. This lets us use the original HDF5 neighbors/distances directly.
    base_limit = 1_000_000
    query_limit = None

    # Randomly sample existing test queries from the original HDF5 test set.
    eval_queries = 100
    random_query_seed = 42

    k_true = 100
    k_eval = 20
    epsilon = 1e-3

    # Existing FAISS-generated GT in your old test used squared L2 from IndexFlatL2.
    # ANN-Benchmarks HDF5 distances are usually Euclidean distances, so we square
    # them by default to keep the same ANNB recall distance scale.
    hdf5_distances_are_squared = False

    # C++ TLHL parameters
    n_centers = 2048
    m = 24
    ef_construction = 200
    ef_search = 96
    n_probe_centers = 1

    # Structure parameters
    cluster_method = "cpd_kmeans"
    anchor_k = 32
    ef_extra = 100

    min_annb_recall = 0.95
    min_overlap_recall = 0.95

    if not os.path.exists(hdf5_path):
        pytest.skip(f"Dataset not found: {hdf5_path}")

    if k_true < k_eval:
        raise ValueError(f"k_true ({k_true}) must be >= k_eval ({k_eval})")

    base, queries_all, true_indices_all, true_distances_all = load_annb_hdf5_top100(
        hdf5_path=hdf5_path,
        base_limit=base_limit,
        query_limit=query_limit,
        k_true=k_true,
        hdf5_distances_are_squared=hdf5_distances_are_squared,
    )

    total_queries = queries_all.shape[0]
    n_eval = min(int(eval_queries), total_queries)
    if n_eval <= 0:
        raise ValueError(f"eval_queries must be > 0, got {eval_queries}")

    rng = np.random.default_rng(random_query_seed)
    sampled_idx = rng.choice(total_queries, size=n_eval, replace=False)

    queries_eval = queries_all[sampled_idx]
    true_indices_top = true_indices_all[sampled_idx, :k_eval]
    true_distances_top = true_distances_all[sampled_idx, :k_eval]

    print(
        f"mode=annb_hdf5_test/random_sample_test base={base.shape[0]} total_queries={total_queries} "
        f"eval_queries={n_eval} k_eval={k_eval} random_seed={random_query_seed} "
        f"hdf5_distances_are_squared={hdf5_distances_are_squared}"
    )

    # HDF5 test queries are not base-prefix/self queries, so do not remove self id.
    query_is_base_prefix = False

    run_indices, run_distances = run_two_layer_index_cpp_queries(
        base=base,
        queries=queries_eval,
        k_eval=k_eval,
        ef_search=ef_search,
        n_centers=n_centers,
        m=m,
        ef_construction=ef_construction,
        n_probe_centers=n_probe_centers,
        query_is_base_prefix=query_is_base_prefix,
        cluster_method=cluster_method,
        anchor_k=anchor_k,
        ef_extra=ef_extra,
    )

    annb_recall = annb_recall_from_distances(
        true_distances=true_distances_top,
        run_distances=run_distances,
        k_eval=k_eval,
        epsilon=epsilon,
    )
    overlap = overlap_recall(
        true_indices=true_indices_top,
        run_indices=run_indices,
        k_eval=k_eval,
    )

    print(f"ANNB recall@{k_eval}: {annb_recall:.4f}")
    print(f"Overlap recall@{k_eval}: {overlap:.4f}")

    assert annb_recall >= min_annb_recall
    assert overlap >= min_overlap_recall
