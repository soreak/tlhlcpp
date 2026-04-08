from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans

from .CPD_Kmeans_old import CPD_KMeans, CenterHyperplaneRouter
from ._tlhl_cpp import TLHLCore


@dataclass
class IndexParams:
    n_centers: int
    m: int = 16
    ef_construction: int = 200
    center_max_degree: Optional[int] = None
    base_max_degree: Optional[int] = None
    random_state: int = 42

    # clustering
    cluster_method: str = "cpd_kmeans"
    cluster_max_iter: int = 20
    cluster_tol: float = 1e-6
    cluster_train_sample_size: Optional[int] = None
    cpd_psd_sample: int = 64
    cpd_psd_exact_k: int = 512
    cpd_mean_sample: int = 64
    cpd_mean_exact_k: int = 512

    # query-structure optimizations
    anchor_k: int = 4
    ef_extra: int = 64
    center_probe_ef_cap: int = 16

    def __post_init__(self) -> None:
        if self.n_centers <= 0:
            raise ValueError("n_centers 必须 > 0")
        if self.m <= 0:
            raise ValueError("m 必须 > 0")
        if self.ef_construction <= 0:
            raise ValueError("ef_construction 必须 > 0")
        if self.cluster_max_iter <= 0:
            raise ValueError("cluster_max_iter 必须 > 0")
        if self.cluster_tol <= 0:
            raise ValueError("cluster_tol 必须 > 0")
        if self.center_max_degree is None:
            self.center_max_degree = 2 * self.m
        if self.base_max_degree is None:
            self.base_max_degree = 2 * self.m
        if self.anchor_k <= 0:
            raise ValueError("anchor_k 必须 > 0")
        if self.ef_extra < 0:
            raise ValueError("ef_extra 必须 >= 0")
        if self.center_probe_ef_cap <= 0:
            raise ValueError("center_probe_ef_cap 必须 > 0")
        self.cluster_method = str(self.cluster_method).lower()
        if self.cluster_method not in {"cpd_kmeans", "kmeans"}:
            raise ValueError("cluster_method 必须是 'cpd_kmeans' 或 'kmeans'")


def _to_float32_matrix(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError("输入数据 X 必须是二维数组 [N, dim]")
    return np.ascontiguousarray(X)


def _flatten_router(router: Optional[CenterHyperplaneRouter], dim: int):
    if router is None:
        return (
            np.empty((0, dim), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
        )

    normals: List[np.ndarray] = []
    bias: List[float] = []
    left: List[int] = []
    right: List[int] = []
    leaf_center: List[int] = []

    def build(node) -> int:
        idx = len(bias)
        normals.append(np.zeros((dim,), dtype=np.float32))
        bias.append(0.0)
        left.append(-1)
        right.append(-1)
        leaf_center.append(-1)

        if node.is_leaf:
            leaf_center[idx] = int(node.center_ids[0])
            return idx

        normals[idx] = np.asarray(node.normal, dtype=np.float32)
        bias[idx] = float(node.bias)
        left[idx] = build(node.left)
        right[idx] = build(node.right)
        return idx

    build(router.root)
    return (
        np.ascontiguousarray(np.vstack(normals).astype(np.float32)),
        np.asarray(bias, dtype=np.float32),
        np.asarray(left, dtype=np.int32),
        np.asarray(right, dtype=np.int32),
        np.asarray(leaf_center, dtype=np.int32),
    )


def _build_center_anchors(
    centers: np.ndarray,
    labels: np.ndarray,
    data_vectors: np.ndarray,
    virtual_base_count: int,
    anchor_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    K = centers.shape[0]
    offsets = np.zeros((K + 1,), dtype=np.int64)
    all_ids: List[np.ndarray] = []

    for cid in range(K):
        members = np.where(labels == cid)[0]
        if members.size == 0:
            all_ids.append(np.empty((0,), dtype=np.int32))
            offsets[cid + 1] = offsets[cid]
            continue

        member_vecs = data_vectors[members]
        dists = np.sum((member_vecs - centers[cid][None, :]) ** 2, axis=1)
        take = min(anchor_k, members.size)
        sel = np.argpartition(dists, take - 1)[:take]
        sel = sel[np.argsort(dists[sel])]
        graph_ids = (members[sel] + virtual_base_count).astype(np.int32, copy=False)
        all_ids.append(graph_ids)
        offsets[cid + 1] = offsets[cid] + graph_ids.size

    if offsets[-1] == 0:
        indices = np.empty((0,), dtype=np.int32)
    else:
        indices = np.concatenate(all_ids).astype(np.int32, copy=False)
    return offsets, indices


class TwoLayerHNSWLikeIndexCPP:
    """
    C++ core TLHL implementation.

    Python is responsible for:
    - balanced clustering
    - flattening the router tree
    - extracting center anchors

    C++ is responsible for:
    - center/base graph construction
    - CSR conversion
    - query path
    """

    def __init__(
        self,
        n_centers: int,
        m: int = 16,
        ef_construction: int = 200,
        center_max_degree: Optional[int] = None,
        base_max_degree: Optional[int] = None,
        random_state: int = 42,
        cluster_method: str = "cpd_kmeans",
        cluster_max_iter: int = 20,
        cluster_tol: float = 1e-6,
        cluster_train_sample_size: Optional[int] = None,
        cpd_psd_sample: int = 64,
        cpd_psd_exact_k: int = 512,
        cpd_mean_sample: int = 64,
        cpd_mean_exact_k: int = 512,
        anchor_k: int = 4,
        ef_extra: int = 64,
        center_probe_ef_cap: int = 16,
    ):
        self.params = IndexParams(
            n_centers=n_centers,
            m=m,
            ef_construction=ef_construction,
            center_max_degree=center_max_degree,
            base_max_degree=base_max_degree,
            random_state=random_state,
            cluster_method=cluster_method,
            cluster_max_iter=cluster_max_iter,
            cluster_tol=cluster_tol,
            cluster_train_sample_size=cluster_train_sample_size,
            cpd_psd_sample=cpd_psd_sample,
            cpd_psd_exact_k=cpd_psd_exact_k,
            cpd_mean_sample=cpd_mean_sample,
            cpd_mean_exact_k=cpd_mean_exact_k,
            anchor_k=anchor_k,
            ef_extra=ef_extra,
            center_probe_ef_cap=center_probe_ef_cap,
        )

        self.data_vectors: Optional[np.ndarray] = None
        self.base_vectors: Optional[np.ndarray] = None
        self.centers: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.virtual_base_count: int = 0
        self.center_router: Optional[CenterHyperplaneRouter] = None
        self.core: Optional[TLHLCore] = None

    def _cluster_points(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.params.cluster_method == "kmeans":
            kmeans = KMeans(
                n_clusters=self.params.n_centers,
                random_state=self.params.random_state,
                n_init=10,
                max_iter=self.params.cluster_max_iter,
                tol=self.params.cluster_tol,
            )
            labels = kmeans.fit_predict(X)
            centers = kmeans.cluster_centers_.astype(np.float32, copy=False)
            self.center_router = CenterHyperplaneRouter(centers)
            return centers, labels.astype(np.int32, copy=False)

        cpd = CPD_KMeans(
            data=X,
            k=self.params.n_centers,
            max_iter=self.params.cluster_max_iter,
            tol=self.params.cluster_tol,
            random_state=self.params.random_state,
            train_sample_size=self.params.cluster_train_sample_size,
            psd_sample=self.params.cpd_psd_sample,
            psd_exact_k=self.params.cpd_psd_exact_k,
            mean_sample=self.params.cpd_mean_sample,
            mean_exact_k=self.params.cpd_mean_exact_k,
        )
        centers, labels = cpd.train()
        centers = np.asarray(centers, dtype=np.float32, order="C")
        labels = np.asarray(labels, dtype=np.int32)
        self.center_router = cpd.router_
        return centers, labels

    def fit(self, X: np.ndarray) -> "TwoLayerHNSWLikeIndexCPP":
        X = _to_float32_matrix(X)
        N = len(X)
        if self.params.n_centers > N:
            raise ValueError(f"n_centers={self.params.n_centers} 不能大于样本数 N={N}")

        self.data_vectors = X
        self.centers, self.labels_ = self._cluster_points(X)
        self.virtual_base_count = int(self.centers.shape[0])
        self.base_vectors = np.ascontiguousarray(np.vstack([self.centers, self.data_vectors]).astype(np.float32))

        if self.center_router is not None:
            insertion_entry_points = self.center_router.route_many(self.data_vectors).astype(np.int32, copy=False)
        else:
            insertion_entry_points = self.labels_.astype(np.int32, copy=False)

        router_normals, router_bias, router_left, router_right, router_leaf_center = _flatten_router(
            self.center_router, self.base_vectors.shape[1]
        )
        anchor_offsets, anchor_indices = _build_center_anchors(
            centers=self.centers,
            labels=self.labels_,
            data_vectors=self.data_vectors,
            virtual_base_count=self.virtual_base_count,
            anchor_k=self.params.anchor_k,
        )

        self.core = TLHLCore(
            m=int(self.params.m),
            ef_construction=int(self.params.ef_construction),
            center_max_degree=int(self.params.center_max_degree),
            base_max_degree=int(self.params.base_max_degree),
            ef_extra=int(self.params.ef_extra),
            anchor_k=int(self.params.anchor_k),
            center_probe_ef_cap=int(self.params.center_probe_ef_cap),
        )
        self.core.build(
            np.ascontiguousarray(self.centers),
            np.ascontiguousarray(self.base_vectors),
            np.ascontiguousarray(insertion_entry_points, dtype=np.int32),
        )
        self.core.set_router(
            np.ascontiguousarray(router_normals),
            np.ascontiguousarray(router_bias),
            np.ascontiguousarray(router_left),
            np.ascontiguousarray(router_right),
            np.ascontiguousarray(router_leaf_center),
        )
        self.core.set_center_anchors(
            np.ascontiguousarray(anchor_offsets),
            np.ascontiguousarray(anchor_indices),
        )
        return self

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        ef: Optional[int] = None,
        n_probe_centers: int = 1,
    ) -> List[Tuple[int, float]]:
        if self.core is None:
            raise RuntimeError("请先 fit()")
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        if ef is None:
            ef = max(self.params.ef_construction, k * 4)
        ids, dists = self.core.search_arrays(q, int(k), int(ef), int(n_probe_centers))
        return [(int(i), float(d)) for i, d in zip(ids.tolist(), dists.tolist())]

    def search_many(
        self,
        queries: np.ndarray,
        k: int = 10,
        ef: Optional[int] = None,
        n_probe_centers: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.core is None:
            raise RuntimeError("请先 fit()")
        Q = _to_float32_matrix(queries)
        if ef is None:
            ef = max(self.params.ef_construction, k * 4)
        return self.core.search_many_arrays(Q, int(k), int(ef), int(n_probe_centers))

    def summary(self) -> dict:
        if self.core is None:
            return {
                "built": False,
                "cluster_method": self.params.cluster_method,
            }
        out = dict(self.core.summary())
        out.update(
            {
                "built": True,
                "cluster_method": self.params.cluster_method,
                "n_real_base_nodes": 0 if self.data_vectors is None else int(self.data_vectors.shape[0]),
                "n_virtual_base_nodes": int(self.virtual_base_count),
                "n_centers": 0 if self.centers is None else int(self.centers.shape[0]),
                "anchor_k": int(self.params.anchor_k),
                "ef_extra": int(self.params.ef_extra),
                "query_backend": "c++",
            }
        )
        if self.labels_ is not None and self.labels_.size > 0:
            counts = np.bincount(self.labels_, minlength=self.params.n_centers)
            out["cluster_balance"] = {
                "min": int(counts.min()),
                "max": int(counts.max()),
                "mean": float(counts.mean()),
                "std": float(counts.std()),
            }
        return out
