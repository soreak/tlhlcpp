from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Tuple

import numpy as np

from ._tlhl_cpp import TLHLCore

if TYPE_CHECKING:
    from .CPD_Kmeans_old import CenterHyperplaneRouter


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

    # query / structure
    anchor_k: int = 4
    ef_extra: int = 64
    center_probe_ef_cap: int = 16

    adaptive_probe: bool = True
    adaptive_ef_extra: bool = True
    min_probe_centers: int = 1
    max_probe_centers: int = 8
    route_margin_low: float = 0.05
    route_margin_high: float = 0.20

    # parallel search_many
    num_threads: int = 0

    # virtual node rewrite
    finalize_virtual_nodes: bool = True

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
        if self.min_probe_centers <= 0:
            raise ValueError("min_probe_centers 必须 > 0")
        if self.max_probe_centers < self.min_probe_centers:
            raise ValueError("max_probe_centers 必须 >= min_probe_centers")
        if self.route_margin_low < 0 or self.route_margin_high < 0:
            raise ValueError("route margin 必须 >= 0")
        if self.route_margin_high < self.route_margin_low:
            raise ValueError("route_margin_high 必须 >= route_margin_low")
        if self.num_threads < 0:
            raise ValueError("num_threads 必须 >= 0")

        self.cluster_method = str(self.cluster_method).lower()
        if self.cluster_method not in {"cpd_kmeans", "kmeans"}:
            raise ValueError("cluster_method 必须是 'cpd_kmeans' 或 'kmeans'")


def _to_float32_matrix(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError("输入数据 X 必须是二维数组 [N, dim]")
    return np.ascontiguousarray(X)


def _flatten_router(router: Optional[Any], dim: int):
    if router is None:
        return (
            np.empty((0, dim), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
        )

    normals = []
    normal_norms = []
    bias = []
    left = []
    right = []
    leaf_center = []

    def build(node) -> int:
        idx = len(bias)
        normals.append(np.zeros((dim,), dtype=np.float32))
        normal_norms.append(0.0)
        bias.append(0.0)
        left.append(-1)
        right.append(-1)
        leaf_center.append(-1)

        if node.is_leaf:
            leaf_center[idx] = int(node.center_ids[0])
            return idx

        n = np.asarray(node.normal, dtype=np.float32)
        normals[idx] = n
        normal_norms[idx] = float(np.linalg.norm(n))
        bias[idx] = float(node.bias)
        left[idx] = build(node.left)
        right[idx] = build(node.right)
        return idx

    build(router.root)
    return (
        np.ascontiguousarray(np.vstack(normals).astype(np.float32)),
        np.asarray(normal_norms, dtype=np.float32),
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
    chunks = []

    for cid in range(K):
        members = np.where(labels == cid)[0]
        if members.size == 0:
            chunks.append(np.empty((0,), dtype=np.int32))
            offsets[cid + 1] = offsets[cid]
            continue

        member_vecs = data_vectors[members]
        dists = np.sum((member_vecs - centers[cid][None, :]) ** 2, axis=1)
        take = min(anchor_k, members.size)
        sel = np.argpartition(dists, take - 1)[:take]
        sel = sel[np.argsort(dists[sel])]
        graph_ids = (members[sel] + virtual_base_count).astype(np.int32, copy=False)
        chunks.append(graph_ids)
        offsets[cid + 1] = offsets[cid] + graph_ids.size

    if offsets[-1] == 0:
        indices = np.empty((0,), dtype=np.int32)
    else:
        indices = np.concatenate(chunks).astype(np.int32, copy=False)
    return offsets, indices


class TwoLayerHNSWLikeIndexCPP:
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
        adaptive_probe: bool = True,
        adaptive_ef_extra: bool = True,
        min_probe_centers: int = 1,
        max_probe_centers: int = 8,
        route_margin_low: float = 0.05,
        route_margin_high: float = 0.20,
        num_threads: int = 0,
        finalize_virtual_nodes: bool = True,
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
            adaptive_probe=adaptive_probe,
            adaptive_ef_extra=adaptive_ef_extra,
            min_probe_centers=min_probe_centers,
            max_probe_centers=max_probe_centers,
            route_margin_low=route_margin_low,
            route_margin_high=route_margin_high,
            num_threads=num_threads,
            finalize_virtual_nodes=finalize_virtual_nodes,
        )

        self.data_vectors: Optional[np.ndarray] = None
        self.base_vectors: Optional[np.ndarray] = None
        self.centers: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.virtual_base_count: int = 0
        self.center_router: Optional[Any] = None
        self.core: Optional[TLHLCore] = None

    def _cluster_kmeans(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        try:
            from sklearn.cluster import KMeans
        except Exception as e:
            raise RuntimeError(f"导入 sklearn.cluster.KMeans 失败: {e}") from e

        try:
            from .CPD_Kmeans_old import CenterHyperplaneRouter
        except Exception as e:
            raise RuntimeError(f"导入 CenterHyperplaneRouter 失败: {e}") from e

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

    def _cluster_cpd(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        try:
            from .CPD_Kmeans_old import CPD_KMeans
        except Exception as e:
            raise RuntimeError(f"导入 CPD_KMeans 失败: {e}") from e

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
        self.center_router = cpd.router_
        return (
            np.asarray(centers, dtype=np.float32, order="C"),
            np.asarray(labels, dtype=np.int32),
        )

    def _cluster_points(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self.center_router = None
        if self.params.cluster_method == "kmeans":
            return self._cluster_kmeans(X)
        return self._cluster_cpd(X)

    def _create_core(self) -> TLHLCore:
        return TLHLCore(
            self.params.m,
            self.params.ef_construction,
            self.params.center_max_degree,
            self.params.base_max_degree,
            self.params.ef_extra,
            self.params.anchor_k,
            self.params.center_probe_ef_cap,
            self.params.min_probe_centers,
            self.params.max_probe_centers,
            float(self.params.route_margin_low),
            float(self.params.route_margin_high),
            bool(self.params.adaptive_probe),
            bool(self.params.adaptive_ef_extra),
            int(self.params.num_threads),
        )

    def fit(self, X: np.ndarray) -> "TwoLayerHNSWLikeIndexCPP":
        X = _to_float32_matrix(X)
        N = len(X)
        if self.params.n_centers > N:
            raise ValueError(f"n_centers={self.params.n_centers} 不能大于样本数 N={N}")

        self.data_vectors = X
        self.centers, self.labels_ = self._cluster_points(X)
        self.virtual_base_count = int(self.centers.shape[0])
        self.base_vectors = np.ascontiguousarray(
            np.vstack([self.centers, self.data_vectors]).astype(np.float32)
        )

        if self.center_router is not None and hasattr(self.center_router, "route_many"):
            insertion_entry_points = self.center_router.route_many(self.data_vectors).astype(
                np.int32, copy=False
            )
        else:
            insertion_entry_points = self.labels_.astype(np.int32, copy=False)

        anchor_offsets, anchor_indices = _build_center_anchors(
            centers=self.centers,
            labels=self.labels_,
            data_vectors=self.data_vectors,
            virtual_base_count=self.virtual_base_count,
            anchor_k=self.params.anchor_k,
        )

        router_normals, router_norms, router_bias, router_left, router_right, router_leaf_center = _flatten_router(
            self.center_router, self.base_vectors.shape[1]
        )

        self.core = self._create_core()
        self.core.build(self.centers, self.base_vectors, insertion_entry_points)
        self.core.set_router(
            router_normals,
            router_norms,
            router_bias,
            router_left,
            router_right,
            router_leaf_center,
        )
        self.core.set_center_anchors(anchor_offsets, anchor_indices)

        if self.params.finalize_virtual_nodes:
            self.core.finalize_virtual_nodes()

        return self

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        ef: int = 50,
        n_probe_centers: int = 1,
    ) -> list[tuple[int, float]]:
        if self.core is None:
            raise RuntimeError("请先 fit()")

        ids, dists = self.core.search_arrays(
            np.ascontiguousarray(np.asarray(query, dtype=np.float32)),
            int(k),
            int(ef),
            int(n_probe_centers),
        )
        ids = np.asarray(ids, dtype=np.int32)
        dists = np.asarray(dists, dtype=np.float32)
        return [(int(i), float(d)) for i, d in zip(ids, dists)]

    def search_many(
        self,
        queries: np.ndarray,
        k: int = 10,
        ef: int = 50,
        n_probe_centers: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.core is None:
            raise RuntimeError("请先 fit()")

        ids, dists = self.core.search_many_arrays(
            _to_float32_matrix(queries),
            int(k),
            int(ef),
            int(n_probe_centers),
        )
        return np.asarray(ids, dtype=np.int32), np.asarray(dists, dtype=np.float32)

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
                "n_centers": self.params.n_centers,
                "m": self.params.m,
                "ef_construction": self.params.ef_construction,
                "anchor_k": self.params.anchor_k,
                "ef_extra": self.params.ef_extra,
                "adaptive_probe": self.params.adaptive_probe,
                "adaptive_ef_extra": self.params.adaptive_ef_extra,
                "min_probe_centers": self.params.min_probe_centers,
                "max_probe_centers": self.params.max_probe_centers,
                "route_margin_low": self.params.route_margin_low,
                "route_margin_high": self.params.route_margin_high,
                "num_threads": self.params.num_threads,
                "finalize_virtual_nodes": self.params.finalize_virtual_nodes,
            }
        )
        return out