from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ._tlhl_cpp import TLHLCore


@dataclass
class IndexParams:
    n_centers: int
    m: int = 16
    ef_construction: int = 200
    center_max_degree: Optional[int] = None
    base_max_degree: Optional[int] = None
    random_state: int = 42

    cluster_method: str = "cpd_kmeans"
    cluster_max_iter: int = 20
    cluster_tol: float = 1e-6
    cluster_train_sample_size: Optional[int] = None
    cpd_psd_sample: int = 64
    cpd_psd_exact_k: int = 512
    cpd_mean_sample: int = 64
    cpd_mean_exact_k: int = 512

    anchor_k: int = 4
    ef_extra: int = 64
    center_probe_ef_cap: int = 16
    adaptive_probe: bool = True
    adaptive_ef_extra: bool = True
    min_probe_centers: int = 1
    max_probe_centers: int = 8
    route_margin_low: float = 0.05
    route_margin_high: float = 0.20
    num_threads: int = 0
    finalize_virtual_nodes: bool = True

    def __post_init__(self) -> None:
        if self.n_centers <= 0:
            raise ValueError("n_centers must be > 0")
        if self.m <= 0:
            raise ValueError("m must be > 0")
        if self.ef_construction <= 0:
            raise ValueError("ef_construction must be > 0")
        if self.cluster_max_iter <= 0:
            raise ValueError("cluster_max_iter must be > 0")
        if self.cluster_tol <= 0:
            raise ValueError("cluster_tol must be > 0")
        if self.center_max_degree is None:
            self.center_max_degree = 2 * self.m
        if self.base_max_degree is None:
            self.base_max_degree = 2 * self.m
        self.cluster_method = str(self.cluster_method).lower()
        if self.cluster_method not in {"cpd_kmeans", "kmeans"}:
            raise ValueError("cluster_method must be 'cpd_kmeans' or 'kmeans'")


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
    ) -> None:
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
        self.core = TLHLCore(
            self.params.m,
            self.params.ef_construction,
            int(self.params.center_max_degree),
            int(self.params.base_max_degree),
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
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        train_sample_size = -1 if self.params.cluster_train_sample_size is None else int(self.params.cluster_train_sample_size)
        self.core.fit_auto(
            X,
            self.params.cluster_method,
            int(self.params.n_centers),
            int(self.params.cluster_max_iter),
            float(self.params.cluster_tol),
            int(self.params.random_state),
            train_sample_size,
            int(self.params.cpd_psd_sample),
            int(self.params.cpd_psd_exact_k),
            int(self.params.cpd_mean_sample),
            int(self.params.cpd_mean_exact_k),
            bool(self.params.finalize_virtual_nodes),
        )
        return self

    def search(self, query: np.ndarray, k: int = 10, ef: int = 50) -> list[tuple[int, float]]:
        ids, dists = self.core.search_arrays(
            np.ascontiguousarray(np.asarray(query, dtype=np.float32)),
            int(k),
            int(ef),
            1,
        )
        ids = np.asarray(ids, dtype=np.int32)
        dists = np.asarray(dists, dtype=np.float32)
        return [(int(i), float(d)) for i, d in zip(ids, dists)]

    def search_many(self, queries: np.ndarray, k: int = 10, ef: int = 50) -> tuple[np.ndarray, np.ndarray]:
        ids, dists = self.core.search_many_arrays(
            np.ascontiguousarray(np.asarray(queries, dtype=np.float32)),
            int(k),
            int(ef),
            1,
        )
        return np.asarray(ids, dtype=np.int32), np.asarray(dists, dtype=np.float32)

    def summary(self) -> dict:
        return dict(self.core.summary())
