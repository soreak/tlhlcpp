from __future__ import annotations
import numpy as np
import faiss

from dataclasses import dataclass
from typing import Optional


@dataclass
class _RouterNode:
    center_ids: np.ndarray
    normal: Optional[np.ndarray] = None
    bias: float = 0.0
    left: Optional["_RouterNode"] = None
    right: Optional["_RouterNode"] = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class CenterHyperplaneRouter:
    """
    用中心点构建一棵二分超平面路由树。

    每个内部节点保存一个超平面：
        score(x) = dot(normal, x) + bias
    score <= 0 -> left
    score >  0 -> right

    叶子节点保存一个中心 id。
    """

    def __init__(self, centers: np.ndarray):
        self.centers = np.asarray(centers, dtype=np.float32, order="C")
        if self.centers.ndim != 2:
            raise ValueError("centers 必须是二维矩阵")
        K = self.centers.shape[0]
        if K == 0:
            raise ValueError("centers 不能为空")

        ids = np.arange(K, dtype=np.int32)
        self.root = self._build(ids)

    def _build(self, ids: np.ndarray) -> _RouterNode:
        ids = np.asarray(ids, dtype=np.int32)
        if ids.size == 1:
            return _RouterNode(center_ids=ids.copy())

        C = self.centers[ids]
        mean = C.mean(axis=0, dtype=np.float32)
        centered = C - mean[None, :]

        direction = None
        if centered.shape[0] >= 2:
            try:
                _, s, vt = np.linalg.svd(centered, full_matrices=False)
                if s.size > 0 and float(s[0]) > 1e-8:
                    direction = vt[0].astype(np.float32, copy=False)
            except np.linalg.LinAlgError:
                direction = None

        if direction is None:
            var = C.var(axis=0)
            direction = np.zeros(C.shape[1], dtype=np.float32)
            direction[int(np.argmax(var))] = 1.0

        proj = C @ direction
        order = np.argsort(proj, kind="mergesort")

        mid = ids.size // 2
        left_ids = ids[order[:mid]]
        right_ids = ids[order[mid:]]

        if left_ids.size == 0 or right_ids.size == 0:
            half = max(1, ids.size // 2)
            left_ids = ids[:half]
            right_ids = ids[half:]

        left_mean = self.centers[left_ids].mean(axis=0, dtype=np.float32)
        right_mean = self.centers[right_ids].mean(axis=0, dtype=np.float32)

        normal = (right_mean - left_mean).astype(np.float32, copy=False)
        norm = float(np.linalg.norm(normal))

        if norm > 1e-8:
            midpoint = 0.5 * (left_mean + right_mean)
            bias = -float(np.dot(normal, midpoint))
        else:
            normal = direction.astype(np.float32, copy=False)
            threshold = float(np.median(proj))
            bias = -threshold

        return _RouterNode(
            center_ids=ids.copy(),
            normal=normal,
            bias=bias,
            left=self._build(left_ids),
            right=self._build(right_ids),
        )

    def route(self, x: np.ndarray) -> int:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        node = self.root
        while not node.is_leaf:
            score = float(np.dot(node.normal, x) + node.bias)
            node = node.left if score <= 0.0 else node.right
        return int(node.center_ids[0])

    def route_many(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32, order="C")
        out = np.empty((X.shape[0],), dtype=np.int32)
        for i in range(X.shape[0]):
            out[i] = self.route(X[i])
        return out

class CPD_KMeans:
    """
    CPD-KMeans balanced clustering.

    保留当前项目实际使用到的最小集合：
      - KMeans++ init
      - assign -> PSD -> weight -> weighted center update -> converge
      - train() 返回 (centers, labels_)
    """

    def __init__(
        self,
        data: np.ndarray,
        k: int,
        max_iter: int = 100,
        tol: float = 1e-6,
        random_state: int = 42,
        train_sample_size: int | None = None,
        psd_sample: int = 64,
        psd_exact_k: int = 512,
        mean_sample: int = 64,
        mean_exact_k: int = 512,
    ) -> None:
        self.data_full = np.asarray(data, dtype=np.float32, order="C")
        self.k = int(k)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = int(random_state) if random_state is not None else 42
        self.train_sample_size = train_sample_size

        self.psd_sample = int(psd_sample) if psd_sample is not None else None
        self.psd_exact_k = int(psd_exact_k)
        self.mean_sample = int(mean_sample) if mean_sample is not None else None
        self.mean_exact_k = int(mean_exact_k)

        self.n_full, self.m = self.data_full.shape
        self.centers: np.ndarray | None = None
        self.labels_: np.ndarray | None = None

        self._rng = np.random.RandomState(self.random_state)

        if train_sample_size is not None and int(train_sample_size) < self.n_full:
            samp = self._rng.choice(self.n_full, int(train_sample_size), replace=False)
            self.data = self.data_full[samp]
        else:
            self.data = self.data_full

        self.n = self.data.shape[0]
        self._assign_index = faiss.IndexFlatL2(self.m)
        self.router_: CenterHyperplaneRouter | None = None

    def init_cluster_center(self) -> np.ndarray:
        X = self.data
        n, d = X.shape
        centers = np.empty((self.k, d), dtype=np.float32)

        centers[0] = X[self._rng.randint(n)]
        diff = X - centers[0][None, :]
        dist2 = np.einsum("ij,ij->i", diff, diff).astype(np.float32)

        for i in range(1, self.k):
            s = float(dist2.sum())
            if s <= 0.0:
                centers[i] = X[self._rng.randint(n)]
                continue
            probs = dist2 / np.float32(s)
            r = self._rng.rand()
            idx = np.searchsorted(np.cumsum(probs), r)
            if idx >= n:
                idx = n - 1
            centers[i] = X[idx]

            diff = X - centers[i][None, :]
            new_d2 = np.einsum("ij,ij->i", diff, diff).astype(np.float32)
            dist2 = np.minimum(dist2, new_d2)

        return centers

    def assign_clusters(self, centers: np.ndarray, X: np.ndarray) -> np.ndarray:
        centers = np.asarray(centers, dtype=np.float32, order="C")
        X = np.asarray(X, dtype=np.float32, order="C")

        self._assign_index.reset()
        self._assign_index.add(centers)
        _, labels = self._assign_index.search(X, 1)
        return labels.reshape(-1).astype(np.int32, copy=False)

    def calculate_potential_difference(
        self,
        labels: np.ndarray,
        centers: np.ndarray,
    ) -> np.ndarray:
        K = self.k
        centers = np.asarray(centers, dtype=np.float32, order="C")
        labels = np.asarray(labels, dtype=np.int32)

        cluster_sizes = np.bincount(labels, minlength=K).astype(np.int32)
        PSD = np.zeros_like(centers, dtype=np.float32)

        exact_k = int(self.psd_exact_k)
        psd_samp = int(self.psd_sample) if self.psd_sample is not None else K

        if K <= exact_k:
            for i in range(K):
                ci = centers[i]
                si = cluster_sizes[i]
                for j in range(K):
                    if i == j:
                        continue
                    sj = cluster_sizes[j]
                    sdiff = sj - si
                    if sdiff == 0:
                        continue
                    sign = -1.0 if sdiff < 0 else 1.0
                    v = (centers[j] - ci) * np.float32(sign)
                    norm = np.sqrt(np.dot(v, v))
                    if norm <= 1e-10:
                        continue
                    unit = v / np.float32(norm)
                    w = np.float32(abs(sdiff) / float(self.n))
                    PSD[i] += unit * w
            return PSD

        for i in range(K):
            ci = centers[i]
            si = cluster_sizes[i]
            m = psd_samp

            if m >= K - 1:
                for j in range(K):
                    if i == j:
                        continue
                    sj = cluster_sizes[j]
                    sdiff = sj - si
                    if sdiff == 0:
                        continue
                    sign = -1.0 if sdiff < 0 else 1.0
                    v = (centers[j] - ci) * np.float32(sign)
                    norm = np.sqrt(np.dot(v, v))
                    if norm <= 1e-10:
                        continue
                    unit = v / np.float32(norm)
                    w = np.float32(abs(sdiff) / float(self.n))
                    PSD[i] += unit * w
                continue

            for _ in range(m):
                j = int(self._rng.randint(0, K - 1))
                if j >= i:
                    j += 1
                sj = cluster_sizes[j]
                sdiff = sj - si
                if sdiff == 0:
                    continue

                sign = -1.0 if sdiff < 0 else 1.0
                v = (centers[j] - ci) * np.float32(sign)
                norm = np.sqrt(np.dot(v, v))
                if norm <= 1e-10:
                    continue
                unit = v / np.float32(norm)
                w = np.float32(abs(sdiff) / float(self.n))
                PSD[i] += unit * w

            PSD[i] *= np.float32(float(K - 1) / float(m))

        return PSD

    def calculate_weight(
        self,
        labels: np.ndarray,
        centers: np.ndarray,
        PSD: np.ndarray,
    ) -> np.ndarray:
        X = self.data
        K = self.k
        n = X.shape[0]

        centers = np.asarray(centers, dtype=np.float32, order="C")
        PSD = np.asarray(PSD, dtype=np.float32, order="C")
        labels = np.asarray(labels, dtype=np.int32)

        weight = np.ones(n, dtype=np.float32)
        mean_center_dist = np.zeros((K,), dtype=np.float32)

        exact_k = int(self.mean_exact_k)
        mean_samp = int(self.mean_sample) if self.mean_sample is not None else K

        if K <= exact_k:
            for i in range(K):
                ci = centers[i]
                acc = np.float32(0.0)
                cnt = 0
                for j in range(K):
                    if i == j:
                        continue
                    diff = centers[j] - ci
                    acc += np.float32(np.sqrt(np.dot(diff, diff)))
                    cnt += 1
                if cnt > 0:
                    mean_center_dist[i] = acc / np.float32(cnt)
        else:
            m = mean_samp
            if m >= K - 1:
                m = K - 1
            for i in range(K):
                ci = centers[i]
                acc = np.float32(0.0)
                for _ in range(m):
                    j = int(self._rng.randint(0, K - 1))
                    if j >= i:
                        j += 1
                    diff = centers[j] - ci
                    acc += np.float32(np.sqrt(np.dot(diff, diff)))
                mean_center_dist[i] = acc / np.float32(max(m, 1))

        for i in range(K):
            idx = np.where(labels == i)[0]
            if idx.size == 0:
                continue

            psd = PSD[i]
            psd_norm = float(np.sqrt(np.dot(psd, psd)))
            if psd_norm <= 0.0:
                continue

            Xi = X[idx] - centers[i]
            x_norm = np.sqrt(np.einsum("ij,ij->i", Xi, Xi)).astype(np.float32)
            x_norm = np.maximum(x_norm, 1e-10)

            cos = (Xi @ psd) / (x_norm * np.float32(psd_norm))
            gain = (
                np.float32(1.0)
                + np.float32(0.5)
                * np.float32(np.sqrt(mean_center_dist[i]))
                * np.float32(psd_norm)
            )
            weight[idx] = np.where(cos > 0, gain, np.float32(1.0))

        return weight

    def compute_cluster_centers(
        self,
        labels: np.ndarray,
        weight: np.ndarray,
    ) -> np.ndarray:
        X = self.data
        K = self.k
        d = self.m

        centers = np.zeros((K, d), dtype=np.float32)

        for j in range(K):
            mask = labels == j
            if not np.any(mask):
                centers[j] = X[self._rng.randint(self.n)]
                continue

            w = weight[mask][:, None]
            sw = float(w.sum())
            if sw <= 1e-12:
                centers[j] = X[self._rng.randint(self.n)]
                continue
            centers[j] = (X[mask] * w).sum(axis=0) / np.float32(sw)

        return centers

    def has_converged(self, old: np.ndarray, new: np.ndarray) -> bool:
        return float(np.linalg.norm(new - old)) < self.tol

    def train(self) -> tuple[np.ndarray, np.ndarray]:
        centers = self.init_cluster_center().astype(np.float32, copy=False)

        for _ in range(self.max_iter):
            old = centers.copy()
            labels = self.assign_clusters(centers, self.data)
            PSD = self.calculate_potential_difference(labels, centers)
            weight = self.calculate_weight(labels, centers, PSD)
            centers = self.compute_cluster_centers(labels, weight)
            if self.has_converged(old, centers):
                break

        self.centers = centers.astype(np.float32, copy=False)
        self.labels_ = self.assign_clusters(self.centers, self.data_full)
        self.router_ = CenterHyperplaneRouter(self.centers)
        return self.centers, self.labels_
