#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <limits>
#include <queue>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>
#include <thread>
#include <string>
#include <random>
#include <numeric>
#include <functional>

namespace py = pybind11;

namespace {

inline float l2_sq_ptr(const float* a, const float* b, int dim) {
    float s = 0.0f;
    for (int i = 0; i < dim; ++i) {
        const float d = a[i] - b[i];
        s += d * d;
    }
    return s;
}

struct CandidateDist {
    int id;
    float dist;
};

struct MinCand {
    float dist;
    int id;
    bool operator>(const MinCand& other) const { return dist > other.dist; }
};

struct MaxBest {
    float dist;
    int id;
    bool operator<(const MaxBest& other) const { return dist < other.dist; }
};

struct CSRGraph {
    std::vector<int64_t> offsets;
    std::vector<int32_t> indices;

    int n_nodes() const {
        return offsets.empty() ? 0 : static_cast<int>(offsets.size() - 1);
    }
};

struct SearchWorkspace {
    std::vector<int> visited;
    int visit_token = 1;

    explicit SearchWorkspace(int n = 0) : visited(static_cast<size_t>(std::max(1, n)), 0) {}

    void resize(int n) {
        visited.assign(static_cast<size_t>(std::max(1, n)), 0);
        visit_token = 1;
    }

    void next_token() {
        ++visit_token;
        if (visit_token == std::numeric_limits<int>::max()) {
            std::fill(visited.begin(), visited.end(), 0);
            visit_token = 1;
        }
    }
};

struct RouteDecision {
    int center_id = 0;
    float margin = std::numeric_limits<float>::infinity();
};

std::vector<int> unique_keep_order(const std::vector<int>& in) {
    std::unordered_set<int> seen;
    std::vector<int> out;
    out.reserve(in.size());
    for (int x : in) {
        if (seen.insert(x).second) out.push_back(x);
    }
    return out;
}

std::vector<int> exact_sorted_ids(
    const float* query,
    const std::vector<float>& vectors,
    int dim,
    const std::vector<int>& ids
) {
    std::vector<CandidateDist> items;
    items.reserve(ids.size());
    for (int id : ids) {
        items.push_back({id, l2_sq_ptr(query, &vectors[static_cast<size_t>(id) * dim], dim)});
    }
    std::sort(items.begin(), items.end(), [](const CandidateDist& a, const CandidateDist& b) {
        return a.dist < b.dist;
    });
    std::vector<int> out;
    out.reserve(items.size());
    for (const auto& it : items) out.push_back(it.id);
    return out;
}

std::vector<int> heuristic_select(
    const float* query,
    const std::vector<int>& candidate_ids,
    const std::vector<float>& vectors,
    int dim,
    int max_neighbors
) {
    if (max_neighbors <= 0) return {};

    std::vector<int> ordered = unique_keep_order(candidate_ids);
    std::vector<CandidateDist> items;
    items.reserve(ordered.size());
    for (int cid : ordered) {
        items.push_back({cid, l2_sq_ptr(query, &vectors[static_cast<size_t>(cid) * dim], dim)});
    }
    std::sort(items.begin(), items.end(), [](const CandidateDist& a, const CandidateDist& b) {
        return a.dist < b.dist;
    });

    std::vector<int> selected;
    selected.reserve(static_cast<size_t>(std::min<int>(max_neighbors, static_cast<int>(items.size()))));

    for (const auto& cand : items) {
        bool ok = true;
        for (int sid : selected) {
            const float d_sc = l2_sq_ptr(
                &vectors[static_cast<size_t>(sid) * dim],
                &vectors[static_cast<size_t>(cand.id) * dim],
                dim
            );
            if (d_sc < cand.dist) {
                ok = false;
                break;
            }
        }
        if (ok) {
            selected.push_back(cand.id);
            if (static_cast<int>(selected.size()) >= max_neighbors) break;
        }
    }

    if (static_cast<int>(selected.size()) < std::min<int>(max_neighbors, static_cast<int>(items.size()))) {
        std::unordered_set<int> used(selected.begin(), selected.end());
        for (const auto& cand : items) {
            if (used.find(cand.id) != used.end()) continue;
            selected.push_back(cand.id);
            used.insert(cand.id);
            if (static_cast<int>(selected.size()) >= max_neighbors) break;
        }
    }

    return selected;
}

void mutual_connect(
    int new_id,
    const std::vector<int>& selected_in,
    const std::vector<float>& vectors,
    int dim,
    std::vector<std::vector<int>>& adj,
    int new_degree_limit,
    int old_degree_limit
) {
    if (new_id < 0 || new_id >= static_cast<int>(adj.size())) {
        throw std::out_of_range("new_id out of range");
    }

    std::vector<int> selected;
    selected.reserve(selected_in.size());
    for (int x : selected_in) {
        if (x != new_id) selected.push_back(x);
    }
    if (static_cast<int>(selected.size()) > new_degree_limit) {
        selected = heuristic_select(
            &vectors[static_cast<size_t>(new_id) * dim],
            selected,
            vectors,
            dim,
            new_degree_limit
        );
    }

    adj[static_cast<size_t>(new_id)] = selected;

    for (int nb : selected) {
        std::vector<int> merged = adj[static_cast<size_t>(nb)];
        if (std::find(merged.begin(), merged.end(), new_id) == merged.end()) {
            merged.push_back(new_id);
        }
        if (static_cast<int>(merged.size()) <= old_degree_limit) {
            adj[static_cast<size_t>(nb)] = std::move(merged);
        } else {
            std::vector<int> tmp;
            tmp.reserve(merged.size());
            for (int x : merged) {
                if (x != nb) tmp.push_back(x);
            }
            tmp = exact_sorted_ids(&vectors[static_cast<size_t>(nb) * dim], vectors, dim, tmp);
            adj[static_cast<size_t>(nb)] = heuristic_select(
                &vectors[static_cast<size_t>(nb) * dim], tmp, vectors, dim, old_degree_limit);
        }
    }
}

std::vector<int> search_layer_list(
    const float* query,
    const std::vector<int>& entry_points,
    const std::vector<float>& vectors,
    int n_nodes,
    int dim,
    const std::vector<std::vector<int>>& adj,
    int ef,
    SearchWorkspace& ws
) {
    if (n_nodes <= 0) return {};
    ef = std::max(1, std::min(ef, n_nodes));
    ws.next_token();

    std::priority_queue<MinCand, std::vector<MinCand>, std::greater<MinCand>> candidate_heap;
    std::priority_queue<MaxBest> top_heap;

    for (int ep : entry_points) {
        if (ep < 0 || ep >= n_nodes) continue;
        if (ws.visited[static_cast<size_t>(ep)] == ws.visit_token) continue;
        ws.visited[static_cast<size_t>(ep)] = ws.visit_token;
        float d = l2_sq_ptr(query, &vectors[static_cast<size_t>(ep) * dim], dim);
        candidate_heap.push({d, ep});
        top_heap.push({d, ep});
    }
    if (candidate_heap.empty()) {
        int ep = 0;
        float d = l2_sq_ptr(query, &vectors[0], dim);
        candidate_heap.push({d, ep});
        top_heap.push({d, ep});
        ws.visited[0] = ws.visit_token;
    }

    while (!candidate_heap.empty()) {
        auto curr = candidate_heap.top();
        candidate_heap.pop();

        float worst_best = top_heap.top().dist;
        if (curr.dist > worst_best && static_cast<int>(top_heap.size()) >= ef) {
            break;
        }

        for (int nb : adj[static_cast<size_t>(curr.id)]) {
            if (ws.visited[static_cast<size_t>(nb)] == ws.visit_token) continue;
            ws.visited[static_cast<size_t>(nb)] = ws.visit_token;
            float d = l2_sq_ptr(query, &vectors[static_cast<size_t>(nb) * dim], dim);
            if (static_cast<int>(top_heap.size()) < ef || d < worst_best) {
                candidate_heap.push({d, nb});
                top_heap.push({d, nb});
                if (static_cast<int>(top_heap.size()) > ef) {
                    top_heap.pop();
                }
            }
        }
    }

    std::vector<CandidateDist> tmp;
    tmp.reserve(top_heap.size());
    while (!top_heap.empty()) {
        auto x = top_heap.top();
        top_heap.pop();
        tmp.push_back({x.id, x.dist});
    }
    std::sort(tmp.begin(), tmp.end(), [](const CandidateDist& a, const CandidateDist& b) {
        return a.dist < b.dist;
    });
    std::vector<int> out;
    out.reserve(tmp.size());
    for (const auto& x : tmp) out.push_back(x.id);
    return out;
}

std::vector<int> search_layer_csr(
    const float* query,
    const std::vector<int>& entry_points,
    const std::vector<float>& vectors,
    int n_nodes,
    int dim,
    const CSRGraph& graph,
    int ef,
    SearchWorkspace& ws
) {
    if (n_nodes <= 0) return {};
    ef = std::max(1, std::min(ef, n_nodes));
    ws.next_token();

    std::priority_queue<MinCand, std::vector<MinCand>, std::greater<MinCand>> candidate_heap;
    std::priority_queue<MaxBest> top_heap;

    for (int ep : entry_points) {
        if (ep < 0 || ep >= n_nodes) continue;
        if (ws.visited[static_cast<size_t>(ep)] == ws.visit_token) continue;
        ws.visited[static_cast<size_t>(ep)] = ws.visit_token;
        float d = l2_sq_ptr(query, &vectors[static_cast<size_t>(ep) * dim], dim);
        candidate_heap.push({d, ep});
        top_heap.push({d, ep});
    }
    if (candidate_heap.empty()) {
        int ep = 0;
        float d = l2_sq_ptr(query, &vectors[0], dim);
        candidate_heap.push({d, ep});
        top_heap.push({d, ep});
        ws.visited[0] = ws.visit_token;
    }

    while (!candidate_heap.empty()) {
        auto curr = candidate_heap.top();
        candidate_heap.pop();

        float worst_best = top_heap.top().dist;
        if (curr.dist > worst_best && static_cast<int>(top_heap.size()) >= ef) {
            break;
        }

        const int64_t start = graph.offsets[static_cast<size_t>(curr.id)];
        const int64_t end = graph.offsets[static_cast<size_t>(curr.id + 1)];
        for (int64_t p = start; p < end; ++p) {
            int nb = graph.indices[static_cast<size_t>(p)];
            if (ws.visited[static_cast<size_t>(nb)] == ws.visit_token) continue;
            ws.visited[static_cast<size_t>(nb)] = ws.visit_token;
            float d = l2_sq_ptr(query, &vectors[static_cast<size_t>(nb) * dim], dim);
            if (static_cast<int>(top_heap.size()) < ef || d < worst_best) {
                candidate_heap.push({d, nb});
                top_heap.push({d, nb});
                if (static_cast<int>(top_heap.size()) > ef) {
                    top_heap.pop();
                }
            }
        }
    }

    std::vector<CandidateDist> tmp;
    tmp.reserve(top_heap.size());
    while (!top_heap.empty()) {
        auto x = top_heap.top();
        top_heap.pop();
        tmp.push_back({x.id, x.dist});
    }
    std::sort(tmp.begin(), tmp.end(), [](const CandidateDist& a, const CandidateDist& b) {
        return a.dist < b.dist;
    });
    std::vector<int> out;
    out.reserve(tmp.size());
    for (const auto& x : tmp) out.push_back(x.id);
    return out;
}

CSRGraph list_to_csr(const std::vector<std::vector<int>>& adj) {
    CSRGraph graph;
    const int n = static_cast<int>(adj.size());
    graph.offsets.resize(static_cast<size_t>(n) + 1, 0);
    int64_t total = 0;
    for (int i = 0; i < n; ++i) {
        graph.offsets[static_cast<size_t>(i)] = total;
        total += static_cast<int64_t>(adj[static_cast<size_t>(i)].size());
    }
    graph.offsets[static_cast<size_t>(n)] = total;
    graph.indices.resize(static_cast<size_t>(total));
    int64_t pos = 0;
    for (const auto& nbrs : adj) {
        for (int nb : nbrs) graph.indices[static_cast<size_t>(pos++)] = static_cast<int32_t>(nb);
    }
    return graph;
}

}  // namespace

class TLHLCore {
public:
    TLHLCore(
        int m,
        int ef_construction,
        int center_max_degree,
        int base_max_degree,
        int ef_extra = 64,
        int anchor_k = 4,
        int center_probe_ef_cap = 16,
        int min_probe_centers = 1,
        int max_probe_centers = 8,
        float route_margin_low = 0.05f,
        float route_margin_high = 0.20f,
        bool adaptive_probe = true,
        bool adaptive_ef_extra = true,
        int num_threads = 0
    )
        : m_(m),
          ef_construction_(ef_construction),
          center_max_degree_(center_max_degree),
          base_max_degree_(base_max_degree),
          ef_extra_(ef_extra),
          anchor_k_(anchor_k),
          center_probe_ef_cap_(center_probe_ef_cap),
          min_probe_centers_(min_probe_centers),
          max_probe_centers_(max_probe_centers),
          route_margin_low_(route_margin_low),
          route_margin_high_(route_margin_high),
          adaptive_probe_(adaptive_probe),
          adaptive_ef_extra_(adaptive_ef_extra),
          num_threads_(num_threads) {}

    int ping() const { return 1; }
    int version() const { return 7; }

    void build(
        py::array_t<float, py::array::c_style | py::array::forcecast> centers,
        py::array_t<float, py::array::c_style | py::array::forcecast> base_vectors,
        py::array_t<int32_t, py::array::c_style | py::array::forcecast> insertion_entry_points
    ) {
        auto cb = centers.request();
        auto bb = base_vectors.request();
        auto ib = insertion_entry_points.request();

        if (cb.ndim != 2) throw std::runtime_error("centers must be 2D");
        if (bb.ndim != 2) throw std::runtime_error("base_vectors must be 2D");
        if (ib.ndim != 1) throw std::runtime_error("insertion_entry_points must be 1D");

        center_count_ = static_cast<int>(cb.shape[0]);
        dim_ = static_cast<int>(cb.shape[1]);
        base_count_ = static_cast<int>(bb.shape[0]);

        if (static_cast<int>(bb.shape[1]) != dim_) {
            throw std::runtime_error("dim mismatch between centers and base_vectors");
        }
        if (base_count_ < center_count_) {
            throw std::runtime_error("base_count must be >= center_count");
        }
        if (static_cast<int>(ib.shape[0]) != base_count_ - center_count_) {
            throw std::runtime_error("insertion_entry_points length mismatch");
        }

        const float* cptr = static_cast<const float*>(cb.ptr);
        const float* bptr = static_cast<const float*>(bb.ptr);
        const int32_t* iptr = static_cast<const int32_t*>(ib.ptr);

        centers_.assign(cptr, cptr + static_cast<std::size_t>(center_count_) * dim_);
        base_vectors_.assign(bptr, bptr + static_cast<std::size_t>(base_count_) * dim_);
        insertion_entry_points_.assign(iptr, iptr + ib.shape[0]);

        build_center_graph();
        build_base_graph();
        build_center_real_pool(center_real_pool_size_);

        center_graph_ = list_to_csr(center_adj_);
        base_graph_ = list_to_csr(base_adj_);

        built_ = true;
        virtuals_finalized_ = false;
    }



    void fit_auto(
        py::array_t<float, py::array::c_style | py::array::forcecast> X,
        const std::string& cluster_method,
        int n_centers,
        int cluster_max_iter,
        float cluster_tol,
        int random_state,
        int train_sample_size,
        int cpd_psd_sample,
        int cpd_psd_exact_k,
        int cpd_mean_sample,
        int cpd_mean_exact_k,
        bool finalize_virtual_nodes_flag
    ) {
        auto xb = X.request();
        if (xb.ndim != 2) throw std::runtime_error("X must be 2D");
        const int n = static_cast<int>(xb.shape[0]);
        const int d = static_cast<int>(xb.shape[1]);
        if (n <= 0 || d <= 0) throw std::runtime_error("X must be non-empty");
        if (n_centers <= 0 || n_centers > n) throw std::runtime_error("invalid n_centers");
        if (cluster_max_iter <= 0) throw std::runtime_error("cluster_max_iter must be > 0");
        if (cluster_tol <= 0.0f) throw std::runtime_error("cluster_tol must be > 0");

        const float* xptr = static_cast<const float*>(xb.ptr);
        std::vector<float> Xfull(xptr, xptr + static_cast<size_t>(n) * d);

        center_count_ = n_centers;
        dim_ = d;
        base_count_ = center_count_ + n;
        built_ = false;
        virtuals_finalized_ = false;
        base_entry_ = -1;

        std::mt19937 rng(static_cast<uint32_t>(random_state < 0 ? 42 : random_state));
        std::vector<int> train_ids = make_train_indices(n, train_sample_size, rng);
        std::vector<float> Xtrain = gather_rows(Xfull, train_ids, d);
        const int n_train = static_cast<int>(train_ids.size());

        std::vector<float> centers;
        std::vector<int32_t> labels_train;
        std::string cm = cluster_method;
        std::transform(cm.begin(), cm.end(), cm.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (cm == "kmeans") {
            run_kmeans(Xtrain, n_train, d, n_centers, cluster_max_iter, cluster_tol, rng, centers, labels_train);
        } else if (cm == "cpd_kmeans") {
            run_cpd_kmeans(
                Xtrain,
                n_train,
                d,
                n_centers,
                cluster_max_iter,
                cluster_tol,
                rng,
                cpd_psd_sample,
                cpd_psd_exact_k,
                cpd_mean_sample,
                cpd_mean_exact_k,
                centers,
                labels_train
            );
        } else {
            throw std::runtime_error("cluster_method must be 'kmeans' or 'cpd_kmeans'");
        }

        centers_ = centers;
        build_router_from_centers();

        std::vector<int32_t> labels_full;
        assign_clusters_exact(Xfull, n, d, centers_, center_count_, labels_full);

        insertion_entry_points_.assign(static_cast<size_t>(n), 0);
        for (int i = 0; i < n; ++i) {
            insertion_entry_points_[static_cast<size_t>(i)] = route_center_with_margin(&Xfull[static_cast<size_t>(i) * d]).center_id;
        }

        base_vectors_.clear();
        base_vectors_.reserve(static_cast<size_t>(base_count_) * d);
        base_vectors_.insert(base_vectors_.end(), centers_.begin(), centers_.end());
        base_vectors_.insert(base_vectors_.end(), Xfull.begin(), Xfull.end());

        build_center_anchors_from_labels(Xfull, n, d, labels_full);
        build_center_graph();
        build_base_graph();
        build_center_real_pool(center_real_pool_size_);

        center_graph_ = list_to_csr(center_adj_);
        base_graph_ = list_to_csr(base_adj_);
        built_ = true;

        if (finalize_virtual_nodes_flag) {
            finalize_virtual_nodes();
        }
    }

    void set_router(
        py::array_t<float, py::array::c_style | py::array::forcecast> normals,
        py::array_t<float, py::array::c_style | py::array::forcecast> normal_norms,
        py::array_t<float, py::array::c_style | py::array::forcecast> bias,
        py::array_t<int32_t, py::array::c_style | py::array::forcecast> left,
        py::array_t<int32_t, py::array::c_style | py::array::forcecast> right,
        py::array_t<int32_t, py::array::c_style | py::array::forcecast> leaf_center
    ) {
        auto nb = normals.request();
        auto nnb = normal_norms.request();
        auto bb = bias.request();
        auto lb = left.request();
        auto rb = right.request();
        auto fb = leaf_center.request();

        if (nb.ndim != 2) throw std::runtime_error("normals must be 2D");
        if (nnb.ndim != 1 || bb.ndim != 1 || lb.ndim != 1 || rb.ndim != 1 || fb.ndim != 1) {
            throw std::runtime_error("router metadata arrays must be 1D except normals");
        }

        const int n = static_cast<int>(bb.shape[0]);
        if (static_cast<int>(nb.shape[0]) != n ||
            static_cast<int>(nnb.shape[0]) != n ||
            static_cast<int>(lb.shape[0]) != n ||
            static_cast<int>(rb.shape[0]) != n ||
            static_cast<int>(fb.shape[0]) != n) {
            throw std::runtime_error("router arrays length mismatch");
        }

        if (dim_ > 0 && static_cast<int>(nb.shape[1]) != dim_) {
            throw std::runtime_error("router normal dim mismatch");
        }

        const float* nptr = static_cast<const float*>(nb.ptr);
        const float* nnptr = static_cast<const float*>(nnb.ptr);
        const float* bptr = static_cast<const float*>(bb.ptr);
        const int32_t* lptr = static_cast<const int32_t*>(lb.ptr);
        const int32_t* rptr = static_cast<const int32_t*>(rb.ptr);
        const int32_t* fptr = static_cast<const int32_t*>(fb.ptr);

        router_nodes_ = n;
        router_normals_.assign(nptr, nptr + static_cast<std::size_t>(n) * dim_);
        router_normal_norms_.assign(nnptr, nnptr + n);
        router_bias_.assign(bptr, bptr + n);
        router_left_.assign(lptr, lptr + n);
        router_right_.assign(rptr, rptr + n);
        router_leaf_center_.assign(fptr, fptr + n);
    }

    void set_center_anchors(
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> offsets,
        py::array_t<int32_t, py::array::c_style | py::array::forcecast> indices
    ) {
        auto ob = offsets.request();
        auto ib = indices.request();

        if (ob.ndim != 1 || ib.ndim != 1) {
            throw std::runtime_error("anchor arrays must be 1D");
        }
        if (center_count_ > 0 && static_cast<int>(ob.shape[0]) != center_count_ + 1) {
            throw std::runtime_error("anchor offsets length mismatch");
        }

        const int64_t* optr = static_cast<const int64_t*>(ob.ptr);
        const int32_t* iptr = static_cast<const int32_t*>(ib.ptr);

        anchor_offsets_len_ = static_cast<int>(ob.shape[0]);
        anchor_indices_len_ = static_cast<int>(ib.shape[0]);

        center_anchor_offsets_.assign(optr, optr + ob.shape[0]);
        center_anchor_indices_.assign(iptr, iptr + ib.shape[0]);

    }

    void finalize_virtual_nodes() {
        if (center_count_ <= 0) {
            virtuals_finalized_ = true;
            return;
        }

        // replacement_pool[vid]:
        // 1) 原虚点邻域中的真实点
        // 2) center anchors
        // 3) 该中心最近真实点池
        std::vector<std::vector<int>> replacement_pool(static_cast<size_t>(center_count_));

        for (int vid = 0; vid < center_count_; ++vid) {
            std::unordered_set<int> pool;

            // (1) 原虚点邻域中的真实点
            for (int nb : base_adj_[static_cast<size_t>(vid)]) {
                if (nb >= center_count_) {
                    pool.insert(nb);
                }
            }

            // (2) center anchors
            if (!center_anchor_offsets_.empty()) {
                int64_t start = center_anchor_offsets_[static_cast<size_t>(vid)];
                int64_t end = center_anchor_offsets_[static_cast<size_t>(vid + 1)];
                for (int64_t p = start; p < end; ++p) {
                    int x = center_anchor_indices_[static_cast<size_t>(p)];
                    if (x >= center_count_) {
                        pool.insert(x);
                    }
                }
            }

            // (3) 该中心最近真实点池
            if (vid < static_cast<int>(center_real_pool_.size())) {
                for (int x : center_real_pool_[static_cast<size_t>(vid)]) {
                    if (x >= center_count_) {
                        pool.insert(x);
                    }
                }
            }

            replacement_pool[static_cast<size_t>(vid)] = std::vector<int>(pool.begin(), pool.end());
        }

        // 对每个真实点，移除指向虚点的边，并补真实边
        for (int u = center_count_; u < base_count_; ++u) {
            std::vector<int> kept;
            std::vector<int> removed_virtuals;
            kept.reserve(base_adj_[static_cast<size_t>(u)].size());
            removed_virtuals.reserve(base_adj_[static_cast<size_t>(u)].size());

            std::unordered_set<int> seen;
            for (int nb : base_adj_[static_cast<size_t>(u)]) {
                if (nb < center_count_) {
                    removed_virtuals.push_back(nb);
                    continue;
                }
                if (nb == u) continue;
                if (seen.insert(nb).second) {
                    kept.push_back(nb);
                }
            }

            if (!removed_virtuals.empty()) {
                std::vector<int> candidates;
                candidates.reserve(64);
                std::unordered_set<int> cand_seen(kept.begin(), kept.end());

                for (int vid : removed_virtuals) {
                    for (int x : replacement_pool[static_cast<size_t>(vid)]) {
                        if (x < center_count_ || x == u) continue;
                        if (cand_seen.insert(x).second) {
                            candidates.push_back(x);
                        }
                    }
                }

                if (!candidates.empty()) {
                    // 先按当前点 u 到候选的距离排序
                    auto ordered = exact_sorted_ids(
                        &base_vectors_[static_cast<size_t>(u) * dim_],
                        base_vectors_,
                        dim_,
                        candidates
                    );

                    // 再补到度数上限
                    for (int x : ordered) {
                        if (static_cast<int>(kept.size()) >= base_max_degree_) break;
                        kept.push_back(x);
                    }
                }

                if (!kept.empty()) {
                    // 最后再做一次 heuristic_select
                    kept = heuristic_select(
                        &base_vectors_[static_cast<size_t>(u) * dim_],
                        kept,
                        base_vectors_,
                        dim_,
                        base_max_degree_
                    );
                }
            }

            base_adj_[static_cast<size_t>(u)] = std::move(kept);
        }

        // 虚点彻底退出 base query graph
        for (int vid = 0; vid < center_count_; ++vid) {
            base_adj_[static_cast<size_t>(vid)].clear();
        }

        base_graph_ = list_to_csr(base_adj_);
        virtuals_finalized_ = true;
    }

    py::tuple search_arrays(
        py::array_t<float, py::array::c_style | py::array::forcecast> query,
        int k = 10,
        int ef = 50,
        int n_probe_centers = 1
    ) const {
        if (!built_) throw std::runtime_error("index is not built");

        auto qb = query.request();
        if (qb.ndim != 1) throw std::runtime_error("query must be 1D");
        if (static_cast<int>(qb.shape[0]) != dim_) throw std::runtime_error("query dim mismatch");
        if (k <= 0) throw std::runtime_error("k must be > 0");

        const float* qptr = static_cast<const float*>(qb.ptr);

        SearchWorkspace center_ws(std::max(1, center_count_));
        SearchWorkspace base_ws(std::max(1, base_count_));
        auto result = search_impl(qptr, k, ef, n_probe_centers, center_ws, base_ws);

        py::array_t<int32_t> ids(static_cast<py::ssize_t>(result.first.size()));
        py::array_t<float> dists(static_cast<py::ssize_t>(result.second.size()));

        auto ib = ids.mutable_unchecked<1>();
        auto db = dists.mutable_unchecked<1>();
        for (py::ssize_t i = 0; i < ib.shape(0); ++i) {
            ib(i) = static_cast<int32_t>(result.first[static_cast<size_t>(i)]);
            db(i) = result.second[static_cast<size_t>(i)];
        }

        return py::make_tuple(ids, dists);
    }

    py::tuple search_many_arrays(
        py::array_t<float, py::array::c_style | py::array::forcecast> queries,
        int k = 10,
        int ef = 50,
        int n_probe_centers = 1
    ) const {
        if (!built_) throw std::runtime_error("index is not built");

        auto qb = queries.request();
        if (qb.ndim != 2) throw std::runtime_error("queries must be 2D");
        if (static_cast<int>(qb.shape[1]) != dim_) throw std::runtime_error("queries dim mismatch");
        if (k <= 0) throw std::runtime_error("k must be > 0");

        const int nq = static_cast<int>(qb.shape[0]);
        const float* qptr = static_cast<const float*>(qb.ptr);

        py::array_t<int32_t> ids({nq, k});
        py::array_t<float> dists({nq, k});

        auto ib = ids.request();
        auto db = dists.request();

        int32_t* ids_ptr = static_cast<int32_t*>(ib.ptr);
        float* dists_ptr = static_cast<float*>(db.ptr);

        auto worker = [&](int begin, int end) {
            SearchWorkspace center_ws(std::max(1, center_count_));
            SearchWorkspace base_ws(std::max(1, base_count_));

            for (int i = begin; i < end; ++i) {
                auto one = search_impl(
                    qptr + static_cast<size_t>(i) * dim_,
                    k,
                    ef,
                    n_probe_centers,
                    center_ws,
                    base_ws
                );

                int32_t* row_ids = ids_ptr + static_cast<size_t>(i) * k;
                float* row_dists = dists_ptr + static_cast<size_t>(i) * k;

                for (int j = 0; j < k; ++j) {
                    if (j < static_cast<int>(one.first.size())) {
                        row_ids[j] = static_cast<int32_t>(one.first[static_cast<size_t>(j)]);
                        row_dists[j] = one.second[static_cast<size_t>(j)];
                    } else {
                        row_ids[j] = -1;
                        row_dists[j] = std::numeric_limits<float>::infinity();
                    }
                }
            }
        };

        const int threads = resolve_num_threads(nq);

        {
            py::gil_scoped_release release;

            if (threads <= 1) {
                worker(0, nq);
            } else {
                std::vector<std::thread> pool;
                pool.reserve(static_cast<size_t>(threads));

                const int chunk = (nq + threads - 1) / threads;
                for (int t = 0; t < threads; ++t) {
                    const int begin = t * chunk;
                    const int end = std::min(nq, begin + chunk);
                    if (begin >= end) break;
                    pool.emplace_back(worker, begin, end);
                }

                for (auto& th : pool) {
                    th.join();
                }
            }
        }

        return py::make_tuple(ids, dists);
    }


    py::dict summary() const {
        py::dict d;
        d["m"] = m_;
        d["ef_construction"] = ef_construction_;
        d["center_max_degree"] = center_max_degree_;
        d["base_max_degree"] = base_max_degree_;
        d["ef_extra"] = ef_extra_;
        d["anchor_k"] = anchor_k_;
        d["center_probe_ef_cap"] = center_probe_ef_cap_;
        d["min_probe_centers"] = min_probe_centers_;
        d["max_probe_centers"] = max_probe_centers_;
        d["route_margin_low"] = route_margin_low_;
        d["route_margin_high"] = route_margin_high_;
        d["adaptive_probe"] = adaptive_probe_;
        d["adaptive_ef_extra"] = adaptive_ef_extra_;
        d["num_threads"] = num_threads_;
        d["built"] = built_;
        d["center_count"] = center_count_;
        d["base_count"] = base_count_;
        d["dim"] = dim_;
        d["router_nodes"] = router_nodes_;
        d["anchor_offsets_len"] = anchor_offsets_len_;
        d["anchor_indices_len"] = anchor_indices_len_;
        d["virtuals_finalized"] = virtuals_finalized_;
        d["base_vectors_size"] = static_cast<int>(base_vectors_.size());
        d["centers_size"] = static_cast<int>(centers_.size());
        d["avg_center_degree"] = avg_degree(center_adj_);
        d["avg_base_degree"] = avg_degree(base_adj_);
        d["base_entry"] = base_entry_;
        d["backend"] = "cpp-stage7";
        return d;
    }

private:
    int m_;
    int ef_construction_;
    int center_max_degree_;
    int base_max_degree_;
    int ef_extra_;
    int anchor_k_;
    int center_probe_ef_cap_;
    int min_probe_centers_;
    int max_probe_centers_;
    float route_margin_low_;
    float route_margin_high_;
    bool adaptive_probe_;
    bool adaptive_ef_extra_;
    int num_threads_;

    bool built_ = false;
    int center_count_ = 0;
    int base_count_ = 0;
    int dim_ = 0;
    int router_nodes_ = 0;
    int anchor_offsets_len_ = 0;
    int anchor_indices_len_ = 0;
    bool virtuals_finalized_ = false;
    int base_entry_ = -1;

    std::vector<float> centers_;
    std::vector<float> base_vectors_;
    std::vector<int32_t> insertion_entry_points_;

    std::vector<std::vector<int>> center_adj_;
    std::vector<std::vector<int>> base_adj_;
    CSRGraph center_graph_;
    CSRGraph base_graph_;

    std::vector<float> router_normals_;
    std::vector<float> router_normal_norms_;
    std::vector<float> router_bias_;
    std::vector<int32_t> router_left_;
    std::vector<int32_t> router_right_;
    std::vector<int32_t> router_leaf_center_;

    std::vector<int64_t> center_anchor_offsets_;
    std::vector<int32_t> center_anchor_indices_;

    std::vector<std::vector<int>> center_real_pool_;
    int center_real_pool_size_ = 32;

    static double avg_degree(const std::vector<std::vector<int>>& adj) {
        if (adj.empty()) return 0.0;
        double s = 0.0;
        for (const auto& x : adj) s += static_cast<double>(x.size());
        return s / static_cast<double>(adj.size());
    }

    int resolve_num_threads(int nq) const {
        int threads = num_threads_;
        if (threads <= 0) {
            unsigned int hc = std::thread::hardware_concurrency();
            threads = hc == 0 ? 1 : static_cast<int>(hc);
        }
        threads = std::max(1, std::min(threads, nq));
        return threads;
    }
    void build_center_real_pool(int pool_size) {
        center_real_pool_.assign(static_cast<size_t>(center_count_), {});
        if (center_count_ <= 0 || base_count_ <= center_count_ || pool_size <= 0) {
            return;
        }

        std::vector<std::vector<CandidateDist>> buckets(static_cast<size_t>(center_count_));

        // 优先用 insertion_entry_points_ 作为中心归属；如果越界，再退化到真最近中心
        for (int rid = 0; rid < base_count_ - center_count_; ++rid) {
            const int gid = center_count_ + rid;
            int cid = -1;

            if (rid < static_cast<int>(insertion_entry_points_.size())) {
                cid = insertion_entry_points_[static_cast<size_t>(rid)];
            }

            if (cid < 0 || cid >= center_count_) {
                cid = 0;
                float best_d = l2_sq_ptr(
                    &base_vectors_[static_cast<size_t>(gid) * dim_],
                    &centers_[0],
                    dim_
                );
                for (int j = 1; j < center_count_; ++j) {
                    float d = l2_sq_ptr(
                        &base_vectors_[static_cast<size_t>(gid) * dim_],
                        &centers_[static_cast<size_t>(j) * dim_],
                        dim_
                    );
                    if (d < best_d) {
                        best_d = d;
                        cid = j;
                    }
                }
            }

            float dc = l2_sq_ptr(
                &base_vectors_[static_cast<size_t>(gid) * dim_],
                &centers_[static_cast<size_t>(cid) * dim_],
                dim_
            );
            buckets[static_cast<size_t>(cid)].push_back({gid, dc});
        }

        for (int cid = 0; cid < center_count_; ++cid) {
            auto& bucket = buckets[static_cast<size_t>(cid)];
            if (bucket.empty()) continue;

            std::sort(bucket.begin(), bucket.end(), [](const CandidateDist& a, const CandidateDist& b) {
                return a.dist < b.dist;
            });

            const int keep = std::min(pool_size, static_cast<int>(bucket.size()));
            auto& out = center_real_pool_[static_cast<size_t>(cid)];
            out.reserve(static_cast<size_t>(keep));
            for (int i = 0; i < keep; ++i) {
                out.push_back(bucket[static_cast<size_t>(i)].id);
            }
        }
    }



    std::vector<int> make_train_indices(int n, int train_sample_size, std::mt19937& rng) const {
        std::vector<int> ids(static_cast<size_t>(n));
        std::iota(ids.begin(), ids.end(), 0);
        if (train_sample_size <= 0 || train_sample_size >= n) {
            return ids;
        }
        std::shuffle(ids.begin(), ids.end(), rng);
        ids.resize(static_cast<size_t>(train_sample_size));
        return ids;
    }

    std::vector<float> gather_rows(const std::vector<float>& X, const std::vector<int>& ids, int dim) const {
        std::vector<float> out(static_cast<size_t>(ids.size()) * dim);
        for (size_t i = 0; i < ids.size(); ++i) {
            const float* src = &X[static_cast<size_t>(ids[i]) * dim];
            std::copy(src, src + dim, out.begin() + static_cast<ptrdiff_t>(i * dim));
        }
        return out;
    }

    void assign_clusters_exact(
        const std::vector<float>& X,
        int n,
        int dim,
        const std::vector<float>& centers,
        int k,
        std::vector<int32_t>& labels
    ) const {
        labels.assign(static_cast<size_t>(n), 0);
        for (int i = 0; i < n; ++i) {
            const float* xi = &X[static_cast<size_t>(i) * dim];
            int best = 0;
            float best_d = l2_sq_ptr(xi, &centers[0], dim);
            for (int j = 1; j < k; ++j) {
                float d2 = l2_sq_ptr(xi, &centers[static_cast<size_t>(j) * dim], dim);
                if (d2 < best_d) {
                    best_d = d2;
                    best = j;
                }
            }
            labels[static_cast<size_t>(i)] = static_cast<int32_t>(best);
        }
    }

    std::vector<float> init_kmeans_pp(
        const std::vector<float>& X,
        int n,
        int dim,
        int k,
        std::mt19937& rng
    ) const {
        std::uniform_int_distribution<int> uid(0, n - 1);
        std::uniform_real_distribution<float> u01(0.0f, 1.0f);
        std::vector<float> centers(static_cast<size_t>(k) * dim, 0.0f);
        int first = uid(rng);
        std::copy(&X[static_cast<size_t>(first) * dim], &X[static_cast<size_t>(first + 1) * dim], centers.begin());

        std::vector<float> min_d2(static_cast<size_t>(n), 0.0f);
        for (int i = 0; i < n; ++i) {
            min_d2[static_cast<size_t>(i)] = l2_sq_ptr(&X[static_cast<size_t>(i) * dim], &centers[0], dim);
        }

        for (int c = 1; c < k; ++c) {
            double sum = 0.0;
            for (float v : min_d2) sum += static_cast<double>(v);
            int idx = uid(rng);
            if (sum > 0.0) {
                double r = u01(rng) * sum;
                double acc = 0.0;
                for (int i = 0; i < n; ++i) {
                    acc += static_cast<double>(min_d2[static_cast<size_t>(i)]);
                    if (acc >= r) {
                        idx = i;
                        break;
                    }
                }
            }
            std::copy(
                &X[static_cast<size_t>(idx) * dim],
                &X[static_cast<size_t>(idx + 1) * dim],
                centers.begin() + static_cast<ptrdiff_t>(c * dim)
            );
            for (int i = 0; i < n; ++i) {
                float d2 = l2_sq_ptr(
                    &X[static_cast<size_t>(i) * dim],
                    &centers[static_cast<size_t>(c) * dim],
                    dim
                );
                if (d2 < min_d2[static_cast<size_t>(i)]) {
                    min_d2[static_cast<size_t>(i)] = d2;
                }
            }
        }
        return centers;
    }

    void recompute_centers(
        const std::vector<float>& X,
        int n,
        int dim,
        const std::vector<int32_t>& labels,
        const std::vector<float>& weights,
        int k,
        std::mt19937& rng,
        std::vector<float>& centers
    ) const {
        centers.assign(static_cast<size_t>(k) * dim, 0.0f);
        std::vector<float> sumw(static_cast<size_t>(k), 0.0f);
        for (int i = 0; i < n; ++i) {
            int cid = labels[static_cast<size_t>(i)];
            float w = weights.empty() ? 1.0f : weights[static_cast<size_t>(i)];
            sumw[static_cast<size_t>(cid)] += w;
            const float* xi = &X[static_cast<size_t>(i) * dim];
            float* ci = &centers[static_cast<size_t>(cid) * dim];
            for (int d0 = 0; d0 < dim; ++d0) ci[d0] += xi[d0] * w;
        }
        std::uniform_int_distribution<int> uid(0, n - 1);
        for (int c = 0; c < k; ++c) {
            float* cc = &centers[static_cast<size_t>(c) * dim];
            if (sumw[static_cast<size_t>(c)] <= 1e-12f) {
                int idx = uid(rng);
                std::copy(&X[static_cast<size_t>(idx) * dim], &X[static_cast<size_t>(idx + 1) * dim], cc);
            } else {
                float inv = 1.0f / sumw[static_cast<size_t>(c)];
                for (int d0 = 0; d0 < dim; ++d0) cc[d0] *= inv;
            }
        }
    }

    float center_shift(const std::vector<float>& a, const std::vector<float>& b) const {
        double s = 0.0;
        for (size_t i = 0; i < a.size(); ++i) {
            double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
            s += d * d;
        }
        return static_cast<float>(std::sqrt(s));
    }

    std::vector<float> calculate_potential_difference(
        const std::vector<int32_t>& labels,
        const std::vector<float>& centers,
        int n,
        int dim,
        int k,
        std::mt19937& rng,
        int psd_sample,
        int psd_exact_k
    ) const {
        std::vector<int> cluster_sizes(static_cast<size_t>(k), 0);
        for (int32_t x : labels) cluster_sizes[static_cast<size_t>(x)] += 1;
        std::vector<float> psd(static_cast<size_t>(k) * dim, 0.0f);

        auto accumulate_pair = [&](int i, int j) {
            int sdiff = cluster_sizes[static_cast<size_t>(j)] - cluster_sizes[static_cast<size_t>(i)];
            if (sdiff == 0) return;
            float sign = sdiff < 0 ? -1.0f : 1.0f;
            const float* ci = &centers[static_cast<size_t>(i) * dim];
            const float* cj = &centers[static_cast<size_t>(j) * dim];
            std::vector<float> v(static_cast<size_t>(dim), 0.0f);
            float norm2 = 0.0f;
            for (int d0 = 0; d0 < dim; ++d0) {
                v[static_cast<size_t>(d0)] = (cj[d0] - ci[d0]) * sign;
                norm2 += v[static_cast<size_t>(d0)] * v[static_cast<size_t>(d0)];
            }
            float norm = std::sqrt(norm2);
            if (norm <= 1e-10f) return;
            float w = static_cast<float>(std::abs(sdiff) / static_cast<double>(n));
            float* pi = &psd[static_cast<size_t>(i) * dim];
            for (int d0 = 0; d0 < dim; ++d0) {
                pi[d0] += (v[static_cast<size_t>(d0)] / norm) * w;
            }
        };

        if (k <= psd_exact_k || psd_sample >= k - 1) {
            for (int i = 0; i < k; ++i) for (int j = 0; j < k; ++j) if (i != j) accumulate_pair(i, j);
            return psd;
        }

        std::uniform_int_distribution<int> uid(0, k - 2);
        int m = std::max(1, psd_sample);
        for (int i = 0; i < k; ++i) {
            for (int t = 0; t < m; ++t) {
                int j = uid(rng);
                if (j >= i) ++j;
                accumulate_pair(i, j);
            }
            float scale = static_cast<float>((k - 1) / static_cast<double>(m));
            float* pi = &psd[static_cast<size_t>(i) * dim];
            for (int d0 = 0; d0 < dim; ++d0) pi[d0] *= scale;
        }
        return psd;
    }

    std::vector<float> calculate_weights(
        const std::vector<float>& X,
        int n,
        int dim,
        const std::vector<int32_t>& labels,
        const std::vector<float>& centers,
        const std::vector<float>& psd,
        int k,
        std::mt19937& rng,
        int mean_sample,
        int mean_exact_k
    ) const {
        std::vector<float> weight(static_cast<size_t>(n), 1.0f);
        std::vector<float> mean_center_dist(static_cast<size_t>(k), 0.0f);

        auto center_dist = [&](int i, int j) {
            return std::sqrt(l2_sq_ptr(&centers[static_cast<size_t>(i) * dim], &centers[static_cast<size_t>(j) * dim], dim));
        };

        if (k <= mean_exact_k) {
            for (int i = 0; i < k; ++i) {
                float acc = 0.0f; int cnt = 0;
                for (int j = 0; j < k; ++j) if (i != j) { acc += center_dist(i, j); ++cnt; }
                if (cnt > 0) mean_center_dist[static_cast<size_t>(i)] = acc / cnt;
            }
        } else {
            int m = std::min(std::max(1, mean_sample), std::max(1, k - 1));
            std::uniform_int_distribution<int> uid(0, k - 2);
            for (int i = 0; i < k; ++i) {
                float acc = 0.0f;
                for (int t = 0; t < m; ++t) {
                    int j = uid(rng);
                    if (j >= i) ++j;
                    acc += center_dist(i, j);
                }
                mean_center_dist[static_cast<size_t>(i)] = acc / m;
            }
        }

        for (int c = 0; c < k; ++c) {
            const float* pc = &psd[static_cast<size_t>(c) * dim];
            float psd_norm2 = 0.0f;
            for (int d0 = 0; d0 < dim; ++d0) psd_norm2 += pc[d0] * pc[d0];
            float psd_norm = std::sqrt(psd_norm2);
            if (psd_norm <= 0.0f) continue;
            float gain = 1.0f + 0.5f * std::sqrt(std::max(0.0f, mean_center_dist[static_cast<size_t>(c)])) * psd_norm;
            const float* cc = &centers[static_cast<size_t>(c) * dim];
            for (int i = 0; i < n; ++i) {
                if (labels[static_cast<size_t>(i)] != c) continue;
                const float* xi = &X[static_cast<size_t>(i) * dim];
                float xnorm2 = 0.0f, dot = 0.0f;
                for (int d0 = 0; d0 < dim; ++d0) {
                    float v = xi[d0] - cc[d0];
                    xnorm2 += v * v;
                    dot += v * pc[d0];
                }
                float xnorm = std::sqrt(std::max(xnorm2, 1e-10f));
                float cosv = dot / (xnorm * std::max(psd_norm, 1e-10f));
                if (cosv > 0.0f) weight[static_cast<size_t>(i)] = gain;
            }
        }
        return weight;
    }

    void run_kmeans(
        const std::vector<float>& X,
        int n,
        int dim,
        int k,
        int max_iter,
        float tol,
        std::mt19937& rng,
        std::vector<float>& centers,
        std::vector<int32_t>& labels
    ) const {
        centers = init_kmeans_pp(X, n, dim, k, rng);
        for (int it = 0; it < max_iter; ++it) {
            std::vector<float> old = centers;
            assign_clusters_exact(X, n, dim, centers, k, labels);
            recompute_centers(X, n, dim, labels, std::vector<float>(), k, rng, centers);
            if (center_shift(old, centers) < tol) break;
        }
        assign_clusters_exact(X, n, dim, centers, k, labels);
    }

    void run_cpd_kmeans(
        const std::vector<float>& X,
        int n,
        int dim,
        int k,
        int max_iter,
        float tol,
        std::mt19937& rng,
        int psd_sample,
        int psd_exact_k,
        int mean_sample,
        int mean_exact_k,
        std::vector<float>& centers,
        std::vector<int32_t>& labels
    ) const {
        centers = init_kmeans_pp(X, n, dim, k, rng);
        for (int it = 0; it < max_iter; ++it) {
            std::vector<float> old = centers;
            assign_clusters_exact(X, n, dim, centers, k, labels);
            std::vector<float> psd = calculate_potential_difference(labels, centers, n, dim, k, rng, psd_sample, psd_exact_k);
            std::vector<float> w = calculate_weights(X, n, dim, labels, centers, psd, k, rng, mean_sample, mean_exact_k);
            recompute_centers(X, n, dim, labels, w, k, rng, centers);
            if (center_shift(old, centers) < tol) break;
        }
        assign_clusters_exact(X, n, dim, centers, k, labels);
    }

    void build_router_from_centers() {
        router_nodes_ = 0;
        router_normals_.clear();
        router_normal_norms_.clear();
        router_bias_.clear();
        router_left_.clear();
        router_right_.clear();
        router_leaf_center_.clear();
        if (center_count_ <= 0) return;

        std::function<int(const std::vector<int>&)> build_node = [&](const std::vector<int>& ids) -> int {
            const int idx = static_cast<int>(router_bias_.size());
            router_bias_.push_back(0.0f);
            router_normal_norms_.push_back(0.0f);
            router_left_.push_back(-1);
            router_right_.push_back(-1);
            router_leaf_center_.push_back(-1);
            router_normals_.resize(static_cast<size_t>(idx + 1) * dim_, 0.0f);

            if (ids.size() == 1) {
                router_leaf_center_[static_cast<size_t>(idx)] = ids[0];
                return idx;
            }

            std::vector<float> mean(static_cast<size_t>(dim_), 0.0f);
            for (int id : ids) {
                const float* c = &centers_[static_cast<size_t>(id) * dim_];
                for (int d0 = 0; d0 < dim_; ++d0) mean[static_cast<size_t>(d0)] += c[d0];
            }
            float inv_n = 1.0f / static_cast<float>(ids.size());
            for (int d0 = 0; d0 < dim_; ++d0) mean[static_cast<size_t>(d0)] *= inv_n;

            int axis = 0;
            float best_var = -1.0f;
            for (int d0 = 0; d0 < dim_; ++d0) {
                float var = 0.0f;
                for (int id : ids) {
                    float diff = centers_[static_cast<size_t>(id) * dim_ + static_cast<size_t>(d0)] - mean[static_cast<size_t>(d0)];
                    var += diff * diff;
                }
                if (var > best_var) {
                    best_var = var;
                    axis = d0;
                }
            }

            std::vector<std::pair<float, int>> proj;
            proj.reserve(ids.size());
            for (int id : ids) {
                proj.push_back({centers_[static_cast<size_t>(id) * dim_ + static_cast<size_t>(axis)], id});
            }
            std::sort(proj.begin(), proj.end(), [](const auto& a, const auto& b) { return a.first < b.first; });

            const size_t mid = proj.size() / 2;
            std::vector<int> left_ids, right_ids;
            left_ids.reserve(mid);
            right_ids.reserve(proj.size() - mid);
            for (size_t i = 0; i < proj.size(); ++i) {
                if (i < mid) left_ids.push_back(proj[i].second); else right_ids.push_back(proj[i].second);
            }
            if (left_ids.empty() || right_ids.empty()) {
                left_ids.clear(); right_ids.clear();
                size_t half = std::max<size_t>(1, ids.size() / 2);
                for (size_t i = 0; i < ids.size(); ++i) {
                    if (i < half) left_ids.push_back(ids[i]); else right_ids.push_back(ids[i]);
                }
            }

            std::vector<float> left_mean(static_cast<size_t>(dim_), 0.0f), right_mean(static_cast<size_t>(dim_), 0.0f);
            for (int id : left_ids) {
                const float* c = &centers_[static_cast<size_t>(id) * dim_];
                for (int d0 = 0; d0 < dim_; ++d0) left_mean[static_cast<size_t>(d0)] += c[d0];
            }
            for (int id : right_ids) {
                const float* c = &centers_[static_cast<size_t>(id) * dim_];
                for (int d0 = 0; d0 < dim_; ++d0) right_mean[static_cast<size_t>(d0)] += c[d0];
            }
            for (int d0 = 0; d0 < dim_; ++d0) {
                left_mean[static_cast<size_t>(d0)] /= static_cast<float>(left_ids.size());
                right_mean[static_cast<size_t>(d0)] /= static_cast<float>(right_ids.size());
            }

            std::vector<float> normal(static_cast<size_t>(dim_), 0.0f);
            float norm2 = 0.0f;
            for (int d0 = 0; d0 < dim_; ++d0) {
                normal[static_cast<size_t>(d0)] = right_mean[static_cast<size_t>(d0)] - left_mean[static_cast<size_t>(d0)];
                norm2 += normal[static_cast<size_t>(d0)] * normal[static_cast<size_t>(d0)];
            }
            float norm = std::sqrt(norm2);
            float bias = 0.0f;
            if (norm > 1e-8f) {
                for (int d0 = 0; d0 < dim_; ++d0) {
                    router_normals_[static_cast<size_t>(idx) * dim_ + static_cast<size_t>(d0)] = normal[static_cast<size_t>(d0)];
                }
                std::vector<float> midpoint(static_cast<size_t>(dim_), 0.0f);
                for (int d0 = 0; d0 < dim_; ++d0) midpoint[static_cast<size_t>(d0)] = 0.5f * (left_mean[static_cast<size_t>(d0)] + right_mean[static_cast<size_t>(d0)]);
                for (int d0 = 0; d0 < dim_; ++d0) bias -= normal[static_cast<size_t>(d0)] * midpoint[static_cast<size_t>(d0)];
                router_normal_norms_[static_cast<size_t>(idx)] = norm;
            } else {
                router_normals_[static_cast<size_t>(idx) * dim_ + static_cast<size_t>(axis)] = 1.0f;
                router_normal_norms_[static_cast<size_t>(idx)] = 1.0f;
                bias = -proj[mid].first;
            }
            router_bias_[static_cast<size_t>(idx)] = bias;
            router_left_[static_cast<size_t>(idx)] = build_node(left_ids);
            router_right_[static_cast<size_t>(idx)] = build_node(right_ids);
            return idx;
        };

        std::vector<int> ids(static_cast<size_t>(center_count_));
        std::iota(ids.begin(), ids.end(), 0);
        build_node(ids);
        router_nodes_ = static_cast<int>(router_bias_.size());
    }

    void build_center_anchors_from_labels(
        const std::vector<float>& X,
        int n,
        int dim,
        const std::vector<int32_t>& labels
    ) {
        center_anchor_offsets_.assign(static_cast<size_t>(center_count_ + 1), 0);
        center_anchor_indices_.clear();
        std::vector<std::vector<int>> members(static_cast<size_t>(center_count_));
        for (int i = 0; i < n; ++i) {
            int cid = labels[static_cast<size_t>(i)];
            if (cid >= 0 && cid < center_count_) members[static_cast<size_t>(cid)].push_back(i);
        }
        int64_t offset = 0;
        for (int cid = 0; cid < center_count_; ++cid) {
            const auto& mem = members[static_cast<size_t>(cid)];
            if (!mem.empty()) {
                std::vector<CandidateDist> dists;
                dists.reserve(mem.size());
                for (int idx : mem) {
                    float d2 = l2_sq_ptr(&X[static_cast<size_t>(idx) * dim], &centers_[static_cast<size_t>(cid) * dim], dim);
                    dists.push_back({idx + center_count_, d2});
                }
                int take = std::min(anchor_k_, static_cast<int>(dists.size()));
                std::nth_element(dists.begin(), dists.begin() + take - 1, dists.end(), [](const CandidateDist& a, const CandidateDist& b) { return a.dist < b.dist; });
                dists.resize(static_cast<size_t>(take));
                std::sort(dists.begin(), dists.end(), [](const CandidateDist& a, const CandidateDist& b) { return a.dist < b.dist; });
                for (const auto& x : dists) center_anchor_indices_.push_back(static_cast<int32_t>(x.id));
                offset += take;
            }
            center_anchor_offsets_[static_cast<size_t>(cid + 1)] = offset;
        }
        anchor_offsets_len_ = static_cast<int>(center_anchor_offsets_.size());
        anchor_indices_len_ = static_cast<int>(center_anchor_indices_.size());
    }

    RouteDecision route_center_with_margin(const float* query) const {
        if (router_nodes_ == 0 || center_count_ == 0) {
            int best = 0;
            float best_d = center_count_ > 0 ? l2_sq_ptr(query, &centers_[0], dim_) : 0.0f;
            for (int i = 1; i < center_count_; ++i) {
                float d = l2_sq_ptr(query, &centers_[static_cast<size_t>(i) * dim_], dim_);
                if (d < best_d) {
                    best_d = d;
                    best = i;
                }
            }
            return {best, std::numeric_limits<float>::infinity()};
        }

        int node = 0;
        float min_margin = std::numeric_limits<float>::infinity();

        while (true) {
            int leaf = router_leaf_center_[static_cast<size_t>(node)];
            if (leaf >= 0) return {leaf, min_margin};

            const float* normal = &router_normals_[static_cast<size_t>(node) * dim_];
            float score = router_bias_[static_cast<size_t>(node)];
            for (int j = 0; j < dim_; ++j) score += normal[j] * query[j];

            float denom = router_normal_norms_[static_cast<size_t>(node)];
            float margin = std::fabs(score) / std::max(denom, 1e-12f);
            if (margin < min_margin) min_margin = margin;

            node = (score <= 0.0f)
                       ? router_left_[static_cast<size_t>(node)]
                       : router_right_[static_cast<size_t>(node)];
            if (node < 0) return {0, min_margin};
        }
    }

    int adaptive_probe_count(int requested_probe_count, float margin) const {
        int base = std::max(min_probe_centers_, std::min(requested_probe_count, max_probe_centers_));
        if (!adaptive_probe_) return base;

        if (margin <= route_margin_low_) {
            return std::max(base, std::min(max_probe_centers_, std::max(4, base)));
        }
        if (margin <= route_margin_high_) {
            return std::max(base, std::min(max_probe_centers_, std::max(2, base)));
        }
        return base;
    }

    int adaptive_extra(int probe_count, float margin, int k, int ef) const {
        if (!adaptive_ef_extra_) return std::max(0, ef_extra_);

        int extra = ef_extra_;
        if (margin > route_margin_high_) {
            extra = std::max(16, ef_extra_ / 2);
        } else if (margin > route_margin_low_) {
            extra = std::max(24, static_cast<int>(0.75f * ef_extra_));
        } else {
            extra = ef_extra_ + std::max(16, probe_count * anchor_k_ * 4);
        }

        if (ef < k) extra += (k - ef);
        return std::max(0, extra);
    }

    std::vector<int> get_center_anchors(int center_id) const {
        std::vector<int> out;
        if (center_anchor_offsets_.empty()) return out;
        int64_t start = center_anchor_offsets_[static_cast<size_t>(center_id)];
        int64_t end = center_anchor_offsets_[static_cast<size_t>(center_id + 1)];
        out.reserve(static_cast<size_t>(std::max<int64_t>(0, end - start)));
        for (int64_t p = start; p < end; ++p) {
            out.push_back(center_anchor_indices_[static_cast<size_t>(p)]);
        }
        return out;
    }

    std::pair<std::vector<int>, std::vector<float>> search_impl(
        const float* query,
        int k,
        int ef,
        int requested_probe_count,
        SearchWorkspace& center_ws,
        SearchWorkspace& base_ws
    ) const {
        if (base_count_ == 0) return {{}, {}};

        RouteDecision decision = route_center_with_margin(query);
        int probe_count = adaptive_probe_count(std::max(1, requested_probe_count), decision.margin);

        std::vector<int> probe_ids;
        if (center_count_ > 0) {
            if (probe_count <= 1 || center_graph_.n_nodes() == 0) {
                probe_ids.push_back(decision.center_id);
            } else {
                int center_probe_ef = std::max(probe_count, std::min(center_probe_ef_cap_, center_count_));
                std::vector<int> center_cand = search_layer_csr(
                    query,
                    std::vector<int>{decision.center_id},
                    centers_,
                    center_count_,
                    dim_,
                    center_graph_,
                    center_probe_ef,
                    center_ws
                );
                if (center_cand.empty()) {
                    probe_ids.push_back(decision.center_id);
                } else {
                    if (static_cast<int>(center_cand.size()) > probe_count) {
                        center_cand.resize(probe_count);
                    }
                    probe_ids = exact_sorted_ids(query, centers_, dim_, center_cand);
                    if (static_cast<int>(probe_ids.size()) > probe_count) {
                        probe_ids.resize(probe_count);
                    }
                }
            }
        }

        std::vector<int> entry_points;
        entry_points.reserve(static_cast<size_t>(probe_ids.size() * std::max(1, anchor_k_) + 2));

        for (int cid : probe_ids) {
            auto anchors = get_center_anchors(cid);
            entry_points.insert(entry_points.end(), anchors.begin(), anchors.end());
        }

        if (base_entry_ >= center_count_) {
            entry_points.push_back(base_entry_);
        }

        if (entry_points.empty()) {
            if (center_count_ < base_count_) {
                entry_points.push_back(center_count_);
            } else {
                entry_points.push_back(0);
            }
        }

        entry_points = unique_keep_order(entry_points);

        int ef_internal = std::min(
            std::max(ef, k) + adaptive_extra(probe_count, decision.margin, k, ef),
            base_count_
        );

        std::vector<int> cand = search_layer_csr(
            query,
            entry_points,
            base_vectors_,
            base_count_,
            dim_,
            base_graph_,
            ef_internal,
            base_ws
        );

        std::vector<CandidateDist> real;
        real.reserve(cand.size());
        for (int gid : cand) {
            if (gid < center_count_) continue;
            real.push_back({gid - center_count_, l2_sq_ptr(query, &base_vectors_[static_cast<size_t>(gid) * dim_], dim_)});
        }

        if (real.empty()) {
            for (int gid = center_count_; gid < base_count_; ++gid) {
                real.push_back({gid - center_count_, l2_sq_ptr(query, &base_vectors_[static_cast<size_t>(gid) * dim_], dim_)});
            }
        }

        int topk = std::min(k, static_cast<int>(real.size()));
        std::nth_element(
            real.begin(),
            real.begin() + topk - 1,
            real.end(),
            [](const CandidateDist& a, const CandidateDist& b) { return a.dist < b.dist; }
        );
        real.resize(static_cast<size_t>(topk));
        std::sort(real.begin(), real.end(), [](const CandidateDist& a, const CandidateDist& b) {
            return a.dist < b.dist;
        });

        std::vector<int> ids;
        std::vector<float> dists;
        ids.reserve(static_cast<size_t>(topk));
        dists.reserve(static_cast<size_t>(topk));

        for (const auto& x : real) {
            ids.push_back(x.id);
            dists.push_back(x.dist);
        }

        return {ids, dists};
    }

    void build_center_graph() {
        center_adj_.assign(static_cast<size_t>(center_count_), {});
        if (center_count_ <= 1) return;

        std::vector<std::vector<int>> raw_out(static_cast<size_t>(center_count_));
        for (int cid = 0; cid < center_count_; ++cid) {
            std::vector<int> cand;
            cand.reserve(static_cast<size_t>(center_count_ - 1));
            for (int j = 0; j < center_count_; ++j) if (j != cid) cand.push_back(j);

            cand = exact_sorted_ids(
                &centers_[static_cast<size_t>(cid) * dim_],
                centers_,
                dim_,
                cand
            );
            raw_out[static_cast<size_t>(cid)] = heuristic_select(
                &centers_[static_cast<size_t>(cid) * dim_],
                cand,
                centers_,
                dim_,
                m_
            );
        }

        center_adj_ = raw_out;
        for (int cid = 0; cid < center_count_; ++cid) {
            for (int nb : raw_out[static_cast<size_t>(cid)]) {
                std::vector<int> merged = center_adj_[static_cast<size_t>(nb)];
                if (std::find(merged.begin(), merged.end(), cid) == merged.end()) {
                    merged.push_back(cid);
                }
                if (static_cast<int>(merged.size()) <= center_max_degree_) {
                    center_adj_[static_cast<size_t>(nb)] = std::move(merged);
                } else {
                    std::vector<int> tmp;
                    tmp.reserve(merged.size());
                    for (int x : merged) if (x != nb) tmp.push_back(x);
                    tmp = exact_sorted_ids(
                        &centers_[static_cast<size_t>(nb) * dim_],
                        centers_,
                        dim_,
                        tmp
                    );
                    center_adj_[static_cast<size_t>(nb)] = heuristic_select(
                        &centers_[static_cast<size_t>(nb) * dim_],
                        tmp,
                        centers_,
                        dim_,
                        center_max_degree_
                    );
                }
            }
        }
    }

    void build_base_graph() {
        base_adj_.assign(static_cast<size_t>(base_count_), {});
        for (int i = 0; i < center_count_; ++i) {
            base_adj_[static_cast<size_t>(i)] = center_adj_[static_cast<size_t>(i)];
        }

        int start_idx = 0;
        if (center_count_ > 0) {
            base_entry_ = center_count_ - 1;
            start_idx = center_count_;
        } else {
            base_entry_ = 0;
            start_idx = 1;
        }

        SearchWorkspace build_ws(base_count_);

        for (int idx = start_idx; idx < base_count_; ++idx) {
            const float* query = &base_vectors_[static_cast<size_t>(idx) * dim_];

            std::vector<int> entry_points;
            int real_offset = idx - center_count_;
            if (real_offset >= 0 && real_offset < static_cast<int>(insertion_entry_points_.size())) {
                entry_points.push_back(insertion_entry_points_[static_cast<size_t>(real_offset)]);
            }
            if (base_entry_ >= 0 &&
                std::find(entry_points.begin(), entry_points.end(), base_entry_) == entry_points.end()) {
                entry_points.push_back(base_entry_);
            }
            if (entry_points.empty()) entry_points.push_back(0);

            int effective_ef = std::min(ef_construction_, std::max(1, idx));
            std::vector<int> cand = search_layer_list(
                query,
                entry_points,
                base_vectors_,
                idx,
                dim_,
                base_adj_,
                effective_ef,
                build_ws
            );

            if (cand.empty()) {
                cand.reserve(static_cast<size_t>(idx));
                for (int i = 0; i < idx; ++i) cand.push_back(i);
                cand = exact_sorted_ids(query, base_vectors_, dim_, cand);
            }

            std::vector<int> selected = heuristic_select(query, cand, base_vectors_, dim_, m_);
            if (selected.empty() && !cand.empty()) selected.push_back(cand[0]);

            mutual_connect(idx, selected, base_vectors_, dim_, base_adj_, m_, base_max_degree_);
            base_entry_ = idx;
        }
    }
};

PYBIND11_MODULE(_tlhl_cpp, m) {
    m.doc() = "TLHL C++ core stage 7";

    py::class_<TLHLCore>(m, "TLHLCore")
        .def(
            py::init<
                int, int, int, int, int, int, int,
                int, int, float, float, bool, bool, int
            >(),
            py::arg("m"),
            py::arg("ef_construction"),
            py::arg("center_max_degree"),
            py::arg("base_max_degree"),
            py::arg("ef_extra") = 64,
            py::arg("anchor_k") = 4,
            py::arg("center_probe_ef_cap") = 16,
            py::arg("min_probe_centers") = 1,
            py::arg("max_probe_centers") = 8,
            py::arg("route_margin_low") = 0.05f,
            py::arg("route_margin_high") = 0.20f,
            py::arg("adaptive_probe") = true,
            py::arg("adaptive_ef_extra") = true,
            py::arg("num_threads") = 0
        )
        .def("ping", &TLHLCore::ping)
        .def("version", &TLHLCore::version)
        .def("fit_auto", &TLHLCore::fit_auto,
             py::arg("X"),
             py::arg("cluster_method"),
             py::arg("n_centers"),
             py::arg("cluster_max_iter") = 20,
             py::arg("cluster_tol") = 1e-6f,
             py::arg("random_state") = 42,
             py::arg("train_sample_size") = -1,
             py::arg("cpd_psd_sample") = 64,
             py::arg("cpd_psd_exact_k") = 512,
             py::arg("cpd_mean_sample") = 64,
             py::arg("cpd_mean_exact_k") = 512,
             py::arg("finalize_virtual_nodes") = true)
        .def("build", &TLHLCore::build,
             py::arg("centers"),
             py::arg("base_vectors"),
             py::arg("insertion_entry_points"))
        .def("set_router", &TLHLCore::set_router,
             py::arg("normals"),
             py::arg("normal_norms"),
             py::arg("bias"),
             py::arg("left"),
             py::arg("right"),
             py::arg("leaf_center"))
        .def("set_center_anchors", &TLHLCore::set_center_anchors,
             py::arg("offsets"),
             py::arg("indices"))
        .def("finalize_virtual_nodes", &TLHLCore::finalize_virtual_nodes)
        .def("search_arrays", &TLHLCore::search_arrays,
             py::arg("query"),
             py::arg("k") = 10,
             py::arg("ef") = 50,
             py::arg("n_probe_centers") = 1)
        .def("search_many_arrays", &TLHLCore::search_many_arrays,
             py::arg("queries"),
             py::arg("k") = 10,
             py::arg("ef") = 50,
             py::arg("n_probe_centers") = 1)
        .def("summary", &TLHLCore::summary);
}