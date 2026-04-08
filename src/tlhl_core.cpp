#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>
#include <cstddef>

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

struct CSRGraph {
    std::vector<int64_t> offsets;
    std::vector<int32_t> indices;

    int n_nodes() const {
        return offsets.empty() ? 0 : static_cast<int>(offsets.size() - 1);
    }
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

std::vector<int> unique_keep_order(const std::vector<int>& in) {
    std::unordered_set<int> seen;
    std::vector<int> out;
    out.reserve(in.size());
    for (int x : in) {
        if (seen.insert(x).second) {
            out.push_back(x);
        }
    }
    return out;
}

struct CandidateDist {
    int id;
    float dist;
};

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
    selected.reserve(std::min<int>(max_neighbors, static_cast<int>(items.size())));

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

    adj[new_id] = selected;

    for (int nb : selected) {
        std::vector<int> merged = adj[nb];
        if (std::find(merged.begin(), merged.end(), new_id) == merged.end()) {
            merged.push_back(new_id);
        }
        if (static_cast<int>(merged.size()) <= old_degree_limit) {
            adj[nb] = std::move(merged);
        } else {
            std::vector<int> tmp;
            tmp.reserve(merged.size());
            for (int x : merged) {
                if (x != nb) tmp.push_back(x);
            }
            tmp = exact_sorted_ids(&vectors[static_cast<size_t>(nb) * dim], vectors, dim, tmp);
            adj[nb] = heuristic_select(&vectors[static_cast<size_t>(nb) * dim], tmp, vectors, dim, old_degree_limit);
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
    std::vector<int>& visited,
    int& visit_token
) {
    if (n_nodes <= 0) return {};
    ef = std::max(1, std::min(ef, n_nodes));

    ++visit_token;
    if (visit_token == std::numeric_limits<int>::max()) {
        std::fill(visited.begin(), visited.end(), 0);
        visit_token = 1;
    }

    std::priority_queue<MinCand, std::vector<MinCand>, std::greater<MinCand>> candidate_heap;
    std::priority_queue<MaxBest> top_heap;

    for (int ep : entry_points) {
        if (ep < 0 || ep >= n_nodes) continue;
        if (visited[ep] == visit_token) continue;
        visited[ep] = visit_token;
        float d = l2_sq_ptr(query, &vectors[static_cast<size_t>(ep) * dim], dim);
        candidate_heap.push({d, ep});
        top_heap.push({d, ep});
    }
    if (candidate_heap.empty()) {
        int ep = 0;
        float d = l2_sq_ptr(query, &vectors[0], dim);
        candidate_heap.push({d, ep});
        top_heap.push({d, ep});
        visited[0] = visit_token;
    }

    while (!candidate_heap.empty()) {
        auto curr = candidate_heap.top();
        candidate_heap.pop();

        float worst_best = top_heap.top().dist;
        if (curr.dist > worst_best && static_cast<int>(top_heap.size()) >= ef) {
            break;
        }

        for (int nb : adj[curr.id]) {
            if (visited[nb] == visit_token) continue;
            visited[nb] = visit_token;
            float d = l2_sq_ptr(query, &vectors[static_cast<size_t>(nb) * dim], dim);
            if (static_cast<int>(top_heap.size()) < ef || d < worst_best) {
                candidate_heap.push({d, nb});
                top_heap.push({d, nb});
                if (static_cast<int>(top_heap.size()) > ef) {
                    top_heap.pop();
                }
                worst_best = top_heap.top().dist;
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
    std::vector<int>& visited,
    int& visit_token
) {
    if (n_nodes <= 0) return {};
    ef = std::max(1, std::min(ef, n_nodes));

    ++visit_token;
    if (visit_token == std::numeric_limits<int>::max()) {
        std::fill(visited.begin(), visited.end(), 0);
        visit_token = 1;
    }

    std::priority_queue<MinCand, std::vector<MinCand>, std::greater<MinCand>> candidate_heap;
    std::priority_queue<MaxBest> top_heap;

    for (int ep : entry_points) {
        if (ep < 0 || ep >= n_nodes) continue;
        if (visited[ep] == visit_token) continue;
        visited[ep] = visit_token;
        float d = l2_sq_ptr(query, &vectors[static_cast<size_t>(ep) * dim], dim);
        candidate_heap.push({d, ep});
        top_heap.push({d, ep});
    }
    if (candidate_heap.empty()) {
        int ep = 0;
        float d = l2_sq_ptr(query, &vectors[0], dim);
        candidate_heap.push({d, ep});
        top_heap.push({d, ep});
        visited[0] = visit_token;
    }

    while (!candidate_heap.empty()) {
        auto curr = candidate_heap.top();
        candidate_heap.pop();

        float worst_best = top_heap.top().dist;
        if (curr.dist > worst_best && static_cast<int>(top_heap.size()) >= ef) {
            break;
        }

        int64_t start = graph.offsets[curr.id];
        int64_t end = graph.offsets[curr.id + 1];
        for (int64_t p = start; p < end; ++p) {
            int nb = graph.indices[static_cast<size_t>(p)];
            if (visited[nb] == visit_token) continue;
            visited[nb] = visit_token;
            float d = l2_sq_ptr(query, &vectors[static_cast<size_t>(nb) * dim], dim);
            if (static_cast<int>(top_heap.size()) < ef || d < worst_best) {
                candidate_heap.push({d, nb});
                top_heap.push({d, nb});
                if (static_cast<int>(top_heap.size()) > ef) {
                    top_heap.pop();
                }
                worst_best = top_heap.top().dist;
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
    int n = static_cast<int>(adj.size());
    graph.offsets.resize(static_cast<size_t>(n) + 1, 0);
    int64_t total = 0;
    for (int i = 0; i < n; ++i) {
        graph.offsets[static_cast<size_t>(i)] = total;
        total += static_cast<int64_t>(adj[i].size());
    }
    graph.offsets[static_cast<size_t>(n)] = total;
    graph.indices.resize(static_cast<size_t>(total));
    int64_t pos = 0;
    for (const auto& nbrs : adj) {
        for (int nb : nbrs) graph.indices[static_cast<size_t>(pos++)] = static_cast<int32_t>(nb);
    }
    return graph;
}

} // namespace

class TLHLCore {
public:
    TLHLCore(int m, int ef_construction, int center_max_degree, int base_max_degree,
             int ef_extra, int anchor_k, int center_probe_ef_cap)
        : m_(m),
          ef_construction_(ef_construction),
          center_max_degree_(center_max_degree),
          base_max_degree_(base_max_degree),
          ef_extra_(ef_extra),
          anchor_k_(anchor_k),
          center_probe_ef_cap_(center_probe_ef_cap) {}

    void build(py::array_t<float, py::array::c_style | py::array::forcecast> centers,
               py::array_t<float, py::array::c_style | py::array::forcecast> base_vectors,
               py::array_t<int32_t, py::array::c_style | py::array::forcecast> insertion_entry_points) {
        auto cb = centers.request();
        auto bb = base_vectors.request();
        auto ib = insertion_entry_points.request();
        if (cb.ndim != 2 || bb.ndim != 2) throw std::runtime_error("centers/base_vectors must be 2D");
        K_ = static_cast<int>(cb.shape[0]);
        N_ = static_cast<int>(bb.shape[0]);
        dim_ = static_cast<int>(bb.shape[1]);
        if (static_cast<int>(cb.shape[1]) != dim_) throw std::runtime_error("dim mismatch");
        if (static_cast<int>(ib.shape[0]) != N_ - K_) throw std::runtime_error("insertion_entry_points length mismatch");

        const float* cptr = static_cast<const float*>(cb.ptr);
        const float* bptr = static_cast<const float*>(bb.ptr);
        centers_.assign(cptr, cptr + static_cast<size_t>(K_) * dim_);
        base_vectors_.assign(bptr, bptr + static_cast<size_t>(N_) * dim_);
        insertion_entry_points_.assign(static_cast<const int32_t*>(ib.ptr), static_cast<const int32_t*>(ib.ptr) + ib.shape[0]);

        build_center_graph();
        build_base_graph();

        center_graph_ = list_to_csr(center_adj_);
        base_graph_ = list_to_csr(base_adj_);
        center_visit_.assign(std::max(1, K_), 0);
        base_visit_.assign(std::max(1, N_), 0);
    }

    void set_router(py::array_t<float, py::array::c_style | py::array::forcecast> normals,
                    py::array_t<float, py::array::c_style | py::array::forcecast> bias,
                    py::array_t<int32_t, py::array::c_style | py::array::forcecast> left,
                    py::array_t<int32_t, py::array::c_style | py::array::forcecast> right,
                    py::array_t<int32_t, py::array::c_style | py::array::forcecast> leaf_center) {
        auto nb = normals.request();
        auto bb = bias.request();
        auto lb = left.request();
        auto rb = right.request();
        auto fb = leaf_center.request();
        if (nb.ndim != 2) throw std::runtime_error("router normals must be 2D");
        router_nodes_ = static_cast<int>(bb.shape[0]);
        if (router_nodes_ == 0) {
            router_normals_.clear(); router_bias_.clear(); router_left_.clear(); router_right_.clear(); router_leaf_center_.clear();
            return;
        }
        if (static_cast<int>(nb.shape[0]) != router_nodes_ || static_cast<int>(nb.shape[1]) != dim_) throw std::runtime_error("router normal shape mismatch");
        router_normals_.assign(static_cast<const float*>(nb.ptr), static_cast<const float*>(nb.ptr) + static_cast<size_t>(router_nodes_) * dim_);
        router_bias_.assign(static_cast<const float*>(bb.ptr), static_cast<const float*>(bb.ptr) + router_nodes_);
        router_left_.assign(static_cast<const int32_t*>(lb.ptr), static_cast<const int32_t*>(lb.ptr) + router_nodes_);
        router_right_.assign(static_cast<const int32_t*>(rb.ptr), static_cast<const int32_t*>(rb.ptr) + router_nodes_);
        router_leaf_center_.assign(static_cast<const int32_t*>(fb.ptr), static_cast<const int32_t*>(fb.ptr) + router_nodes_);
    }

    void set_center_anchors(py::array_t<int64_t, py::array::c_style | py::array::forcecast> offsets,
                            py::array_t<int32_t, py::array::c_style | py::array::forcecast> indices) {
        auto ob = offsets.request();
        auto ib = indices.request();
        if (ob.ndim != 1) throw std::runtime_error("anchor offsets must be 1D");
        if (static_cast<int>(ob.shape[0]) != K_ + 1) throw std::runtime_error("anchor offsets length mismatch");
        center_anchor_offsets_.assign(static_cast<const int64_t*>(ob.ptr), static_cast<const int64_t*>(ob.ptr) + ob.shape[0]);
        center_anchor_indices_.assign(static_cast<const int32_t*>(ib.ptr), static_cast<const int32_t*>(ib.ptr) + ib.shape[0]);
    }

    py::tuple search_arrays(py::array_t<float, py::array::c_style | py::array::forcecast> query,
                            int k, int ef, int n_probe_centers) {
        auto qb = query.request();
        if (qb.ndim != 1 || static_cast<int>(qb.shape[0]) != dim_) throw std::runtime_error("query shape mismatch");
        const float* q = static_cast<const float*>(qb.ptr);
        auto result = search_impl(q, k, ef, n_probe_centers);

        py::array_t<int32_t> ids(result.first.size());
        py::array_t<float> dists(result.second.size());
        auto ib = ids.mutable_unchecked<1>();
        auto db = dists.mutable_unchecked<1>();
        for (py::ssize_t i = 0; i < ib.shape(0); ++i) {
            ib(i) = static_cast<int32_t>(result.first[static_cast<size_t>(i)]);
            db(i) = result.second[static_cast<size_t>(i)];
        }
        return py::make_tuple(ids, dists);
    }

    py::tuple search_many_arrays(py::array_t<float, py::array::c_style | py::array::forcecast> queries,
                                 int k, int ef, int n_probe_centers) {
        auto qb = queries.request();
        if (qb.ndim != 2 || static_cast<int>(qb.shape[1]) != dim_) throw std::runtime_error("queries shape mismatch");
        int nq = static_cast<int>(qb.shape[0]);
        const float* qptr = static_cast<const float*>(qb.ptr);

        py::array_t<int32_t> ids({nq, k});
        py::array_t<float> dists({nq, k});
        auto ib = ids.mutable_unchecked<2>();
        auto db = dists.mutable_unchecked<2>();
        for (int i = 0; i < nq; ++i) {
            auto result = search_impl(qptr + static_cast<size_t>(i) * dim_, k, ef, n_probe_centers);
            int out_n = static_cast<int>(result.first.size());
            for (int j = 0; j < k; ++j) {
                if (j < out_n) {
                    ib(i, j) = static_cast<int32_t>(result.first[static_cast<size_t>(j)]);
                    db(i, j) = result.second[static_cast<size_t>(j)];
                } else {
                    ib(i, j) = -1;
                    db(i, j) = std::numeric_limits<float>::infinity();
                }
            }
        }
        return py::make_tuple(ids, dists);
    }

    py::dict summary() const {
        py::dict d;
        d["center_nodes"] = K_;
        d["base_nodes"] = N_;
        d["dim"] = dim_;
        d["base_entry"] = base_entry_;
        d["avg_center_degree"] = avg_degree(center_adj_);
        d["avg_base_degree"] = avg_degree(base_adj_);
        d["router_nodes"] = router_nodes_;
        d["has_center_anchors"] = !center_anchor_offsets_.empty();
        d["anchor_k"] = anchor_k_;
        d["ef_extra"] = ef_extra_;
        d["query_graph_storage"] = "csr";
        d["builder_backend"] = "c++";
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

    int K_ = 0;
    int N_ = 0;
    int dim_ = 0;
    int base_entry_ = -1;

    std::vector<float> centers_;
    std::vector<float> base_vectors_;
    std::vector<int> insertion_entry_points_;

    std::vector<std::vector<int>> center_adj_;
    std::vector<std::vector<int>> base_adj_;
    CSRGraph center_graph_;
    CSRGraph base_graph_;

    std::vector<int> center_visit_;
    std::vector<int> base_visit_;
    int center_visit_token_ = 1;
    int base_visit_token_ = 1;

    int router_nodes_ = 0;
    std::vector<float> router_normals_;
    std::vector<float> router_bias_;
    std::vector<int32_t> router_left_;
    std::vector<int32_t> router_right_;
    std::vector<int32_t> router_leaf_center_;

    std::vector<int64_t> center_anchor_offsets_;
    std::vector<int32_t> center_anchor_indices_;

    static double avg_degree(const std::vector<std::vector<int>>& adj) {
        if (adj.empty()) return 0.0;
        double s = 0.0;
        for (const auto& x : adj) s += static_cast<double>(x.size());
        return s / static_cast<double>(adj.size());
    }

    int route_center(const float* query) const {
        if (router_nodes_ == 0) {
            int best = 0;
            float best_d = l2_sq_ptr(query, &centers_[0], dim_);
            for (int i = 1; i < K_; ++i) {
                float d = l2_sq_ptr(query, &centers_[static_cast<size_t>(i) * dim_], dim_);
                if (d < best_d) {
                    best_d = d;
                    best = i;
                }
            }
            return best;
        }
        int node = 0;
        while (true) {
            int leaf = router_leaf_center_[static_cast<size_t>(node)];
            if (leaf >= 0) return leaf;
            const float* normal = &router_normals_[static_cast<size_t>(node) * dim_];
            float score = router_bias_[static_cast<size_t>(node)];
            for (int j = 0; j < dim_; ++j) score += normal[j] * query[j];
            node = (score <= 0.0f) ? router_left_[static_cast<size_t>(node)] : router_right_[static_cast<size_t>(node)];
            if (node < 0) return 0;
        }
    }

    std::vector<int> get_center_anchors(int center_id) const {
        std::vector<int> out;
        if (center_anchor_offsets_.empty()) return out;
        int64_t start = center_anchor_offsets_[static_cast<size_t>(center_id)];
        int64_t end = center_anchor_offsets_[static_cast<size_t>(center_id + 1)];
        out.reserve(static_cast<size_t>(end - start));
        for (int64_t p = start; p < end; ++p) out.push_back(center_anchor_indices_[static_cast<size_t>(p)]);
        return out;
    }

    int adaptive_extra(int probe_count) const {
        int base = std::max(16, anchor_k_ * std::max(1, probe_count) * 8);
        return std::min(ef_extra_, base);
    }

    std::pair<std::vector<int>, std::vector<float>> search_impl(const float* query, int k, int ef, int n_probe_centers) {
        if (N_ == 0) return {{}, {}};
        if (ef <= 0) ef = std::max(ef_construction_, k * 4);
        int probe_count = std::max(1, std::min(n_probe_centers, K_));

        int routed_center = route_center(query);

        std::vector<int> probe_ids;
        if (probe_count == 1 || center_graph_.n_nodes() == 0) {
            probe_ids.push_back(routed_center);
        } else {
            int center_probe_ef = std::max(probe_count, std::min(center_probe_ef_cap_, K_));
            std::vector<int> center_cand = search_layer_csr(
                query,
                std::vector<int>{routed_center},
                centers_,
                K_,
                dim_,
                center_graph_,
                center_probe_ef,
                center_visit_,
                center_visit_token_
            );
            if (center_cand.empty()) {
                probe_ids.push_back(routed_center);
            } else {
                int keep = std::max(probe_count, 4);
                if (static_cast<int>(center_cand.size()) > keep) center_cand.resize(keep);
                auto ordered = exact_sorted_ids(query, centers_, dim_, center_cand);
                if (static_cast<int>(ordered.size()) > probe_count) ordered.resize(probe_count);
                probe_ids = std::move(ordered);
            }
        }

        std::vector<int> entry_points;
        entry_points.reserve(static_cast<size_t>(probe_ids.size() * (anchor_k_ + 1) + 1));
        for (int cid : probe_ids) {
            entry_points.push_back(cid);
            auto anchors = get_center_anchors(cid);
            entry_points.insert(entry_points.end(), anchors.begin(), anchors.end());
        }
        if (base_entry_ >= 0) entry_points.push_back(base_entry_);
        entry_points = unique_keep_order(entry_points);

        int ef_internal = std::min(std::max(ef, k) + adaptive_extra(probe_count), N_);
        std::vector<int> cand = search_layer_csr(
            query,
            entry_points,
            base_vectors_,
            N_,
            dim_,
            base_graph_,
            ef_internal,
            base_visit_,
            base_visit_token_
        );

        if (cand.empty()) {
            cand.reserve(std::max(0, N_ - K_));
            for (int i = K_; i < N_; ++i) cand.push_back(i);
        }

        std::vector<CandidateDist> real;
        real.reserve(cand.size());
        for (int idx : cand) {
            if (idx < K_) continue;
            real.push_back({idx, l2_sq_ptr(query, &base_vectors_[static_cast<size_t>(idx) * dim_], dim_)});
        }
        if (real.empty()) return {{}, {}};

        int topk = std::min(k, static_cast<int>(real.size()));
        std::nth_element(real.begin(), real.begin() + topk, real.end(), [](const CandidateDist& a, const CandidateDist& b) {
            return a.dist < b.dist;
        });
        real.resize(topk);
        std::sort(real.begin(), real.end(), [](const CandidateDist& a, const CandidateDist& b) {
            return a.dist < b.dist;
        });

        std::vector<int> ids;
        std::vector<float> dists;
        ids.reserve(real.size());
        dists.reserve(real.size());
        for (const auto& x : real) {
            ids.push_back(x.id - K_);
            dists.push_back(x.dist);
        }
        return {ids, dists};
    }

    void build_center_graph() {
        center_adj_.assign(static_cast<size_t>(K_), {});
        if (K_ <= 1) return;
        std::vector<std::vector<int>> raw_out(static_cast<size_t>(K_));
        for (int cid = 0; cid < K_; ++cid) {
            std::vector<int> cand;
            cand.reserve(static_cast<size_t>(K_ - 1));
            for (int j = 0; j < K_; ++j) if (j != cid) cand.push_back(j);
            cand = exact_sorted_ids(&centers_[static_cast<size_t>(cid) * dim_], centers_, dim_, cand);
            raw_out[static_cast<size_t>(cid)] = heuristic_select(
                &centers_[static_cast<size_t>(cid) * dim_], cand, centers_, dim_, m_);
        }

        center_adj_ = raw_out;
        for (int cid = 0; cid < K_; ++cid) {
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
                    tmp = exact_sorted_ids(&centers_[static_cast<size_t>(nb) * dim_], centers_, dim_, tmp);
                    center_adj_[static_cast<size_t>(nb)] = heuristic_select(
                        &centers_[static_cast<size_t>(nb) * dim_], tmp, centers_, dim_, center_max_degree_);
                }
            }
        }

        for (int cid = 0; cid < K_; ++cid) {
            std::vector<int> cleaned;
            std::unordered_set<int> seen;
            for (int nb : center_adj_[static_cast<size_t>(cid)]) {
                if (nb == cid) continue;
                if (seen.insert(nb).second) cleaned.push_back(nb);
            }
            if (static_cast<int>(cleaned.size()) > center_max_degree_) {
                cleaned = exact_sorted_ids(&centers_[static_cast<size_t>(cid) * dim_], centers_, dim_, cleaned);
                cleaned = heuristic_select(&centers_[static_cast<size_t>(cid) * dim_], cleaned, centers_, dim_, center_max_degree_);
            }
            center_adj_[static_cast<size_t>(cid)] = std::move(cleaned);
        }
    }

    void build_base_graph() {
        base_adj_.assign(static_cast<size_t>(N_), {});
        for (int i = 0; i < K_; ++i) base_adj_[static_cast<size_t>(i)] = center_adj_[static_cast<size_t>(i)];
        int start_idx = 0;
        if (K_ > 0) {
            base_entry_ = K_ - 1;
            start_idx = K_;
        } else {
            base_entry_ = 0;
            start_idx = 1;
        }
        std::vector<int> build_visit(std::max(1, N_), 0);
        int build_visit_token = 1;
        for (int idx = start_idx; idx < N_; ++idx) {
            const float* query = &base_vectors_[static_cast<size_t>(idx) * dim_];
            std::vector<int> entry_points;
            int real_offset = idx - K_;
            if (real_offset >= 0 && real_offset < static_cast<int>(insertion_entry_points_.size())) {
                entry_points.push_back(insertion_entry_points_[static_cast<size_t>(real_offset)]);
            }
            if (base_entry_ >= 0 && std::find(entry_points.begin(), entry_points.end(), base_entry_) == entry_points.end()) {
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
                build_visit,
                build_visit_token
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
    py::class_<TLHLCore>(m, "TLHLCore")
        .def(py::init<int, int, int, int, int, int, int>(),
             py::arg("m"), py::arg("ef_construction"), py::arg("center_max_degree"),
             py::arg("base_max_degree"), py::arg("ef_extra") = 64,
             py::arg("anchor_k") = 4, py::arg("center_probe_ef_cap") = 16)
        .def("build", &TLHLCore::build, py::arg("centers"), py::arg("base_vectors"), py::arg("insertion_entry_points"))
        .def("set_router", &TLHLCore::set_router,
             py::arg("normals"), py::arg("bias"), py::arg("left"), py::arg("right"), py::arg("leaf_center"))
        .def("set_center_anchors", &TLHLCore::set_center_anchors, py::arg("offsets"), py::arg("indices"))
        .def("search_arrays", &TLHLCore::search_arrays,
             py::arg("query"), py::arg("k") = 10, py::arg("ef") = 50, py::arg("n_probe_centers") = 1)
        .def("search_many_arrays", &TLHLCore::search_many_arrays,
             py::arg("queries"), py::arg("k") = 10, py::arg("ef") = 50, py::arg("n_probe_centers") = 1)
        .def("summary", &TLHLCore::summary);
}
