/**
 * Name: Tong Wai
 * Student id: 20271356
 * ITSC email: wtong@connect.ust.hk
*/

#include <cstring>
#include <cstdint>
#include <cstdlib>

#include <vector>
#include <iostream>

#include "mpi_push_relabel.h"

using namespace std;

/*
 *  You can add helper functions and variables as you wish.
 */
void pre_flow(int *dist, int64_t *excess, int *cap, int *flow, int N, int src) {
    dist[src] = N;
    for (auto v = 0; v < N; v++) {
        flow[utils::idx(src, v, N)] = cap[utils::idx(src, v, N)];
        flow[utils::idx(v, src, N)] = -flow[utils::idx(src, v, N)];
        excess[v] = flow[utils::idx(src, v, N)];
    }
}

int push_relabel(int my_rank, int p, MPI_Comm comm, int N, int src, int sink, int *cap, int *flow) {
    /*
     *  Please fill in your codes here.
     */

    MPI_Bcast(&N, 1, MPI_INT, 0, comm);
    MPI_Bcast(&src, 1, MPI_INT, 0, comm);
    MPI_Bcast(&sink, 1, MPI_INT, 0, comm);
    int *local_cap = (int *) malloc(N * N * sizeof(int));
    int *local_flow = (int *) malloc(N * N * sizeof(int));
    if (my_rank == 0) {
        memcpy(local_cap, cap, N * N * sizeof(int));
        memcpy(local_flow, flow, N * N * sizeof(int));
    }
    MPI_Bcast(local_cap, N * N, MPI_INT, 0, comm);
    MPI_Bcast(local_flow, N * N, MPI_INT, 0, comm);

    int *dist = (int *) calloc(N, sizeof(int));
    int *stash_dist = (int *) calloc(N, sizeof(int));
    auto *excess = (int64_t *) calloc(N, sizeof(int64_t));
    auto *stash_excess = (int64_t *) malloc(N * sizeof(int64_t));

    // PreFlow.
    pre_flow(dist, excess, local_cap, local_flow, N, src);

    vector<int> active_nodes;
    int *stash_send = (int *) calloc(N * N, sizeof(int));

    for (auto u = 0; u < N; u++) {
        if (u != src && u != sink) {
            active_nodes.emplace_back(u);
        }
    }
    int count = 0; // DEBUG

    // Four-Stage Pulses.
    while (!active_nodes.empty()) {
        // if (count == 4) {
        //     break;
        // }
        // if (my_rank == 0) {
        //     cout << "Start loop " << count << endl;
        // }
        // cout << "------------------Loop " << count << " P" << my_rank << ": ";
        // for (int e: active_nodes) {
        //     cout << e << ", ";
        // }
        // cout << endl;
        // Create local active nodes
        int active_nodes_size = active_nodes.size();
        int p_used = active_nodes_size;
        if (active_nodes_size > p) {
            p_used = p;
        }
        int local_first = active_nodes_size * my_rank / p_used;
        int local_last = active_nodes_size * (my_rank + 1) / p_used;
        vector<int> local_active_nodes;
        if (local_first < active_nodes_size) {
            local_active_nodes.assign(active_nodes.begin() + local_first, active_nodes.begin() + local_last);
        }
        // cout << "Loop " << count << " P" << my_rank << " start " << local_first << " end " << local_last << endl;
        // cout << my_rank << ": ";
        // for (int e: local_active_nodes) {
        //     cout << e << ", ";
        // }
        // cout << endl;
        
        // cout << my_rank << " loop " << count << endl;
        // Stage 1: push.
        // memcpy(local_excess, excess, N * sizeof(int64_t));
        // memcpy(local_stash_send, stash_send, N * N * sizeof(int));
        auto *local_excess_change = (int64_t *) calloc(N, sizeof(int64_t));
        int *local_stash_send = (int *) calloc(N * N, sizeof(int));
        for (auto u : local_active_nodes) {
            for (auto v = 0; v < N; v++) {
                // cout << my_rank << ": handle u: " << u << " v: " << v << endl;
                auto residual_cap = local_cap[utils::idx(u, v, N)] -
                                    local_flow[utils::idx(u, v, N)];
                // if (u == 2 && v == 4) {
                //     cout << my_rank << " test: ";
                //     cout << u << " " << v << ": " << residual_cap << endl;
                // }
                if (residual_cap > 0 && dist[u] > dist[v] && excess[u] + local_excess_change[u] > 0) {
                    // cout << my_rank << ": ";
                    // cout << u << " " << v << ": " << residual_cap << endl;
                    // local_stash_send[utils::idx(u, v, N)] = std::min<int64_t>(excess[u], residual_cap);
                    local_stash_send[utils::idx(u, v, N)] = std::min<int64_t>(excess[u] + local_excess_change[u], residual_cap);
                    // excess[u] -= local_stash_send[utils::idx(u, v, N)];
                    local_excess_change[u] -= local_stash_send[utils::idx(u, v, N)];
                }
            }
        }
        if (my_rank == 0) {
            for (int i = 0; i < N; i++)
                local_excess_change[i] += excess[i];
        }
        // for ( int i = 0; i < N * N; i++)
        //     cout << *(local_flow + i) << ", ";
        // cout << endl;
        // cout << my_rank << ": ";
        // for ( int i = 0; i < N; i++)
        //     cout << *(local_excess_change + i) << ", ";
        // cout << endl;
        MPI_Allreduce(local_excess_change, excess, N, MPI_INT64_T, MPI_SUM, comm);
        MPI_Allreduce(local_stash_send, stash_send, N * N, MPI_INT, MPI_SUM, comm);
        free(local_stash_send);
        free(local_excess_change);
        // if (my_rank == 0) {
        //     for ( int i = 0; i < N; i++)
        //         cout << *(excess + i) << ", ";
        //     cout << endl;
        // }
        // if (my_rank == 0) {
        //     for (int i = 0; i < N * N; i++)
        //         cout << *(stash_send + i) << ", ";
        //     cout << endl;
        // }
        // if (count == 10)
        //     break;
        
        int *local_flow_change = (int *) calloc(N * N, sizeof(int));
        auto *local_stash_excess = (int64_t *) calloc(N, sizeof(int64_t));
        if (my_rank == 0) {
            memcpy(local_flow_change, local_flow, N * N * sizeof(int));
        }
        for (auto u : local_active_nodes) {
            for (auto v = 0; v < N; v++) {
                if (stash_send[utils::idx(u, v, N)] > 0) {
                    local_flow_change[utils::idx(u, v, N)] += stash_send[utils::idx(u, v, N)];
                    local_flow_change[utils::idx(v, u, N)] -= stash_send[utils::idx(u, v, N)];
                    local_stash_excess[v] += stash_send[utils::idx(u, v, N)];
                    stash_send[utils::idx(u, v, N)] = 0;
                }
            }
        }
        MPI_Allreduce(local_flow_change, local_flow, N * N, MPI_INT, MPI_SUM, comm);
        free(local_flow_change);
        MPI_Allreduce(local_stash_excess, stash_excess, N, MPI_INT64_T, MPI_SUM, comm);
        free(local_stash_excess);

        // if (my_rank == 0) {
        //     cout << "flow: ";
        //     for (int i = 0; i < N * N; i++)
        //         cout << *(local_flow + i) << ", ";
        //     cout << endl;
        // }
        // if (count == 1)
        //     break;

        // Stage 2: relabel (update dist to stash_dist).
        int *local_stash_dist = (int *) calloc(N, sizeof(int));
        int *local_stash_dict_change = (int *) malloc(N * sizeof(int));
        memcpy(stash_dist, dist, N * sizeof(int));
        for (auto u : local_active_nodes) {
            if (excess[u] > 0) {
                int min_dist = INT32_MAX;
                for (auto v = 0; v < N; v++) {
                    auto residual_cap = local_cap[utils::idx(u, v, N)] - local_flow[utils::idx(u, v, N)];
                    if (residual_cap > 0) {
                        min_dist = min(min_dist, dist[v]);
                        local_stash_dist[u] = min_dist + 1;
                    }
                }
            }
        }
        MPI_Reduce(local_stash_dist, local_stash_dict_change, N, MPI_INT, MPI_SUM, 0, comm);
        if (my_rank == 0) {
            for (int i = 0; i < N; i++) {
                if (local_stash_dict_change[i] > 0) {
                    stash_dist[i] = local_stash_dict_change[i];
                }
            }
        }
        MPI_Bcast(stash_dist, N, MPI_INT, 0, comm);
        free(local_stash_dict_change);
        free(local_stash_dist);
        // cout << my_rank << ": ";
        // for ( int i = 0; i < N; i++)
        //     cout << *(stash_dist + i) << ", ";
        // cout << endl;
        // break;

        // Stage 3: update dist.
        if (my_rank == 0) {
            swap(dist, stash_dist);
        }
        MPI_Bcast(dist, N, MPI_INT, 0, comm);

        // Stage 4: apply excess-flow changes for destination vertices.
        if (my_rank == 0) {
            for (auto v = 0; v < N; v++) {
                if (stash_excess[v] != 0) {
                    excess[v] += stash_excess[v];
                    stash_excess[v] = 0;
                }
            }
        }
        MPI_Bcast(excess, N, MPI_INT64_T, 0, comm);
        MPI_Bcast(stash_excess, N, MPI_INT64_T, 0, comm);

        // cout << my_rank << ": ";
        // for ( int i = 0; i < N; i++)
        //     cout << *(excess + i) << ", ";
        // cout << endl;
        // break;
        // if (my_rank == 0) {
        //     for ( int i = 0; i < N; i++)
        //         cout << *(excess + i) << ", ";
        //     cout << endl;
        // }

        // Construct active nodes.
        // if (my_rank == 0) {
            active_nodes.clear();
            for (auto u = 0; u < N; u++) {
                if (excess[u] > 0 && u != src && u != sink) {
                    active_nodes.emplace_back(u);
                }
            }
        // }
        // MPI_Bcast(&active_nodes[0], active_nodes.size(), MPI_INT, 0, comm);
        // if (my_rank == 0)
        //     cout << "End loop " << count << endl;
        count++;
    }

    if (my_rank == 0) {
        memcpy(cap, local_cap, N * N * sizeof(int));
        memcpy(flow, local_flow, N * N * sizeof(int));
    }
    free(local_cap);
    free(local_flow);

    free(dist);
    free(stash_dist);
    free(excess);
    free(stash_excess);
    free(stash_send);
    
    return 0;
}
