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

void print_2d(int *mat, int N) {
    cout << "------ Matrix -----" << endl;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            cout << mat[utils::idx(i, j, N)] << ", ";
        }
        cout << endl;
    }
    cout << "-------------------" << endl;
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
    auto *stash_excess = (int64_t *) calloc(N, sizeof(int64_t));

    // PreFlow.
    pre_flow(dist, excess, local_cap, local_flow, N, src);

    vector<int> active_nodes;
    vector<int> local_active_nodes;
    int *stash_send = (int *) calloc(N * N, sizeof(int));

    for (auto u = 0; u < N; u++) {
        if (u != src && u != sink) {
            active_nodes.emplace_back(u);
        }
    }
    int count = 0; // DEBUG

    // Four-Stage Pulses.
    while (!active_nodes.empty()) {
        // if (count == 5)
        //     break;
        // Create local active nodes
        int active_nodes_size = active_nodes.size();
        int p_used = active_nodes_size;
        if (active_nodes_size > p) {
            p_used = p;
        }
        int local_first = active_nodes_size * my_rank / p_used;
        int local_last = active_nodes_size * (my_rank + 1) / p_used;
        int local_size = 0;
        if (local_first < active_nodes_size) {
            local_size = local_last - local_first;
            local_active_nodes.resize(local_size);
            memcpy(&local_active_nodes[0], &active_nodes[local_first], local_size * sizeof(int));
        }
        // cout << my_rank << ": ";
        // for (int e: local_active_nodes) {
        //     cout << e << ", ";
        // }
        // cout << endl;

        // Stage 1: push.
        auto *local_excess_change = (int64_t *) calloc(N, sizeof(int64_t));
        // int *local_stash_send = (int *) calloc(N * N, sizeof(int));
        int *local_stash_send_N = NULL;
        int local_stash_send_N_size = 0;
        // local_stash_send_N = (int *) calloc(local_size * N, sizeof(int));
        if (!local_active_nodes.empty()) {
            local_stash_send_N_size = local_active_nodes.back() - local_active_nodes.front() + 1;
            local_stash_send_N = (int *) calloc(local_stash_send_N_size * N, sizeof(int));
        }
        // int *stash_send_N = (int *) calloc(N * N, sizeof(int));
        for (auto u : local_active_nodes) {
            int i = u - local_active_nodes.front();
            for (auto v = 0; v < N; v++) {
                auto residual_cap = local_cap[utils::idx(u, v, N)] -
                                    local_flow[utils::idx(u, v, N)];
                if (residual_cap > 0 && dist[u] > dist[v] && excess[u] + local_excess_change[u] > 0) {
                    local_stash_send_N[utils::idx(i, v, N)] = std::min<int64_t>(excess[u] + local_excess_change[u], residual_cap);
                    // local_stash_send[utils::idx(u, v, N)] = std::min<int64_t>(excess[u] + local_excess_change[u], residual_cap);
                    // local_excess_change[u] -= local_stash_send[utils::idx(u, v, N)];
                    local_excess_change[u] -= local_stash_send_N[utils::idx(i, v, N)];
                }
            }
        }
        int *sizes = NULL;
        int *displs = NULL;
        int local_displs = 0;
        if (!local_active_nodes.empty())
            local_displs = local_active_nodes.front() * N;
        // cout << my_rank << ": ";
        // for (int e: local_active_nodes) {
        //     cout << e << ", ";
        // }
        // cout << endl;
        // if (my_rank == 0) {
            sizes = (int *) malloc(p * sizeof(int));
            MPI_Allgather(&local_stash_send_N_size, 1, MPI_INT, sizes, 1, MPI_INT, comm);
            for (int i = 0; i < p; i++) {
                sizes[i] *= N;
            }
            // int total_size = sizes[0];
            displs = (int *) malloc(p * sizeof(int));
            MPI_Allgather(&local_displs, 1, MPI_INT, displs, 1, MPI_INT, comm);
            // for (int i=1; i < p; i++) {
            //     displs[i] = displs[i-1] + sizes[i-1];
            //     total_size += sizes[i];
            // }
            // stash_send_N = (int *) calloc(total_size * N, sizeof(int));
            // if (my_rank == 0) {
            //     cout << "Sizes: ";
            //     for (int i = 0; i < p; i++) {
            //         cout << sizes[i] << ", ";
            //     }
            //     cout << endl;
            //     cout << "Displs: ";
            //     for (int i = 0; i < p; i++) {
            //         cout << displs[i] << ", ";
            //     }
            //     cout << endl;
            // }
            MPI_Allgatherv(local_stash_send_N, local_stash_send_N_size * N, MPI_INT, stash_send, sizes, displs, MPI_INT, comm);
            // for (int i = 0; i < p; i++) {
            //     for (int j = 0; j < sizes[i]; j++)
            //         cout << stash_send_N[utils::idx(i, j, sizes[i])] << ", ";
            //     cout << endl;
            // }
            free(sizes);
            // free(stash_send_N);
        // } else {
        //     MPI_Gather(&local_stash_send_N_size, 1, MPI_INT, sizes, 1, MPI_INT, 0, comm);
        //     MPI_Gather(&local_displs, 1, MPI_INT, displs, 1, MPI_INT, 0, comm);
        //     MPI_Allgatherv(local_stash_send_N, local_stash_send_N_size * N, MPI_INT, stash_send, sizes, displs, MPI_INT, comm);
        // }
        if (my_rank == 0) {
            for (int i = 0; i < N; i++)
                local_excess_change[i] += excess[i];
        }
        free(local_stash_send_N);
        // if (my_rank == 0) {
        //     print_2d(stash_send, N);
        // }

        MPI_Allreduce(local_excess_change, excess, N, MPI_INT64_T, MPI_SUM, comm);
        free(local_excess_change);
        // MPI_Allreduce(local_stash_send, stash_send, N * N, MPI_INT, MPI_SUM, comm);
        // free(local_stash_send);
        // if (my_rank == 0) {
        //     cout << "excess:";
        //     for ( int i = 0; i < N; i++)
        //         cout << *(excess + i) << ", ";
        //     cout << endl;
        //     cout << "stash send:";
        //     for ( int i = 0; i < N * N; i++)
        //         cout << *(stash_send + i) << ", ";
        //     cout << endl;
        // }
        for (auto u : active_nodes) {
            for (auto v = 0; v < N; v++) {
                if (stash_send[utils::idx(u, v, N)] > 0) {
                    local_flow[utils::idx(u, v, N)] += stash_send[utils::idx(u, v, N)];
                    local_flow[utils::idx(v, u, N)] -= stash_send[utils::idx(u, v, N)];
                    stash_excess[v] += stash_send[utils::idx(u, v, N)];
                    stash_send[utils::idx(u, v, N)] = 0;
                }
            }
        }
        // if (my_rank == 0) {
        //     cout << "flow: ";
        //     for ( int i = 0; i < N*N; i++)
        //         cout << *(local_flow + i) << ", ";
        //     cout << endl;
        // }

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

        // Stage 3: update dist.
        swap(dist, stash_dist);

        // Stage 4: apply excess-flow changes for destination vertices.
        for (auto v = 0; v < N; v++) {
            if (stash_excess[v] != 0) {
                excess[v] += stash_excess[v];
                stash_excess[v] = 0;
            }
        }

        // Construct active nodes.
        active_nodes.clear();
        for (auto u = 0; u < N; u++) {
            if (excess[u] > 0 && u != src && u != sink) {
                active_nodes.emplace_back(u);
            }
        }
        local_active_nodes.resize(0);
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
