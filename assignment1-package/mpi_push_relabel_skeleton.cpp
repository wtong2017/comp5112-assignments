/**
 * Name: Tong Wai
 * Student id: 20271356
 * ITSC email: wtong@connect.ust.hk
*/

#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <cmath>

#include <vector>
#include <iostream>
#include <chrono>

#include "mpi_push_relabel.h"

using namespace std;
using namespace std::chrono;

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

void print_1d(int64_t *array, int N) {
    for (int i = 0; i < N; i++)
        cout << array[i] << ", ";
    cout << endl;
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
        // auto start_clock = high_resolution_clock::now();
        int *local_stash_send_N = NULL;
        int64_t *local_excess = nullptr; 
        int local_N_size = 0;
        // if (my_rank == 0) {
        //     cout << "old" << endl;
        //     print_1d(excess, N);
        //     cout << "--------------" << endl;
        // }
        if (!local_active_nodes.empty()) {
            local_N_size = local_active_nodes.back() - local_active_nodes.front() + 1;
            local_stash_send_N = (int *) calloc(local_N_size * N, sizeof(int));
            local_excess = (int64_t *) malloc(local_N_size *sizeof(int64_t));
            memcpy(local_excess, &excess[local_active_nodes.front()], local_N_size * sizeof(int64_t));
        }
        for (auto u : local_active_nodes) {
            int i = u - local_active_nodes.front();
            for (auto v = 0; v < N; v++) {
                auto residual_cap = local_cap[utils::idx(u, v, N)] -
                                    local_flow[utils::idx(u, v, N)];
                if (residual_cap > 0 && dist[u] > dist[v] && local_excess[i] > 0) {
                    local_stash_send_N[utils::idx(i, v, N)] = std::min<int64_t>(local_excess[i], residual_cap);
                    local_excess[i] -= local_stash_send_N[utils::idx(i, v, N)];
                }
            }
        }
        // for (int i = 0; i < local_N_size; i++)
        //     cout << local_excess[i] << ", ";
        // cout << endl;
        int *sizes = (int *) malloc(p * sizeof(int));
        int *displs = (int *) malloc(p * sizeof(int));
        int local_displs = 0;
        if (!local_active_nodes.empty())
            local_displs = local_active_nodes.front();
        MPI_Allgather(&local_N_size, 1, MPI_INT, sizes, 1, MPI_INT, comm);
        MPI_Allgather(&local_displs, 1, MPI_INT, displs, 1, MPI_INT, comm);
        MPI_Allgatherv(local_excess, local_N_size, MPI_INT64_T, excess, sizes, displs, MPI_INT64_T, comm);
        // if (my_rank == 0) {
        //     print_1d(excess, N);
        // }
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
        // For 2d stash_send
        for (int i = 0; i < p; i++) {
            sizes[i] *= N;
            displs[i] *= N;
        }
        MPI_Allgatherv(local_stash_send_N, local_N_size * N, MPI_INT, stash_send, sizes, displs, MPI_INT, comm);
        free(sizes);
        free(displs);
        free(local_stash_send_N);
        free(local_excess);
        // if (my_rank == 0) {
        //     print_2d(stash_send, N);
        // }

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
        // MPI_Barrier(comm);
        // auto end_clock = high_resolution_clock::now();
        // if (my_rank == 0)
        //     fprintf(stderr, "Elapsed Time: %.9lf s\n", duration_cast<nanoseconds>(end_clock - start_clock).count() / pow(10, 9));
        
        // auto start_clock = high_resolution_clock::now();
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
        // MPI_Barrier(comm);
        // auto end_clock = high_resolution_clock::now();
        // if (my_rank == 0)
        //     fprintf(stderr, "Elapsed Time: %.9lf s\n", duration_cast<nanoseconds>(end_clock - start_clock).count() / pow(10, 9));
        // if (my_rank == 0) {
        //     cout << "flow: ";
        //     for ( int i = 0; i < N*N; i++)
        //         cout << *(local_flow + i) << ", ";
        //     cout << endl;
        // }

        // Stage 2: relabel (update dist to stash_dist).
        // auto start_clock = high_resolution_clock::now();
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
        // MPI_Barrier(comm);
        // auto end_clock = high_resolution_clock::now();
        // if (my_rank == 0)
        //     fprintf(stderr, "Elapsed Time: %.9lf s\n", duration_cast<nanoseconds>(end_clock - start_clock).count() / pow(10, 9));

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
