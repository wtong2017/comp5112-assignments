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

#include "cuda_push_relabel.h"

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

__global__ void push(int *active_nodes, int *cap, int *flow, int *dist, int64_t *excess, int64_t *stash_excess, int active_nodes_size, int N) {
    int num_threads = gridDim.x * blockDim.x;
    int my_rank = blockDim.x * blockIdx.x + threadIdx.x;

    int avg = (active_nodes_size + num_threads - 1) / num_threads;
    int nodes_beg = avg * my_rank;
    int nodes_end = min(avg * (my_rank + 1), active_nodes_size);

    // printf("Hello from block %d, thread %d, begin %d, end %d \n", blockIdx.x, threadIdx.x, nodes_beg, nodes_end);
    for (auto nodes_it = nodes_beg; nodes_it < nodes_end; nodes_it++) {
        auto u = active_nodes[nodes_it];
        for (auto v = 0; v < N; v++) {
            auto residual_cap = cap[utils::dev_idx(u, v, N)] -
                                flow[utils::dev_idx(u, v, N)];
            if (residual_cap > 0 && dist[u] > dist[v] && excess[u] > 0) {
                auto send = (excess[u] - residual_cap > 0 ? residual_cap : excess[u]);
                flow[utils::dev_idx(u, v, N)] += send;
                flow[utils::dev_idx(v, u, N)] -= send;
                excess[u] -= send;
                stash_excess[v] += send;
            }
        }
    }

    // for (int i = 0; i < N; i++) {
    //     printf("%d, ", excess[i]);
    // }
    // printf("\n");

    // for (int i = 0; i < N; i++) {
    //     cout << stash_excess[i] << ", ";
    // }
    // cout << endl;

    // for (int i = 0; i < N; i++) {
    //     for (int j = 0; j < N; j++) {
    //         cout << flow[utils::idx(i, j, N)] << ", ";
    //     }
    //     cout << endl;
    // }

}

int push_relabel(int blocks_per_grid, int threads_per_block, int N, int src, int sink, int *cap, int *flow) {
    /*
     *  Please fill in your codes here.
     */
    int *dist = (int *) calloc(N, sizeof(int));
    int *stash_dist = (int *) calloc(N, sizeof(int));
    auto *excess = (int64_t *) calloc(N, sizeof(int64_t));
    auto *stash_excess = (int64_t *) calloc(N, sizeof(int64_t));

    size_t sizeNNInt = N * N * sizeof(int);
    size_t sizeNInt = N * sizeof(int);
    size_t sizeNInt64 = N * sizeof(int64_t);
    int *d_cap, *d_flow, *d_dist;
    int64_t *d_excess, *d_stash_excess;

    // PreFlow
    pre_flow(dist, excess, cap, flow, N, src);

    cudaMalloc(&d_cap, sizeNNInt);
    cudaMalloc(&d_flow, sizeNNInt);
    cudaMalloc(&d_dist, sizeNInt);
    // cudaMalloc(&d_stash_dist, sizeNInt);
    cudaMalloc(&d_excess, sizeNInt64);
    cudaMalloc(&d_stash_excess, sizeNInt64);

    cudaMemcpy(d_cap, cap, sizeNNInt, cudaMemcpyHostToDevice);
    cudaMemcpy(d_flow, flow, sizeNNInt, cudaMemcpyHostToDevice);

    vector<int> active_nodes;
    int *stash_send = (int *) calloc(N * N, sizeof(int));
    int *d_active_nodes;
    for (auto u = 0; u < N; u++) {
        if (u != src && u != sink) {
            active_nodes.emplace_back(u);
        }
    }

    int counter = 0;
    // Four-Stage Pulses.
    while (!active_nodes.empty()) {
        // if (counter > 3)
        //     break;
        int active_nodes_size = active_nodes.size();
        cudaMalloc(&d_active_nodes, sizeof(int) * active_nodes_size);
        cudaMemcpy(d_active_nodes, &active_nodes[0], sizeof(int) * active_nodes_size, cudaMemcpyHostToDevice);

        cudaMemcpy(d_dist, dist, sizeNInt, cudaMemcpyHostToDevice);
        cudaMemcpy(d_excess, excess, sizeNInt64, cudaMemcpyHostToDevice);
        cudaMemcpy(d_stash_excess, stash_excess, sizeNInt64, cudaMemcpyHostToDevice);

        // Stage 1: push.
        // for (auto u : active_nodes) {
        //     for (auto v = 0; v < N; v++) {
        //         auto residual_cap = cap[utils::idx(u, v, N)] -
        //                             flow[utils::idx(u, v, N)];
        //         if (residual_cap > 0 && dist[u] > dist[v] && excess[u] > 0) {
        //             stash_send[utils::idx(u, v, N)] = std::min<int64_t>(excess[u], residual_cap);
        //             excess[u] -= stash_send[utils::idx(u, v, N)];
        //         }
        //     }
        // }
        // for (auto u : active_nodes) {
        //     for (auto v = 0; v < N; v++) {
        //         if (stash_send[utils::idx(u, v, N)] > 0) {
        //             flow[utils::idx(u, v, N)] += stash_send[utils::idx(u, v, N)];
        //             flow[utils::idx(v, u, N)] -= stash_send[utils::idx(u, v, N)];
        //             stash_excess[v] += stash_send[utils::idx(u, v, N)];
        //             stash_send[utils::idx(u, v, N)] = 0;
        //         }
        //     }
        // }

        push<<<blocks_per_grid, threads_per_block>>>(d_active_nodes, d_cap, d_flow, d_dist, d_excess, d_stash_excess, active_nodes_size, N);

        cudaMemcpy(flow, d_flow, sizeNNInt, cudaMemcpyDeviceToHost);
        cudaMemcpy(excess, d_excess, sizeNInt64, cudaMemcpyDeviceToHost);
        cudaMemcpy(stash_excess, d_stash_excess, sizeNInt64, cudaMemcpyDeviceToHost);

        // if (counter == 2) {
        //     for (int i = 0; i < N; i++) {
        //         cout << excess[i] << ", ";
        //     }
        //     cout << endl;
        // }

        // for (int i = 0; i < N; i++) {
        //     cout << stash_excess[i] << ", ";
        // }
        // cout << endl;

        // for (int i = 0; i < N; i++) {
        //     for (int j = 0; j < N; j++) {
        //         cout << flow[utils::idx(i, j, N)] << ", ";
        //     }
        //     cout << endl;
        // }
        // break;

        // Stage 2: relabel (update dist to stash_dist).
        memcpy(stash_dist, dist, N * sizeof(int));
        for (auto u : active_nodes) {
            if (excess[u] > 0) {
                int min_dist = INT32_MAX;
                for (auto v = 0; v < N; v++) {
                    auto residual_cap = cap[utils::idx(u, v, N)] - flow[utils::idx(u, v, N)];
                    if (residual_cap > 0) {
                        min_dist = min(min_dist, dist[v]);
                        stash_dist[u] = min_dist + 1;
                    }
                }
            }
        }

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
        cudaFree(d_active_nodes);
        active_nodes.clear();
        for (auto u = 0; u < N; u++) {
            if (excess[u] > 0 && u != src && u != sink) {
                active_nodes.emplace_back(u);
            }
        }
        counter++;
        // printf("Finish %d\n", counter);
    }

    cudaFree(d_cap);
    cudaFree(d_flow);
    cudaFree(d_excess);

    free(dist);
    free(stash_dist);
    free(excess);
    free(stash_excess);
    free(stash_send);

    return 0;
}
