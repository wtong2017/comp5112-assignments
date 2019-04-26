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

__global__ void push(int *active_nodes, int *cap, int *flow, int *dist, int64_t *excess, int64_t *stash_excess, int *stash_send, int active_nodes_size, int N) {
    extern __shared__ int s_residual_cap[];

    // Parallel on u
    int block_avg = (active_nodes_size + gridDim.x - 1) / gridDim.x;
    int block_beg = block_avg * blockIdx.x;
    int block_end = min(block_avg * (blockIdx.x + 1), active_nodes_size);

    // Parallel on v
    int thread_avg = (N + blockDim.x - 1) / blockDim.x;
    int thread_beg = thread_avg * threadIdx.x;
    int thread_end = min(thread_avg * (threadIdx.x + 1), N);

    int i = 0;
    for (auto nodes_it = block_beg; nodes_it < block_end; nodes_it++) {
        auto u = active_nodes[nodes_it];
        for (auto v = thread_beg; v < thread_end; v++) {
            s_residual_cap[utils::dev_idx(i, v, N)] = cap[utils::dev_idx(u, v, N)] -
                                flow[utils::dev_idx(u, v, N)];
        }
        i++;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        i = 0;
        for (auto nodes_it = block_beg; nodes_it < block_end; nodes_it++) {
            auto u = active_nodes[nodes_it];
            for (auto v = 0; v < N; v++) {
                auto residual_cap = s_residual_cap[utils::dev_idx(i, v, N)];
                if (residual_cap > 0 && dist[u] > dist[v] && excess[u] > 0) {
                    auto send = (excess[u] - residual_cap > 0 ? residual_cap : excess[u]);
                    atomicAdd(&flow[utils::dev_idx(u, v, N)], send);
                    atomicSub(&flow[utils::dev_idx(v, u, N)], send);
                    excess[u] -= send;
                    atomicAdd((unsigned long long int*)&stash_excess[v], send);
                }
            }
            i++;
        }
    }
}

__global__ void relabel(int *active_nodes, int *cap, int *flow, int *dist, int64_t *excess, int active_nodes_size, int N) {
    extern __shared__ int s_min[];

    // Parallel on u
    int block_avg = (active_nodes_size + gridDim.x - 1) / gridDim.x;
    int block_beg = block_avg * blockIdx.x;
    int block_end = min(block_avg * (blockIdx.x + 1), active_nodes_size);

    // Parallel on v
    int thread_avg = (N + blockDim.x - 1) / blockDim.x;
    int thread_beg = thread_avg * threadIdx.x;
    int thread_end = min(thread_avg * (threadIdx.x + 1), N);

    int i = 0;
    if (threadIdx.x == 0) {
        for (auto nodes_it = block_beg; nodes_it < block_end; nodes_it++) {
            s_min[i] = INT32_MAX;
            i++;
        }
    }
    __syncthreads();

    i = 0;
    for (auto nodes_it = block_beg; nodes_it < block_end; nodes_it++) {
        auto u = active_nodes[nodes_it];
        if (excess[u] > 0) {
            for (auto v = thread_beg; v < thread_end; v++) {
                auto residual_cap = cap[utils::dev_idx(u, v, N)] - flow[utils::dev_idx(u, v, N)];
                if (residual_cap > 0) {
                    atomicMin(&s_min[i], dist[v]);
                }
            }
        }
        i++;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        i = 0;
        for (auto nodes_it = block_beg; nodes_it < block_end; nodes_it++) {
            auto u = active_nodes[nodes_it];
            if (s_min[i] != INT32_MAX)
                dist[u] = s_min[i] + 1;
            i++;
        }
    }
}

__global__ void update(int64_t* excess, int64_t* stash_excess, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;

    // Parallel on v
    int thread_avg = (N + num_threads - 1) / num_threads;
    int thread_beg = thread_avg * i;
    int thread_end = min(thread_avg * (i + 1), N);

    for (auto v = thread_beg; v < thread_end; v++) {
        excess[v] += stash_excess[v];
        stash_excess[v] = 0;
    }
}

int push_relabel(int blocks_per_grid, int threads_per_block, int N, int src, int sink, int *cap, int *flow) {
    /*
     *  Please fill in your codes here.
     */
    int *dist = (int *) calloc(N, sizeof(int));
    auto *excess = (int64_t *) calloc(N, sizeof(int64_t));
    auto *stash_excess = (int64_t *) calloc(N, sizeof(int64_t));

    size_t sizeNNInt = N * N * sizeof(int);
    size_t sizeNInt = N * sizeof(int);
    size_t sizeNInt64 = N * sizeof(int64_t);
    int *d_cap, *d_flow, *d_dist, *d_stash_send;
    int64_t *d_excess, *d_stash_excess;

    // PreFlow
    pre_flow(dist, excess, cap, flow, N, src);

    cudaMalloc(&d_cap, sizeNNInt);
    cudaMalloc(&d_flow, sizeNNInt);
    cudaMalloc(&d_dist, sizeNInt);
    cudaMalloc(&d_stash_send, sizeNNInt);
    cudaMalloc(&d_excess, sizeNInt64);
    cudaMalloc(&d_stash_excess, sizeNInt64);

    cudaMemcpy(d_cap, cap, sizeNNInt, cudaMemcpyHostToDevice);
    cudaMemcpy(d_flow, flow, sizeNNInt, cudaMemcpyHostToDevice);

    cudaMemcpy(d_dist, dist, sizeNInt, cudaMemcpyHostToDevice);
    cudaMemcpy(d_excess, excess, sizeNInt64, cudaMemcpyHostToDevice);
    cudaMemcpy(d_stash_excess, stash_excess, sizeNInt64, cudaMemcpyHostToDevice);

    vector<int> active_nodes;
    int *stash_send = (int *) calloc(N * N, sizeof(int));
    int *d_active_nodes;
    for (auto u = 0; u < N; u++) {
        if (u != src && u != sink) {
            active_nodes.emplace_back(u);
        }
    }

    cudaMemcpy(d_stash_send, stash_send, sizeNNInt, cudaMemcpyHostToDevice);

    int counter = 0;
    // Four-Stage Pulses.
    while (!active_nodes.empty()) {
        // if (counter > 3)
        //     break;
        int active_nodes_size = active_nodes.size();
        int block_avg = (active_nodes_size + blocks_per_grid - 1) / blocks_per_grid;
        cudaMalloc(&d_active_nodes, sizeof(int) * active_nodes_size);
        cudaMemcpy(d_active_nodes, &active_nodes[0], sizeof(int) * active_nodes_size, cudaMemcpyHostToDevice);

        // Stage 1: push.
        push<<<blocks_per_grid, threads_per_block, block_avg * N * sizeof(int)>>>(d_active_nodes, d_cap, d_flow, d_dist, d_excess, d_stash_excess, d_stash_send, active_nodes_size, N);
        cudaMemcpy(flow, d_flow, sizeNNInt, cudaMemcpyDeviceToHost);

        // Stage 2: relabel
        // TODO: it could be faster here
        relabel<<<blocks_per_grid, threads_per_block, block_avg * sizeof(int)>>>(d_active_nodes, d_cap, d_flow, d_dist, d_excess, active_nodes_size, N);

        // Stage 3: apply excess-flow changes for destination vertices.
        update<<<blocks_per_grid, threads_per_block>>>(d_excess, d_stash_excess, N);
        cudaMemcpy(excess, d_excess, sizeNInt64, cudaMemcpyDeviceToHost);

        // Construct active nodes.
        cudaFree(d_active_nodes);
        active_nodes.clear();
        for (auto u = 0; u < N; u++) {
            if (excess[u] > 0 && u != src && u != sink) {
                active_nodes.emplace_back(u);
            }
        }
        // printf("Finish %d\n", counter);
        counter++;
    }
    // printf("Finish %d\n", counter);

    cudaFree(d_cap);
    cudaFree(d_flow);
    cudaFree(d_excess);

    free(dist);
    free(excess);
    free(stash_excess);
    free(stash_send);

    return 0;
}
