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

__global__ void push(int *active_nodes, int *cap, int *flow, int *dist, int64_t *excess, int64_t *stash_excess, int *stash_send, int active_nodes_size, int N, int iter) {
    extern __shared__ int s_residual_cap[];
    __shared__ int s_offset;

    int active_nodes_index = blockIdx.x + gridDim.x * iter;
    if (active_nodes_index >= active_nodes_size) {
        return;
    }
    int u = active_nodes[active_nodes_index];

    // Parallel on v
    int thread_avg = (N + blockDim.x - 1) / blockDim.x;
    int thread_beg = thread_avg * threadIdx.x;
    int thread_end = min(thread_avg * (threadIdx.x + 1), N);

    // Init shared memory
    if (threadIdx.x == 0) {
        s_offset = 0;
    }
    __syncthreads();

    //#pragma unroll 8
    for (auto v = thread_beg; v < thread_end; v++) {
        auto residual_cap = cap[utils::dev_idx(u, v, N)] -
                            flow[utils::dev_idx(u, v, N)];
        if (residual_cap > 0 && dist[u] > dist[v]) {
            int old_offset = atomicAdd(&s_offset, 2);
            s_residual_cap[old_offset] = residual_cap;
            s_residual_cap[old_offset+1] = v;
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        #pragma unroll 8
        for (auto i = 0; i < s_offset; i+=2) {
            auto residual_cap = s_residual_cap[i];
            auto v = s_residual_cap[i+1];
            if (excess[u] > 0) {
                auto send = (excess[u] - residual_cap > 0 ? residual_cap : excess[u]);
                stash_send[utils::dev_idx(u, v, N)] = send;
                excess[u] -= send;
            }
        }
    }
    __syncthreads();

    for (auto v = thread_beg; v < thread_end; v++) {
        auto send = stash_send[utils::dev_idx(u, v, N)];
        atomicAdd(&flow[utils::dev_idx(u, v, N)], send);
        atomicSub(&flow[utils::dev_idx(v, u, N)], send);
        atomicAdd((unsigned long long int*)&stash_excess[v], send);
        stash_send[utils::dev_idx(u, v, N)] = 0;
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
        //#pragma unroll 8
        for (auto nodes_it = block_beg; nodes_it < block_end; nodes_it++) {
            s_min[i] = INT32_MAX;
            i++;
        }
    }
    __syncthreads();

    i = 0;
    //#pragma unroll 8
    for (auto nodes_it = block_beg; nodes_it < block_end; nodes_it++) {
        auto u = active_nodes[nodes_it];
        if (excess[u] > 0) {
            int d_min = INT32_MAX;
            //#pragma unroll 8
            for (auto v = thread_beg; v < thread_end; v++) {
                auto residual_cap = cap[utils::dev_idx(u, v, N)] - flow[utils::dev_idx(u, v, N)];
                if (residual_cap > 0) {
                    d_min = min(d_min, dist[v]);
                }
            }
            atomicMin(&s_min[i], d_min);
        }
        i++;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        i = 0;
        //#pragma unroll 8
        for (auto nodes_it = block_beg; nodes_it < block_end; nodes_it++) {
            auto u = active_nodes[nodes_it];
            if (s_min[i] != INT32_MAX)
                dist[u] = s_min[i] + 1;
            i++;
        }
    }
}

__global__ void update(int *active_nodes, int *offset, int64_t* excess, int64_t* stash_excess, int N, int src, int sink) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;

    if (i == 0) {
        *offset = 0;
    }

    // Parallel on v
    int thread_avg = (N + num_threads - 1) / num_threads;
    int thread_beg = thread_avg * i;
    int thread_end = min(thread_avg * (i + 1), N);

    //#pragma unroll 8
    for (auto v = thread_beg; v < thread_end; v++) {
        excess[v] += stash_excess[v];
        stash_excess[v] = 0;
        if (excess[v] > 0 && v != src && v != sink) {
            // Construct active nodes.
            int old_offset = atomicAdd(offset, 1);
            active_nodes[old_offset] = v;
        }
    }
}

int push_relabel(int blocks_per_grid, int threads_per_block, int N, int src, int sink, int *cap, int *flow) {
    /*
     *  Please fill in your codes here.
     */
    int *dist = (int *) calloc(N, sizeof(int));
    auto *excess = (int64_t *) calloc(N, sizeof(int64_t));
    auto *stash_excess = (int64_t *) calloc(N, sizeof(int64_t));
    int *stash_send = (int *) calloc(N * N, sizeof(int));

    size_t sizeNNInt = N * N * sizeof(int);
    size_t sizeNInt = N * sizeof(int);
    size_t sizeNInt64 = N * sizeof(int64_t);
    int *d_cap, *d_flow, *d_dist, *d_stash_send, *d_active_nodes_size;
    int64_t *d_excess, *d_stash_excess;

    // PreFlow
    pre_flow(dist, excess, cap, flow, N, src);

    cudaMalloc(&d_cap, sizeNNInt);
    cudaMalloc(&d_flow, sizeNNInt);
    cudaMalloc(&d_stash_send, sizeNNInt);
    cudaMalloc(&d_dist, sizeNInt);
    cudaMalloc(&d_excess, sizeNInt64);
    cudaMalloc(&d_stash_excess, sizeNInt64);

    cudaMemcpy(d_cap, cap, sizeNNInt, cudaMemcpyHostToDevice);
    cudaMemcpy(d_flow, flow, sizeNNInt, cudaMemcpyHostToDevice);
    cudaMemcpy(d_stash_send, stash_send, sizeNNInt, cudaMemcpyHostToDevice);

    cudaMemcpy(d_dist, dist, sizeNInt, cudaMemcpyHostToDevice);
    cudaMemcpy(d_excess, excess, sizeNInt64, cudaMemcpyHostToDevice);
    cudaMemcpy(d_stash_excess, stash_excess, sizeNInt64, cudaMemcpyHostToDevice);

    vector<int> active_nodes;
    int *d_active_nodes;
    for (auto u = 0; u < N; u++) {
        if (u != src && u != sink) {
            active_nodes.emplace_back(u);
        }
    }
    int active_nodes_size = active_nodes.size();
    cudaMalloc(&d_active_nodes, sizeof(int) * active_nodes_size);
    cudaMalloc(&d_active_nodes_size, sizeof(int));
    cudaMemcpy(d_active_nodes, &active_nodes[0], sizeof(int) * active_nodes_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_active_nodes_size, &active_nodes_size, sizeof(int), cudaMemcpyHostToDevice);
    
    int counter = 0;
    // Four-Stage Pulses.
    while (active_nodes_size > 0) {
        // if (counter > 1)
        //     break;
        int block_avg = (active_nodes_size + blocks_per_grid - 1) / blocks_per_grid;

        // Stage 1: push.
        // Parallel on u
        for (int i = 0; i < block_avg; i++) {
            push<<<blocks_per_grid, threads_per_block, 2 * N * sizeof(int)>>>(d_active_nodes, d_cap, d_flow, d_dist, d_excess, d_stash_excess, d_stash_send, active_nodes_size, N, i);
        }

        // Stage 2: relabel
        relabel<<<blocks_per_grid, threads_per_block, block_avg * sizeof(int)>>>(d_active_nodes, d_cap, d_flow, d_dist, d_excess, active_nodes_size, N);

        // Stage 3: apply excess-flow changes for destination vertices.
        update<<<blocks_per_grid, threads_per_block>>>(d_active_nodes, d_active_nodes_size, d_excess, d_stash_excess, N, src, sink);

        cudaMemcpy(&active_nodes_size, d_active_nodes_size, sizeof(int), cudaMemcpyDeviceToHost);
        // printf("Finish %d\n", counter);
        counter++;
    }
    // printf("Finish %d\n", counter);

    cudaMemcpy(flow, d_flow, sizeNNInt, cudaMemcpyDeviceToHost);

    cudaFree(d_cap);
    cudaFree(d_flow);
    cudaFree(d_excess);
    cudaFree(d_dist);
    cudaFree(d_stash_excess);
    cudaFree(d_stash_send);
    cudaFree(d_active_nodes);
    cudaFree(d_active_nodes_size);

    free(dist);
    free(excess);
    free(stash_excess);

    return 0;
}
