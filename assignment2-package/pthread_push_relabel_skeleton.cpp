/**
 * Name: TONG Wai
 * Student id: 20271356
 * ITSC email: wtong@connect.ust.hk
*/

#include <cstring>
#include <cstdint>
#include <cstdlib>

#include <vector>
#include <iostream>

#include <pthread.h>

#include "pthread_push_relabel.h"

using namespace std;

/*
 *  You can add helper functions and variables as you wish.
 */
struct thread_arg {
    long rank;
    int num_threads;
    int N;
    int *cap;
    int *flow;
    int *dist;
    int *stash_dist;
    int64_t *excess;
    int64_t *stash_excess;
    int *stash_send;
};

void pre_flow(int *dist, int64_t *excess, int *cap, int *flow, int N, int src) {
    dist[src] = N;
    for (auto v = 0; v < N; v++) {
        flow[utils::idx(src, v, N)] = cap[utils::idx(src, v, N)];
        flow[utils::idx(v, src, N)] = -flow[utils::idx(src, v, N)];
        excess[v] = flow[utils::idx(src, v, N)];
    }
}

// Global variables
vector<int> active_nodes;

// Thread functions
void *push(void *args) {
    struct thread_arg *my_args = (struct thread_arg *) args;
    long my_rank = my_args->rank;
    int num_threads = my_args->num_threads;
    int loc_n = my_args->N;
    int *loc_cap = my_args->cap;
    int *loc_flow = my_args->flow;
    int *dist = my_args->dist;
    int64_t *excess = my_args->excess;
    int *stash_send = my_args->stash_send;
    int avg = (active_nodes.size() + num_threads - 1) / num_threads;
    int nodes_beg = avg * my_rank;
    int nodes_end = min<int>(avg * (my_rank + 1), active_nodes.size());

    // printf("Hello from %ld\n", my_rank);
    // for (auto u : active_nodes)
    //     printf("%i\n", u);
    // for (int i = 0; i < loc_n * loc_n; i++)
    //     printf("%i, ", loc_cap[i]);
    // printf("\n");
    for (auto nodes_it = nodes_beg; nodes_it < nodes_end; nodes_it++) {
        auto u = active_nodes[nodes_it];
        for (auto v = 0; v < loc_n; v++) {
            auto residual_cap = loc_cap[utils::idx(u, v, loc_n)] -
                                loc_flow[utils::idx(u, v, loc_n)];
            if (residual_cap > 0 && dist[u] > dist[v] && excess[u] > 0) {
                stash_send[utils::idx(u, v, loc_n)] = min<int64_t>(excess[u], residual_cap);
                excess[u] -= stash_send[utils::idx(u, v, loc_n)];
            }
        }
    }
    return NULL;
}
void *relabel(void *args) {
    struct thread_arg *my_args = (struct thread_arg *) args;
    long my_rank = my_args->rank;
    int num_threads = my_args->num_threads;
    int loc_n = my_args->N;
    int *loc_cap = my_args->cap;
    int *loc_flow = my_args->flow;
    int *dist = my_args->dist;
    int *stash_dist = my_args->stash_dist;
    int64_t *excess = my_args->excess;
    int *stash_send = my_args->stash_send;
    int avg = (active_nodes.size() + num_threads - 1) / num_threads;
    int nodes_beg = avg * my_rank;
    int nodes_end = min<int>(avg * (my_rank + 1), active_nodes.size());

    for (auto nodes_it = nodes_beg; nodes_it < nodes_end; nodes_it++) {
        auto u = active_nodes[nodes_it];
        if (excess[u] > 0) {
            int min_dist = INT32_MAX;
            for (auto v = 0; v < loc_n; v++) {
                auto residual_cap = loc_cap[utils::idx(u, v, loc_n)] - loc_flow[utils::idx(u, v, loc_n)];
                if (residual_cap > 0) {
                    min_dist = min(min_dist, dist[v]);
                    stash_dist[u] = min_dist + 1;
                }
            }
        }
    }
    return NULL;
}

int push_relabel(int num_threads, int N, int src, int sink, int *cap, int *flow) {
    /*
     *  Please fill in your codes here.
     */
    if (num_threads == 0) {
        printf("Number of threads should not be 0 because it should be equal to number of threads equals 1");
        return 0;
    }
    long thread;
    struct thread_arg *thread_args = (struct thread_arg *) malloc(num_threads * sizeof(struct thread_arg));
    pthread_t *thread_handles = (pthread_t *) malloc(num_threads * sizeof(pthread_t));

    int *dist = (int *) calloc(N, sizeof(int));
    int *stash_dist = (int *) calloc(N, sizeof(int));
    auto *excess = (int64_t *) calloc(N, sizeof(int64_t));
    auto *stash_excess = (int64_t *) calloc(N, sizeof(int64_t));

    // PreFlow.
    pre_flow(dist, excess, cap, flow, N, src);

    int *stash_send = (int *) calloc(N * N, sizeof(int));
    for (auto u = 0; u < N; u++) {
        if (u != src && u != sink) {
            active_nodes.emplace_back(u);
        }
    }

    // Four-Stage Pulses.
    while (!active_nodes.empty()) {
        // Stage 1: push.
        for (thread = 0; thread < num_threads; thread++) {
            thread_args[thread].rank = thread;
            thread_args[thread].num_threads = num_threads;
            thread_args[thread].N = N;
            thread_args[thread].cap = cap;
            thread_args[thread].flow = flow;
            thread_args[thread].dist = dist;
            thread_args[thread].stash_dist = stash_dist;
            thread_args[thread].excess = excess;
            thread_args[thread].stash_excess = stash_excess;
            thread_args[thread].stash_send = stash_send;
            pthread_create(&thread_handles[thread], NULL, push, (void *) &thread_args[thread]);
        }
        for (thread = 0; thread < num_threads; thread++) {
            pthread_join(thread_handles[thread], NULL);
        }
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
        for (auto u : active_nodes) {
            for (auto v = 0; v < N; v++) {
                if (stash_send[utils::idx(u, v, N)] > 0) {
                    flow[utils::idx(u, v, N)] += stash_send[utils::idx(u, v, N)];
                    flow[utils::idx(v, u, N)] -= stash_send[utils::idx(u, v, N)];
                    stash_excess[v] += stash_send[utils::idx(u, v, N)];
                    stash_send[utils::idx(u, v, N)] = 0;
                }
            }
        }

        // Stage 2: relabel (update dist to stash_dist).
        memcpy(stash_dist, dist, N * sizeof(int));
        // for (auto u : active_nodes) {
        //     if (excess[u] > 0) {
        //         int min_dist = INT32_MAX;
        //         for (auto v = 0; v < N; v++) {
        //             auto residual_cap = cap[utils::idx(u, v, N)] - flow[utils::idx(u, v, N)];
        //             if (residual_cap > 0) {
        //                 min_dist = min(min_dist, dist[v]);
        //                 stash_dist[u] = min_dist + 1;
        //             }
        //         }
        //     }
        // }
        for (thread = 0; thread < num_threads; thread++) {
            pthread_create(&thread_handles[thread], NULL, relabel, (void *) &thread_args[thread]);
        }
        for (thread = 0; thread < num_threads; thread++) {
            pthread_join(thread_handles[thread], NULL);
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
        active_nodes.clear();
        for (auto u = 0; u < N; u++) {
            if (excess[u] > 0 && u != src && u != sink) {
                active_nodes.emplace_back(u);
            }
        }
    }

    free(dist);
    free(stash_dist);
    free(excess);
    free(stash_excess);
    free(stash_send);

    return 0;
}
