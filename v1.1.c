#include <stdio.h>
#include <stdlib.h>

#define CL_TARGET_OPENCL_VERSION 120

#include "ocl_boiler.h"
#include "setup.h"

cl_event kernel(cl_command_queue q, cl_kernel k, size_t preferred_multiple_init, cl_mem d_permutations, cl_mem d_adj,
                cl_mem d_costs, cl_int v, cl_int work_size) {
    cl_int err;
    AddKernelArg(k, 0, sizeof(d_permutations), &d_permutations);
    AddKernelArg(k, 1, sizeof(d_adj), &d_adj);
    AddKernelArg(k, 2, sizeof(d_costs), &d_costs);
    AddKernelArg(k, 3, sizeof(v), &v);
    AddKernelArg(k, 4, sizeof(work_size), &work_size);

    size_t gws[] = {round_mul_up(work_size, preferred_multiple_init)};

    cl_event kernel_evt;
    err = clEnqueueNDRangeKernel(q, k,
                                 1, NULL, gws, NULL,
                                 0, NULL, &kernel_evt);
    ocl_check(err, "launch kernel");
    return kernel_evt;
}

int main(int argc, char *argv[]) {
    struct Task task = {0};
    cl_int err;
    //region Params
    if (argc < 2 || argc > 6) {
        fprintf(stderr, "Usage: %s <nVertexes> [chunkSize] [seed] [missCoeficient] [maxWeight]\n", argv[0]);
        return 1;
    }
    int p = 1;

    const int v = atoi(argv[p++]);
    const int chunks = argc > p ? atoi(argv[p++]) : 1024;
    const int seed = argc > p ? atoi(argv[p++]) : 42;
    const int missCoeficient = argc > p ? atoi(argv[p++]) : 2;
    const int maxWeight = argc > p ? atoi(argv[p++]) : 100;

    task.vertexes = v;
    task.chunkCoeficient = chunks;
    task.seed = seed;
    task.missCoeficient = missCoeficient;
    task.maxWeight = maxWeight;

    if (v < 2) {
        fprintf(stderr, "Number of vertices must be at least 2\n");
        return 2;
    }
    //endregion

    struct Info info = initialize("Cached_GlobalArray_SingleResult_char");

    //region Initialize Graph
    size_t adjsize = v * v * sizeof(int);

    int *adj = malloc(adjsize);

    init_graph(adj, v, seed, missCoeficient, maxWeight);

    cl_mem d_adj = clCreateBuffer(info.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR,
                                  adjsize, adj,
                                  &err);
    ocl_check(err, "create d_adj");
    //endregion

    //region Kernel Execution
    int i;
    int minCost = INFINITY;

    unsigned long long chunk_size = info.preferred_multiple_init * chunks;
    unsigned long long permutations_size = chunk_size * (v - 1) * sizeof(int);

    char *permutations = malloc(permutations_size);

    char *path = malloc((v - 1) * sizeof(char));
    for (i = 1; i < v; ++i) {
        path[i - 1] = i;
    }

    task.chunkSize = chunk_size;
    task.totalPermutations = factorial(v - 1);
    task.totalChunks = (task.totalPermutations + chunk_size - 1) / chunk_size;
    task.chunkRuns = malloc(task.totalChunks * sizeof(struct ChunkRun));

    cl_event kernel_evt;
    cl_event read_evt;
    cl_event unmap_evt;

    cl_event first_evt;
    cl_event last_evt;

    int costs_size = chunk_size * sizeof(int);
    int *costs = malloc(costs_size);
    cl_mem d_costs = clCreateBuffer(info.context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, costs_size, costs,
                                    &err);
    ocl_check(err, "create d_costs");

    char continueLoop = 1;
    char first = 1;
    int nChunk = 0;
    int percentageUpadte = task.totalChunks / 10 + 1;
    printf("Total chunks: %llu\nUpdate each %d chunks\n", task.totalChunks, percentageUpadte);
    while (continueLoop) {
        nChunk++;
        int current_number_of_permutations = 0;
        do {
            for (i = 0; i < v - 1; ++i) {
                permutations[current_number_of_permutations * (v - 1) + i] = path[i];
            }
            ++current_number_of_permutations;
            if (current_number_of_permutations == chunk_size) {
                continueLoop = next_permutation_chars(path, v - 1);
                break;
            }
            continueLoop = next_permutation_chars(path, v - 1);
        } while (continueLoop);

        cl_mem d_permutations = clCreateBuffer(info.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                               current_number_of_permutations * (v - 1) * sizeof(int), permutations,
                                               &err);
        ocl_check(err, "create d_permutations");

        kernel_evt = kernel(info.queue, info.kernel, info.preferred_multiple_init, d_permutations, d_adj,
                            d_costs, v, current_number_of_permutations);

        clEnqueueMapBuffer(info.queue, d_costs, CL_FALSE, CL_MAP_READ, 0, costs_size, 1, &kernel_evt, &read_evt,
                           &err);
        ocl_check(err, "read costs");
        err = clEnqueueUnmapMemObject(info.queue, d_costs, costs, 1, &read_evt, &unmap_evt);
        ocl_check(err, "unmap costs");

        err = clWaitForEvents(1, &unmap_evt);
        ocl_check(err, "wait for events");

        for (i = 0; i < current_number_of_permutations; ++i) {
            if (costs[i] < minCost) {
                minCost = costs[i];
            }
        }

        struct ChunkRun chunkRun = {0};
        chunkRun.size = current_number_of_permutations;
        chunkRun.write_runtime = 0;
        chunkRun.kernel_runtime = runtime_ms(kernel_evt);
        chunkRun.read_runtime = total_runtime_ms(read_evt, unmap_evt);
        chunkRun.write_bw = 0;
        chunkRun.kernel_bw = 2.0 * v * current_number_of_permutations * sizeof(int);
        chunkRun.read_bw = costs_size;
        if (nChunk % percentageUpadte == 0) {
            printf("Chunk %d/%llu\n", nChunk, task.totalChunks);
        }

        clReleaseMemObject(d_permutations);
        task.chunkRuns[nChunk - 1] = chunkRun;

        if (first) {
            first = 0;
            first_evt = kernel_evt;
        }
    }
    last_evt = unmap_evt;

    task.cost = minCost;
    task.runtime = total_runtime_ms(first_evt, last_evt);

    printf("Chunk %d/%llu\n", nChunk, task.totalChunks);
    printf("Total runtime: %.3f ms\n", task.runtime);
    //endregion

    FILE *f = fopen("v1.1.csv", "a");
    printResult(f, task);
    fclose(f);

    freeInfo(info);
    clReleaseMemObject(d_adj);
    clReleaseMemObject(d_costs);

    return 0;
}
