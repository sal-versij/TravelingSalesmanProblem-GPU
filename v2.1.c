#include <stdio.h>
#include <stdlib.h>

#define CL_TARGET_OPENCL_VERSION 120

#include "ocl_boiler.h"
#include "setup.h"

cl_event kernel(cl_command_queue q, cl_kernel k, size_t preferred_multiple_init, cl_mem d_adj, cl_mem d_costs, cl_int v,
                cl_ulong work_size) {
    cl_int err;
    int i = 0;
    AddKernelArg(k, i++, sizeof(d_adj), &d_adj);
    AddKernelArg(k, i++, sizeof(d_costs), &d_costs);
    AddKernelArg(k, i++, sizeof(v), &v);
    AddKernelArg(k, i++, sizeof(work_size), &work_size);
    AddKernelArg(k, i++, v * v * sizeof(int), NULL);

    size_t gws = preferred_multiple_init;
    size_t lws = preferred_multiple_init;

    size_t supportSize = lws * ((v - 1) >> 1) * sizeof(char);

    AddKernelArg(k, i++, lws * sizeof(int), NULL);
    AddKernelArg(k, i++, supportSize, NULL);
    AddKernelArg(k, i++, supportSize, NULL);
    AddKernelArg(k, i++, supportSize, NULL);

    cl_event kernel_evt;
    err = clEnqueueNDRangeKernel(q, k, 1, NULL, &gws, &lws, 0, NULL, &kernel_evt);
    ocl_check(err, "launch kernel");
    return kernel_evt;
}

int main(int argc, char *argv[]) {
    struct Task task = {0};
    cl_int err;
    //region Params
    if (argc < 2 || argc > 6) {
        fprintf(stderr, "Usage: %s <nVertexes> [seed] [missCoeficient] [maxWeight]\n", argv[0]);
        return 1;
    }
    int p = 1;

    const int v = atoi(argv[p++]);
    const int seed = argc > p ? atoi(argv[p++]) : 42;
    const int missCoeficient = argc > p ? atoi(argv[p++]) : 2;
    const int maxWeight = argc > p ? atoi(argv[p++]) : 100;

    task.vertexes = v;
    task.chunkCoeficient = 1;
    task.seed = seed;
    task.missCoeficient = missCoeficient;
    task.maxWeight = maxWeight;

    if (v < 2) {
        fprintf(stderr, "Number of vertices must be at least 2\n");
        return 2;
    }
    //endregion

    struct Info info = initialize("Procedural_Sliding_Window");

    //region Initialize Graph
    size_t adjsize = v * v * sizeof(int);

    int *adj = malloc(adjsize);

    init_graph(adj, v, seed, missCoeficient, maxWeight);

    cl_mem d_adj = clCreateBuffer(info.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR,
                                  adjsize, adj, &err);
    ocl_check(err, "create d_adj");
    //endregion

    //region Kernel Execution
    int i;
    int minCost = INFINITY;

    unsigned long totalPermutations = factorial(v - 1);

    task.chunkSize = totalPermutations;
    task.totalPermutations = totalPermutations;
    task.totalChunks = 1;

    cl_event kernel_evt;
    cl_event read_evt;

    int costs_size = info.preferred_multiple_init * sizeof(int);
    int *costs = malloc(costs_size);
    cl_mem d_costs = clCreateBuffer(info.context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, costs_size, costs,
                                    &err);
    ocl_check(err, "create d_costs");

    kernel_evt = kernel(info.queue, info.kernel, info.preferred_multiple_init, d_adj, d_costs, v, totalPermutations);

    clEnqueueMapBuffer(info.queue, d_costs, CL_TRUE, CL_MAP_READ, 0, costs_size, 1, &kernel_evt, &read_evt, &err);
    ocl_check(err, "map costs");

    for (i = 0; i < info.preferred_multiple_init; ++i) {
        if (costs[i] < minCost) {
            minCost = costs[i];
        }
    }

    task.chunkRuns = malloc(sizeof(struct ChunkRun));
    task.chunkRuns->size = totalPermutations;
    task.chunkRuns->write_runtime = 0;
    task.chunkRuns->kernel_runtime = runtime_ms(kernel_evt);
    task.chunkRuns->read_runtime = runtime_ms(read_evt);
    task.chunkRuns->write_bw = 0;
    task.chunkRuns->kernel_bw = v * v * sizeof(int) + costs_size;
    task.chunkRuns->read_bw = costs_size;

    task.cost = minCost;
    task.runtime = total_runtime_ms(kernel_evt, read_evt);

    printf("Total runtime: %.3f ms\n", task.runtime);
//endregion

    FILE *f = fopen("v2.1.csv", "a");
    printResult(f, task
    );
    fclose(f);


    err = clEnqueueUnmapMemObject(info.queue, d_costs, costs, 1, &read_evt, NULL);
    ocl_check(err, "unmap costs");

    freeInfo(info);
    clReleaseMemObject(d_adj);
    clReleaseMemObject(d_costs);

    return 0;
}
