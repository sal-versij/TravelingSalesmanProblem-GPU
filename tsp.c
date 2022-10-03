#include <stdio.h>
#include <stdlib.h>

#define CL_TARGET_OPENCL_VERSION 120

#include "ocl_boiler.h"
#include "setup.h"

cl_event kernel(cl_command_queue q, cl_kernel k, size_t lws, size_t nws, cl_mem d_adj, cl_mem d_costs, cl_int v,
                cl_ulong work_size) {
    cl_int err;
    int i = 0;
    AddKernelArg(k, i++, sizeof(d_adj), &d_adj);
    AddKernelArg(k, i++, sizeof(d_costs), &d_costs);
    AddKernelArg(k, i++, sizeof(v), &v);
    AddKernelArg(k, i++, sizeof(work_size), &work_size);
    AddKernelArg(k, i++, v * v * sizeof(int), NULL);

    size_t supportSize = lws * (v >> 1) * sizeof(char);

    AddKernelArg(k, i++, lws * sizeof(int), NULL);
    AddKernelArg(k, i++, supportSize, NULL);
    AddKernelArg(k, i++, supportSize, NULL);
    AddKernelArg(k, i++, supportSize, NULL);

    size_t gws = nws * lws;

    cl_event kernel_evt;
    err = clEnqueueNDRangeKernel(q, k, 1, NULL, &gws, &lws, 0, NULL, &kernel_evt);
    ocl_check(err, "launch kernel");
    return kernel_evt;
}

cl_event search_min(cl_command_queue q, cl_kernel k, cl_event *waiting_event, cl_uint waiting_list_length, cl_int nwg,
                    size_t lws,
                    cl_mem d_output, cl_mem d_input, cl_int nquarts) {
    cl_int err, i = 0;
    AddKernelArg(k, i++, sizeof(d_output), &d_output);
    AddKernelArg(k, i++, sizeof(d_input), &d_input);
    AddKernelArg(k, i++, lws * sizeof(cl_int), NULL);
    AddKernelArg(k, i++, sizeof(nquarts), &nquarts);

    size_t gws = nwg * lws;

    cl_event reduce_evt;
    err = clEnqueueNDRangeKernel(q, k, 1, NULL, &gws, &lws, waiting_list_length, waiting_event, &reduce_evt);
    ocl_check(err, "launch kernel search_min");
    return reduce_evt;
}

int main(int argc, char *argv[]) {
    struct TaskV3 task = {0};
    cl_int err;
    //region Params
    if (argc < 2 || argc > 6) {
        fprintf(stderr, "Usage: %s <nVertexes> [lws] [nwg] [seed] [missCoeficient] [maxWeight]\n", argv[0]);
        return 1;
    }
    int p = 1;

    const int v = atoi(argv[p++]);
    const int lws = argc > p ? atoi(argv[p++]) : 32;
    const int nwg = argc > p ? atoi(argv[p++]) : 32;
    const int seed = argc > p ? atoi(argv[p++]) : 42;
    const int missCoeficient = argc > p ? atoi(argv[p++]) : 2;
    const int maxWeight = argc > p ? atoi(argv[p++]) : 100;

    task.vertexes = v;
    task.lws = lws;
    task.nwg = nwg;
    task.seed = seed;
    task.missCoeficient = missCoeficient;
    task.maxWeight = maxWeight;

    unsigned long long totalPermutations = factorial(v - 1);
    if (v < 2) {
        fprintf(stderr, "Number of vertices must be at least 2\n");
        return 2;
    }

    if (nwg > 1 && (nwg & 3)) {
        fprintf(stderr, "Number of workgroups must be a multiple of 4\n");
    } else {
        printf("Using %d work-groups (initial) of size %d to process %llu permutations of %d vertexes\n", nwg, lws,
               totalPermutations, v);
    }

    //endregion

    struct Info info = initialize("Procedural_Sliding_Window");

    cl_kernel search_kernel = clCreateKernel(info.program, "search_min", &err);
    ocl_check(err, "create kernel search_min");

    //region Initialize Graph
    size_t adjsize = v * v * sizeof(int);

    int *adj = malloc(adjsize);

    init_graph(adj, v, seed, missCoeficient, maxWeight);

    cl_mem d_adj = clCreateBuffer(info.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR,
                                  adjsize, adj, &err);
    ocl_check(err, "create d_adj");
    //endregion

    //region Kernel Execution
    int minCost = INFINITY;

    printf("Total Permutations: %llu\n", totalPermutations);

    task.chunkSize = totalPermutations;
    task.totalPermutations = totalPermutations;
    task.totalChunks = 1;

    cl_event kernel_evt;
    cl_event read_evt;

    int costs_size = nwg * lws * sizeof(int);
    cl_mem d_costs = clCreateBuffer(info.context, CL_MEM_READ_WRITE, costs_size, NULL, &err);
    ocl_check(err, "create d_costs");

    kernel_evt = kernel(info.queue, info.kernel, lws, nwg, d_adj, d_costs, v, totalPermutations);

    const int npass = nwg == 1 ? 1 : 2;
    printf("Expected passes: %d\n", npass);
    cl_event search_evt[npass];

    cl_mem d_output = clCreateBuffer(info.context, CL_MEM_READ_WRITE, nwg * sizeof(int), NULL, &err);
    ocl_check(err, "create d_output");

    search_evt[0] = search_min(info.queue, search_kernel, &kernel_evt, 1, nwg, lws, d_output, d_costs, nwg * lws / 4);

    if (nwg > 1)
        search_evt[1] = search_min(info.queue, search_kernel, search_evt, 1, 1, lws, d_output, d_output, nwg / 4);

    err = clEnqueueReadBuffer(info.queue, d_output, CL_TRUE, 0, sizeof(minCost), &minCost, 1, &search_evt[npass - 1],
                              &read_evt);
    ocl_check(err, "read minCost");

    task.cost = minCost;
    task.runtime = total_runtime_ms(kernel_evt, read_evt);

    printf("Total runtime: %.3f ms, %.3f MPerm/s\n", task.runtime, totalPermutations / task.runtime / 1e3);
//endregion

    freeInfo(info);
    clReleaseMemObject(d_adj);
    clReleaseMemObject(d_costs);
    clReleaseMemObject(d_output);
    clReleaseKernel(search_kernel);
    free(adj);

    return 0;
}
