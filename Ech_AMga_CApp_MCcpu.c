#include <stdio.h>
#include <stdlib.h>

#define CL_TARGET_OPENCL_VERSION 120

#include "ocl_boiler.h"
#include "setup.h"

cl_event
kernel(cl_command_queue q, cl_kernel k, size_t preferred_multiple_init, cl_mem d_adj, cl_mem d_permutations,
       cl_mem d_costs, cl_int v, cl_int work_size) {
    cl_int err;
    AddKernelArg(k, 0, sizeof(d_permutations), &d_permutations);
    AddKernelArg(k, 1, sizeof(d_adj), &d_adj);
    AddKernelArg(k, 2, sizeof(d_costs), &d_costs);
    AddKernelArg(k, 3, sizeof(v), &v);
    AddKernelArg(k, 4, sizeof(work_size), &work_size);

    size_t gws[] = {round_mul_up(work_size, preferred_multiple_init)};

    cl_event init_evt;
    err = clEnqueueNDRangeKernel(q, k,
                                 1, NULL, gws, NULL,
                                 0, NULL, &init_evt);
    ocl_check(err, "launch kernel");
    return init_evt;
}

int main(int argc, char *argv[]) {
    if (argc < 2 || argc > 5) {
        fprintf(stderr, "Usage: %s <nVertexes> [seed] [missCoeficient] [maxValue]\n", argv[0]);
        return 1;
    }

    const int v = atoi(argv[1]);
    const int seed = argc > 2 ? atoi(argv[2]) : 42;
    const int missCoeficient = argc > 3 ? atoi(argv[3]) : 2;
    const int maxWeight = argc > 4 ? atoi(argv[4]) : 10;

    if (v < 2) {
        fprintf(stderr, "Number of vertices must be at least 2\n");
        return 2;
    }

    //region Initialize OpenCL
    cl_platform_id p = select_platform();
    cl_device_id d = select_device(p);
    cl_context ctx = create_context(p, d);
    cl_command_queue que = create_queue(ctx, d);
    //endregion

    cl_program prog = create_program("kernels.cl", ctx, d);

    cl_int err;

    cl_kernel calculate_cost_k = clCreateKernel(prog, "calculateCost", &err);
    ocl_check(err, "create kernel calculateCost");

    size_t preferred_multiple_init;
    clGetKernelWorkGroupInfo(calculate_cost_k, d, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
                             sizeof(preferred_multiple_init), &preferred_multiple_init, NULL);

    size_t adjsize = v * v * sizeof(int);

    int *adj = malloc(adjsize);

    init_graph(adj, v, seed, missCoeficient, maxWeight);
    print_graph(adj, v);

    cl_mem d_adj = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, adjsize, adj, &err);
    ocl_check(err, "create d_adj");

    int totalPermutations = factorial(v - 1);
    int chunkSize = preferred_multiple_init * 16;
    int permssize = chunkSize * (v - 1) * sizeof(int);
    int *permutations = malloc(permssize);
    ocl_check(err, "create d_permutations");
    int minCost = INFINITY;
    int *path = malloc((v - 1) * sizeof(int));
    int i;
    int totalchunks = (totalPermutations + chunkSize - 1) / chunkSize;
    for (i = 0; i < v - 1; ++i) {
        path[i] = i + 1;
    }
    char continueLoop = 1;
    int nChunk = 0;
    while (continueLoop) {
        nChunk++;
        int currentSize = 0;
        do {
            for (i = 0; i < v - 1; ++i) {
                permutations[currentSize * (v - 1) + i] = path[i];
            }
            ++currentSize;
            if (currentSize == chunkSize) {
                break;
            }
            continueLoop = next_permutation(path, v - 1);
        } while (continueLoop);

        cl_mem d_permutations = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, permssize, permutations,
                                               &err);
        ocl_check(err, "create d_permutations");

        int costsize = currentSize * sizeof(int);

        cl_mem d_costs = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, costsize, NULL, &err);
        ocl_check(err, "create d_costs");

        cl_event init_evt = kernel(que, calculate_cost_k, preferred_multiple_init, d_adj, d_permutations, d_costs, v,
                                   currentSize);

        int *costs = malloc(costsize);
        cl_event read_evt;

        err = clEnqueueReadBuffer(que, d_costs, CL_TRUE, 0, costsize, costs, 1, &init_evt, &read_evt);
        ocl_check(err, "read costs");

        for (int i = 0; i < currentSize; ++i) {
            if (costs[i] < minCost) {
                minCost = costs[i];
            }
        }
        printf("Chunk %d/%d, Permutation %d/%d: minCost = %d\n", nChunk, totalchunks,
               (nChunk - 1) * chunkSize + currentSize,
               totalPermutations, minCost);

        clReleaseMemObject(d_permutations);
        clReleaseMemObject(d_costs);
    }

    printf("Min cost: %d\n", minCost);

    clReleaseKernel(calculate_cost_k);
    clReleaseMemObject(d_adj);
    clReleaseProgram(prog);
    clReleaseCommandQueue(que);
    clReleaseContext(ctx);

    return 0;
}
