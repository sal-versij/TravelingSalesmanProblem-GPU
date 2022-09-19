#include <stdio.h>
#include <stdlib.h>

#define CL_TARGET_OPENCL_VERSION 120

#include "ocl_boiler.h"
#include "setup.h"

struct ChunkRun {
    unsigned long long int size;
    double kernel_runtime;
    double write_runtime;
    double read_runtime;
    double total_runtime;
    double write_bw;
    double kernel_bw;
    double read_bw;
    double total_bw;
    double permsPerSec;
};

struct Result {
    unsigned long long int cost;
    unsigned long long int totalPermutations;
    unsigned long long int totalchunks;
    struct ChunkRun *chunkRuns;
    long double runtime;
};

void printResult(struct Result result) {
    printf("Cost: %llu\n", result.cost);
    printf("Total Permutations: %llu\n", result.totalPermutations);
    printf("Total Chunks: %llu\n", result.totalchunks);
    printf("Runtime: %Lf ms, %.3Lf perms/s, %.3Lf chunks/s\n", result.runtime,
           result.totalPermutations / result.runtime * 1000, result.totalchunks / result.runtime * 1000);
    printf("Chunk Runs:\n");
    for (int i = 0; i < result.totalchunks; i++) {
        printf("- Chunk %d:\n", i);
        printf("    Chunk Size: %llu\n", result.chunkRuns[i].size);
        printf("    Write: %f ms %.3f GB/s\n", result.chunkRuns[i].write_runtime, result.chunkRuns[i].write_bw);
        printf("    Kernel: %f ms %.3f GB/s\n", result.chunkRuns[i].kernel_runtime, result.chunkRuns[i].kernel_bw);
        printf("    Read: %f ms %.3f GB/s\n", result.chunkRuns[i].read_runtime, result.chunkRuns[i].read_bw);
        printf("    Total: %f ms %.3f GB/s\n", result.chunkRuns[i].total_runtime, result.chunkRuns[i].total_bw);
        printf("    Permutations Per Second: %f\n", result.chunkRuns[i].permsPerSec);
    }
}

cl_event kernel(
        cl_command_queue q, cl_kernel k, cl_event *waiting_event, cl_uint waiting_list_length,
        size_t preferred_multiple_init, cl_mem d_permutations, cl_mem d_adj, cl_mem d_costs, cl_int v,
        cl_int work_size) {
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
                                 waiting_list_length, waiting_event, &kernel_evt);
    ocl_check(err, "launch kernel");
    return kernel_evt;
}

int main(int argc, char *argv[]) {
    cl_int err;
    //region Params
    if (argc < 2 || argc > 6) {
        fprintf(stderr, "Usage: %s <nVertexes> [chunks] [seed] [missCoeficient] [maxWeight]\n", argv[0]);
        return 1;
    }
    int p = 1;

    const int v = atoi(argv[p++]);
    const int chunks = argc > p ? atoi(argv[p++]) : 1024;
    const int seed = argc > p ? atoi(argv[p++]) : 42;
    const int missCoeficient = argc > p ? atoi(argv[p++]) : 2;
    const int maxWeight = argc > p ? atoi(argv[p++]) : 100;

    if (v < 2) {
        fprintf(stderr, "Number of vertices must be at least 2\n");
        return 2;
    }
    //endregion

    struct Info info = initialize("GlobalArray_SingleResult_char");
    struct Result result = {0};

    //region Initialize Graph
    size_t adjsize = v * v * sizeof(int);

    int *adj = malloc(adjsize);

    init_graph(adj, v, seed, missCoeficient, maxWeight);
//    print_graph(adj, v);

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
    int costs_size = chunk_size * sizeof(int);

    char *permutations = malloc(permutations_size);

    char *path = malloc((v - 1) * sizeof(char));
    for (i = 1; i < v; ++i) {
        path[i - 1] = i;
    }
    int *costs = malloc(costs_size);
    result.totalPermutations = factorial(v - 1);
    result.totalchunks = (result.totalPermutations + chunk_size - 1) / chunk_size;
    result.chunkRuns = malloc(result.totalchunks * sizeof(struct ChunkRun));

    cl_mem d_permutations = clCreateBuffer(info.context, CL_MEM_READ_ONLY, permutations_size, NULL, &err);
    ocl_check(err, "create d_permutations");


    cl_mem d_costs = clCreateBuffer(info.context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, costs_size, NULL,
                                    &err);
    ocl_check(err, "create d_costs");

    cl_event write_evt;
    cl_event kernel_evt;
    cl_event read_evt;

    cl_event first_evt;
    cl_event last_evt;

    char continueLoop = 1;
    char first = 1;
    int nChunk = 0;
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

        clEnqueueMapBuffer(info.queue, d_permutations, CL_FALSE, CL_MAP_WRITE, 0, permutations_size, 0, NULL,
                           &write_evt, &err);
        ocl_check(err, "enqueue write");

        kernel_evt = kernel(
                info.queue, info.kernel, &write_evt, 1, info.preferred_multiple_init,
                d_permutations, d_adj, d_costs, v, current_number_of_permutations);

        err = clEnqueueReadBuffer(info.queue, d_costs, CL_TRUE, 0, costs_size, costs, 1, &kernel_evt, &read_evt);
        ocl_check(err, "read costs");

        for (i = 0; i < current_number_of_permutations; ++i) {
            if (costs[i] < minCost) {
                minCost = costs[i];
            }
        }

        struct ChunkRun chunkRun = {0};
        chunkRun.size = current_number_of_permutations;
        chunkRun.write_runtime = runtime_ms(write_evt);
        chunkRun.kernel_runtime = runtime_ms(kernel_evt);
        chunkRun.read_runtime = runtime_ms(read_evt);
        chunkRun.total_runtime = runtime_ms(write_evt) + runtime_ms(read_evt);
        chunkRun.write_bw = current_number_of_permutations * (v - 1) * sizeof(char) / chunkRun.read_runtime / 1e6;
        chunkRun.kernel_bw = 2.0 * v * current_number_of_permutations * sizeof(char) / chunkRun.read_runtime / 1e6;
        chunkRun.read_bw = costs_size / chunkRun.total_runtime / 1e6;
        chunkRun.total_bw =
                (current_number_of_permutations * (v - 1) * sizeof(char) +
                 2.0 * v * current_number_of_permutations * sizeof(char) +
                 costs_size) / chunkRun.total_runtime / 1e6;
        chunkRun.permsPerSec = current_number_of_permutations / chunkRun.total_runtime * 1000;
        printf("Chunk %d/%llu, Permutation %llu/%llu: minCost = %d\n", nChunk, result.totalchunks,
               (nChunk - 1) * chunk_size + current_number_of_permutations, result.totalPermutations, minCost);
//        printf("init: %g ms, %g GB/s; read: %g ms, %g GB/s; total: %g ms\n", chunkRun.kernel_runtime,
//               chunkRun.kernel_bw, chunkRun.read_runtime, chunkRun.read_bw, chunkRun.total_runtime);

        result.chunkRuns[nChunk - 1] = chunkRun;

        if (first) {
            first = 0;
            first_evt = write_evt;
        }
    }
    last_evt = read_evt;

//    printf("Min cost: %d\n", minCost);
    result.cost = minCost;
    result.runtime = total_runtime_ms(first_evt, last_evt);
    //endregion

    printResult(result);

    freeInfo(info);
    clReleaseMemObject(d_adj);
    clReleaseMemObject(d_permutations);
    clReleaseMemObject(d_costs);

    return 0;
}
