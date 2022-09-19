#define INFINITY 99999

// Inits a graph with n vertices with random weights and some missing edges(denoted by INFINITY)
void init_graph(int *adj, int v, int seed, int missCoeficient, int maxWeight) {
    srand(seed);
    int i, j;
    for (i = 0; i < v; i++) {
        for (j = 0; j < v; j++) {
            if (i == j) {
                adj[j * v + i] = INFINITY;
                continue;
            }

            int miss = rand() % missCoeficient;
            if (miss == 0) {
                adj[j * v + i] = INFINITY;
            } else {
                adj[j * v + i] = rand() % maxWeight;
                if (adj[j * v + i] == 0) {
                    adj[j * v + i] = 1;
                }
            }
        }
    }
}

// Prints the graph
void print_graph(int *adj, int v) {
    int i, j;
    for (i = 0; i < v; i++) {
        for (j = 0; j < v; j++) {
            if (adj[j * v + i] == INFINITY) {
                printf("-\t");
            } else {
                printf("%d\t", adj[j * v + i]);
            }
        }
        printf("\n");
    }
}

char next_permutation_chars(char *path, int v) {
    int i, j;
    for (i = v - 1; i > 0; i--) {
        if (path[i - 1] < path[i]) {
            break;
        }
    }
    if (i == 0) {
        return 0;
    }
    for (j = v - 1; j > i; j--) {
        if (path[j] > path[i - 1]) {
            break;
        }
    }
    char temp = path[i - 1];
    path[i - 1] = path[j];
    path[j] = temp;
    for (j = i, i = v - 1; i > j; i--, j++) {
        temp = path[i];
        path[i] = path[j];
        path[j] = temp;
    }
    return 1;
}

char next_permutation(int *path, int v) {
    int i, j;
    for (i = v - 1; i > 0; i--) {
        if (path[i - 1] < path[i]) {
            break;
        }
    }
    if (i == 0) {
        return 0;
    }
    for (j = v - 1; j > i; j--) {
        if (path[j] > path[i - 1]) {
            break;
        }
    }
    int temp = path[i - 1];
    path[i - 1] = path[j];
    path[j] = temp;
    for (j = i, i = v - 1; i > j; i--, j++) {
        temp = path[i];
        path[i] = path[j];
        path[j] = temp;
    }
    return 1;
}

unsigned long long factorial(int n) {
    unsigned long long result = 1;
    for (int i = 1; i <= n; i++) {
        result *= i;
    }
    return result;
}

struct Info {
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    size_t preferred_multiple_init;
};

void freeInfo(struct Info info) {
    clReleaseKernel(info.kernel);
    clReleaseProgram(info.program);
    clReleaseCommandQueue(info.queue);
    clReleaseContext(info.context);
}

struct Info initialize(const char *kernel_name) {
    cl_int err;
    //region Initialize OpenCL
    cl_platform_id p = select_platform();
    cl_device_id d = select_device(p);
    cl_context ctx = create_context(p, d);
    cl_command_queue que = create_queue(ctx, d);
    //endregion

    //region Initialize Kernel
    cl_program prog = create_program("kernels.cl", ctx, d);

    cl_kernel calculate_cost_k = clCreateKernel(prog, kernel_name, &err);
    ocl_check(err, "create kernel");

    size_t preferred_multiple_init;
    clGetKernelWorkGroupInfo(calculate_cost_k, d, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
                             sizeof(preferred_multiple_init), &preferred_multiple_init, NULL);
    //endregion

    return (struct Info) {p, d, ctx, que, prog, calculate_cost_k, preferred_multiple_init};
}

void AddKernelArg(cl_kernel kernel, cl_uint arg_index, size_t arg_size, void *arg_value) {
    cl_int err = clSetKernelArg(kernel, arg_index, arg_size, arg_value);
    ocl_check(err, "setting kernel arg %u", arg_index);
}
