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

int factorial(int n) {
    int result = 1;
    for (int i = 1; i <= n; i++) {
        result *= i;
    }
    return result;
}

void AddKernelArg(cl_kernel kernel, cl_uint arg_index, size_t arg_size, void *arg_value) {
    cl_int err = clSetKernelArg(kernel, arg_index, arg_size, arg_value);
    ocl_check(err, "setting kernel arg %u", arg_index);
}
