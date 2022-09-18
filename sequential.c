#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#define INFINITY 99999

// Inits a graph with n vertices with random weights and some missing edges(denoted by INFINITY)
void init_graph(int **adj, int v, int seed, int missCoeficient, int maxWeight) {
    srand(seed);
    int i, j;
    for (i = 0; i < v; i++) {
        for (j = 0; j < v; j++) {
            if (i == j) {
                adj[i][j] = INFINITY;
                continue;
            }

            int miss = rand() % missCoeficient;
            if (miss == 0) {
                adj[i][j] = INFINITY;
            } else {
                adj[i][j] = rand() % maxWeight;
                if (adj[i][j] == 0) {
                    adj[i][j] = 1;
                }
            }
        }
    }
}

// Prints the graph
void print_graph(int **adj, int v) {
    int i, j;
    for (i = 0; i < v; i++) {
        for (j = 0; j < v; j++) {
            if (adj[i][j] == INFINITY) {
                printf("-\t");
            } else {
                printf("%d\t", adj[i][j]);
            }
        }
        printf("\n");
    }
}

bool next_permutation(int *path, int v) {
    int i, j;
    for (i = v - 1; i > 0; i--) {
        if (path[i - 1] < path[i]) {
            break;
        }
    }
    if (i == 0) {
        return false;
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
    return true;
}

// Bruteforce of Traveling Salesman Problem
void bruteforce(int **adj, int v) {
    printf("----------\n");
    long min_cost = INFINITY;
    long cost = 0;
    int *path = malloc((v - 1) * sizeof(int));
    int i;
    for (i = 0; i < v - 1; i++) {
        path[i] = i + 1;
    }
    int count = 0;
    do {
        printf("0\t");
        for (i = 0; i < v - 1; i++) {
            printf("%d\t", path[i]);
        }
        printf("\n");

        int cost = 0;
        int previous = 0; // fixed vertex 0
        int edge;
        for (int i = 0; i < v - 1; ++i) {
            edge = adj[previous][path[i]];
            if (edge == INFINITY) {
                cost = INFINITY;
                break;
            }
            cost += edge;
            previous = path[i];
        }

        if (cost != INFINITY) {
            // add the last edge to close the loop
            edge = adj[0][previous];
            if (edge == INFINITY) {
                cost = INFINITY;
            } else {
                cost += edge;
            }
        }

        if (cost < min_cost) {
            min_cost = cost;
            printf("0\t");
            for (i = 0; i < v - 1; i++) {
                printf("%d\t", path[i]);
            }
            printf("\nMin cost: %ld\n----------\n", min_cost);
        }
        cost = 0;

        count++;
    } while (next_permutation(path, v - 1));

    printf("Total permutations: %d\n", count);

    free(path);
}

int main(int argc, char **args) {
    // Usage: bruteforce <n-vertices> [seed]

    if (argc < 2 || argc > 3) {
        printf("Usage: bruteforce <n-vertices> [seed]\n");
        return 1;
    }

    int v = atoi(args[1]);
    int seed = 42;
    if (argc > 2) {
        seed = atoi(args[2]);
    }

    if (v < 2) {
        printf("Number of vertices must be at least 2\n");
        return 1;
    }

    int **adj = malloc(v * sizeof(int *));

    for (int i = 0; i < v; i++) {
        adj[i] = malloc(v * sizeof(int));
    }

    init_graph(adj, v, seed, 2, 10);

    print_graph(adj, v);

    bruteforce(adj, v);
}