#include <stdio.h>
#include <stdlib.h>

void calcPermutation(int permutation, int out[], int n) {
    int v = permutation;
    int *start = malloc((n << 1) * sizeof(int));
    int *end = malloc((n << 1) * sizeof(int));
    start[0] = 0;
    end[0] = n - 1;
    int n_j = 1;
    for (int i = n; i > 0; v /= i--) {
        int a = v % i;

        for (int j = 0; j < n_j; j++) {
//            printf("a: %d; j: %d\n", a, j);
//            for (int k = 0; k < n_j; k++) {
//                printf("pair #%d/%d: (%d, %d)\t", k + 1, n_j, start[k], end[k]);
//            }
//            printf("\n");

            a += start[j];
            if (a > end[j]) {
                a -= end[j] + 1;
                continue;
            }
            if (start[j] == a) {
                if (end[j] == a) {
                    for (int k = j; k < n_j; k++) {
                        start[k] = start[k + 1];
                        end[k] = end[k + 1];
                    }
                    n_j--;
                    break;
                }
                start[j] += 1;
                break;
            }
            if (end[j] == a) {
                end[j] -= 1;
                break;
            }
            for (int k = n_j - 1; k > j; k--) {
                start[k + 1] = start[k];
                end[k + 1] = end[k];
            }
            end[j + 1] = end[j];
            end[j] = a - 1;
            start[j + 1] = a + 1;
            end[j] = a - 1;
            ++n_j;
            break;
        }

//        printf("Final a: %d\n", a);

        out[n - i] = a;
    }

}

int factorial(int n) {
    int result = 1;
    for (int i = 1; i <= n; i++) {
        result *= i;
    }
    $();
    return result;
}

int singlePermutationMain(int argc, char **args) {
    if (argc != 3) {
        printf("Usage: %s <permutation> <digits>", args[0]);
        return -1;
    }

    int permutation = atoi(args[1]);
    int n = atoi(args[2]);
    int total = factorial(n);

    if (permutation > total) {
        permutation %= total;
        printf("Permutation too large, using %d instead\n", permutation);
    }

    printf("Permutation %d/%d of %d digits:\n", permutation, total, n);

    int *out = malloc(n * sizeof(int));

    calcPermutation(permutation, out, n);

    for (int i = 0; i < n; i++) {
        printf("%d\t", out[i]);
    }
    return 0;
}

int statsMain(int argc, char **args) {
    if (argc < 2 || argc > 3) {
        printf("Usage: %s <digits> [chunk-size]", args[0]);
        return -1;
    }

    int n = atoi(args[1]) - 1;
    int size;
    if (n < 255) {
        size = sizeof(unsigned char);
    } else if (n < 65535) {
        size = sizeof(unsigned short);
    } else {
        size = sizeof(unsigned int);
    }

    int chunkSize = 0;
    if (argc == 3) {
        chunkSize = atoi(args[2]);
    }

    int permutations = factorial(n);
    double total = (double) permutations * n * size;
    int i = 0;
    while (total > 1024) {
        total /= 1024;
        if (++i == 5)
            break;
    }

    char *units[] = {"B", "KB", "MB", "GB", "TB", "PB"};

    printf("Calculating stats for %d digits, %d permutations, with a typesize of %dB: %.1f%s\n",
           n+1, permutations, size, total, units[i]);

    if (chunkSize == 0) {
        return 0;
    }

    int chunks = (permutations + chunkSize - 1) / chunkSize;

    total = (double) chunkSize * n * size;
    i = 0;
    while (total > 1024) {
        total /= 1024;
        if (++i == 5)
            break;
    }
    printf("Using %d chunks of size %d, each chunk will take %.1f%s\n", chunks, chunkSize, total, units[i]);

    return 0;
}

int main(int argc, char **args) {
//    return singlePermutationMain(argc, args);
    return statsMain(argc, args);
}
