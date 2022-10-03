static int infinity = 99999;

kernel void Cached_GlobalArray_SingleResult(global int *permutations,
                                            global int *adjacencyMatrix,
                                            global int *costs, int v,
                                            int workSize) {
  int id = get_global_id(0);
  if (id >= workSize) {
    return;
  }
  // We are searching a loop, so doesn't matter from which vertex to start,
  // to reduce number of permutations to process we fix the first vertex to
  // vertex 0

  // each permutation has only the vertices from 1 to v-1

  int cost = 0;
  int previous = 0; // fixed vertex 0
  int current;
  int edge;

  for (int i = 0; i < v - 1; i++) {
    current = permutations[id * (v - 1) + i];

    edge = adjacencyMatrix[previous * v + current];

    if (edge == infinity) {
      cost = infinity;
      break;
    }
    cost += edge;
    previous = current;
  }

  if (cost != infinity) {
    // add the last edge to close the loop
    edge = adjacencyMatrix[previous * v];

    if (edge == infinity) {
      cost = infinity;
    } else {
      cost += edge;
    }
  }

  costs[id] = cost;
}

kernel void Cached_GlobalArray_SingleResult_char(global char *permutations,
                                                 global int *adjacencyMatrix,
                                                 global int *costs, int v,
                                                 int workSize) {
  int id = get_global_id(0);
  if (id >= workSize) {
    return;
  }

  // We are searching a loop, so doesn't matter from which vertex to start,
  // to reduce number of permutations to process we fix the first vertex to
  // vertex 0

  // each permutation has only the vertices from 1 to v-1

  int cost = 0;
  char previous = 0; // fixed vertex 0
  char current;
  int edge;

  for (int i = 0; i < v - 1; i++) {
    current = permutations[id * (v - 1) + i];

    edge = adjacencyMatrix[previous * v + current];

    if (edge == infinity) {
      cost = infinity;
      break;
    }
    cost += edge;
    previous = current;
  }

  if (cost != infinity) {
    // add the last edge to close the loop
    edge = adjacencyMatrix[previous * v];

    if (edge == infinity) {
      cost = infinity;
    } else {
      cost += edge;
    }
  }

  costs[id] = cost;
}

kernel void Cached_LocalArray_SingleResult_char(global char *permutations,
                                                global int *_adjacencyMatrix,
                                                global int *costs, int v,
                                                int workSize,
                                                local int *adjacencyMatrix) {
  int id = get_global_id(0);
  if (id >= workSize) {
    return;
  }

  // Load adjacency matrix to local memory
  for (int i = get_local_id(0); i < v * v; i += get_local_size(0)) {
    adjacencyMatrix[i] = _adjacencyMatrix[i];
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  // We are searching a loop, so doesn't matter from which vertex to start,
  // to reduce number of permutations to process we fix the first vertex to
  // vertex 0

  // each permutation has only the vertices from 1 to v-1

  int cost = 0;
  char previous = 0; // fixed vertex 0
  char current;
  int edge;

  for (int i = 0; i < v - 1; i++) {
    current = permutations[id * (v - 1) + i];

    edge = adjacencyMatrix[previous * v + current];

    if (edge == infinity) {
      cost = infinity;
      break;
    }
    cost += edge;
    previous = current;
  }

  if (cost != infinity) {
    // add the last edge to close the loop
    edge = adjacencyMatrix[previous * v];

    if (edge == infinity) {
      cost = infinity;
    } else {
      cost += edge;
    }
  }

  costs[id] = cost;
}

kernel void Cached_LocalArray_SingleResult_char_v2(
    global char *permutations, global int *_adjacencyMatrix, global int *costs,
    int v, int workSize, int permutationStride, local int *adjacencyMatrix) {
  int id = get_global_id(0);
  if (id >= workSize) {
    return;
  }

  // Load adjacency matrix to local memory
  for (int i = get_local_id(0); i < v * v; i += get_local_size(0)) {
    adjacencyMatrix[i] = _adjacencyMatrix[i];
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  // We are searching a loop, so doesn't matter from which vertex to start,
  // to reduce number of permutations to process we fix the first vertex to
  // vertex 0

  // each permutation has only the vertices from 1 to v-1

  int cost = 0;
  char previous = 0; // fixed vertex 0
  char current;
  int edge;

  for (int i = 0; i < v - 1; i++) {
    current = permutations[id + permutationStride * i];

    edge = adjacencyMatrix[previous * v + current];

    if (edge == infinity) {
      cost = infinity;
      break;
    }
    cost += edge;
    previous = current;
  }

  if (cost != infinity) {
    // add the last edge to close the loop
    edge = adjacencyMatrix[previous * v];

    if (edge == infinity) {
      cost = infinity;
    } else {
      cost += edge;
    }
  }

  costs[id] = cost;
}

kernel void Procedural_Single(global int *_adjacencyMatrix, global int *costs,
                              int v, int workSize, local int *adjacencyMatrix,
                              local char *start, local char *end,
                              local char *next) {
  int id = get_global_id(0);
  if (id >= workSize) {
    return;
  }

  int l_id = get_local_id(0);
  int l_size = get_local_size(0);
  // Load adjacency matrix to local memory
  for (int i = l_id; i < v * v; i += l_size) {
    adjacencyMatrix[i] = _adjacencyMatrix[i];
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  // We are searching a loop, so doesn't matter from which vertex to start,
  // to reduce number of permutations to process we fix the first vertex to
  // vertex 0

  // each permutation has only the vertices from 1 to v-1
  int n = v - 1;
  int previous = 0; // fixed vertex 0
  int perm = id;
  int edge = 0;
  int cost = 0;

  start[l_id + l_size * 0] = 0;
  end[l_id + l_size * 0] = n - 1;
  int n_j = 1;
  int current;

  for (int i = n; i > 0; perm /= i--) {
    current = perm % i;

    for (int j = 0; j < n_j; j++) {
      current += start[l_id + l_size * j];
      if (current > end[l_id + l_size * j]) {
        current -= end[l_id + l_size * j] + 1;
        continue;
      }

      if (start[l_id + l_size * j] == current) {
        if (end[l_id + l_size * j] == current) {
          start[l_id + l_size * j] = 0;
          end[l_id + l_size * j] = -1;
          break;
        }
        start[l_id + l_size * j] += 1;
        break;
      }
      if (end[l_id + l_size * j] == current) {
        end[l_id + l_size * j] -= 1;
        break;
      }
      start[l_id + l_size * n_j] = current + 1;

      next[l_id + l_size * n_j] = next[l_id + l_size * j];
      next[l_id + l_size * j] = n_j;

      end[l_id + l_size * n_j] = end[l_id + l_size * j];
      end[l_id + l_size * j] = current - 1;

      ++n_j;
      break;
    }

    // Fix the permutation so that the vertexes are in range 1 to v-1
    ++current;

    edge = adjacencyMatrix[previous * v + current];

    if (edge == infinity) {
      cost = infinity;
      break;
    }
    cost += edge;
    previous = current;
  }

  if (cost != infinity) {
    // add the last edge to close the loop
    edge = adjacencyMatrix[previous * v];

    if (edge == infinity) {
      cost = infinity;
    } else {
      cost += edge;
    }
  }

  costs[id] = cost;
}

kernel void Procedural_Sliding_Window(global int *_adjacencyMatrix,
                                      global int *_costs, int v, ulong workSize,
                                      local int *adjacencyMatrix,
                                      local int *costs, local char *start,
                                      local char *end, local char *next) {
  int g_id = get_global_id(0);
  int l_id = get_local_id(0);
  int l_size = get_local_size(0);

  // Load adjacency matrix to local memory
  for (int i = l_id; i < v * v; i += l_size) {
    adjacencyMatrix[i] = _adjacencyMatrix[i];
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  costs[l_id] = infinity;
  for (ulong id = l_id; id < workSize; id += l_size) {
    //  We are searching a loop, so doesn't matter from which vertex to start,
    //  to reduce number of permutations to process we fix the first vertex to
    //  vertex 0

    // each permutation has only the vertices from 1 to v-1
    int n = v - 1;
    char previous = 0; // fixed vertex 0
    char current;
    ulong perm = id;
    int edge = 0;
    int cost = 0;
    int n_j = 1;

    start[l_id + l_size * 0] = 0;
    end[l_id + l_size * 0] = n - 1;

    for (char i = n; i > 0; perm /= i--) {
      current = perm % i;

      for (int j = 0; j < n_j; j++) {
        current += start[l_id + l_size * j];
        if (current > end[l_id + l_size * j]) {
          current -= end[l_id + l_size * j] + 1;
          continue;
        }

        if (start[l_id + l_size * j] == current) {
          if (end[l_id + l_size * j] == current) {
            start[l_id + l_size * j] = 0;
            end[l_id + l_size * j] = -1;
            break;
          }
          start[l_id + l_size * j] += 1;
          break;
        }
        if (end[l_id + l_size * j] == current) {
          end[l_id + l_size * j] -= 1;
          break;
        }
        start[l_id + l_size * n_j] = current + 1;

        next[l_id + l_size * n_j] = next[l_id + l_size * j];
        next[l_id + l_size * j] = n_j;

        end[l_id + l_size * n_j] = end[l_id + l_size * j];
        end[l_id + l_size * j] = current - 1;

        ++n_j;
        break;
      }

      // Fix the permutation so that the vertexes are in range 1 to v-1
      ++current;

      edge = adjacencyMatrix[previous * v + current];

      if (edge == infinity) {
        cost = infinity;
        break;
      }
      cost += edge;
      previous = current;
    }

    if (cost != infinity) {
      // add the last edge to close the loop
      edge = adjacencyMatrix[previous * v];

      if (edge == infinity) {
        cost = infinity;
      } else {
        cost += edge;
      }
    }

    if (cost < costs[l_id]) {
      costs[l_id] = cost;
    }
  }
  _costs[g_id] = costs[l_id];
}

kernel void Procedural_Sliding_Window_nwg(global int *_adjacencyMatrix,
                                          global int *_costs, int v,
                                          ulong workSize,
                                          local int *adjacencyMatrix,
                                          local int *costs, local char *start,
                                          local char *end, local char *next) {
  int l_id = get_local_id(0);
  int l_size = get_local_size(0);
  int g_id = get_global_id(0);
  int g_size = get_global_size(0);

  // Load adjacency matrix to local memory
  for (int i = l_id; i < v * v; i += l_size) {
    adjacencyMatrix[i] = _adjacencyMatrix[i];
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  costs[l_id] = infinity;
  for (ulong id = g_id; id < workSize; id += g_size) {
    //  We are searching a loop, so doesn't matter from which vertex to start,
    //  to reduce number of permutations to process we fix the first vertex to
    //  vertex 0

    // each permutation has only the vertices from 1 to v-1
    int n = v - 1;
    char previous = 0; // fixed vertex 0
    char current;
    ulong perm = id;
    int edge = 0;
    int cost = 0;
    int n_j = 1;

    start[l_id + l_size * 0] = 0;
    end[l_id + l_size * 0] = n - 1;

    for (char i = n; i > 0; perm /= i--) {
      current = perm % i;

      for (int j = 0; j < n_j; j++) {
        current += start[l_id + l_size * j];
        if (current > end[l_id + l_size * j]) {
          current -= end[l_id + l_size * j] + 1;
          continue;
        }

        if (start[l_id + l_size * j] == current) {
          if (end[l_id + l_size * j] == current) {
            start[l_id + l_size * j] = 0;
            end[l_id + l_size * j] = -1;
            break;
          }
          start[l_id + l_size * j] += 1;
          break;
        }
        if (end[l_id + l_size * j] == current) {
          end[l_id + l_size * j] -= 1;
          break;
        }
        start[l_id + l_size * n_j] = current + 1;

        next[l_id + l_size * n_j] = next[l_id + l_size * j];
        next[l_id + l_size * j] = n_j;

        end[l_id + l_size * n_j] = end[l_id + l_size * j];
        end[l_id + l_size * j] = current - 1;

        ++n_j;
        break;
      }

      // Fix the permutation so that the vertexes are in range 1 to v-1
      ++current;

      edge = adjacencyMatrix[previous * v + current];

      if (edge == infinity) {
        cost = infinity;
        break;
      }
      cost += edge;
      previous = current;
    }

    if (cost != infinity) {
      // add the last edge to close the loop
      edge = adjacencyMatrix[previous * v];

      if (edge == infinity) {
        cost = infinity;
      } else {
        cost += edge;
      }
    }

    if (cost < costs[l_id]) {
      costs[l_id] = cost;
    }
  }
  _costs[g_id] = costs[l_id];
}

kernel void search_min(global int *restrict output,
                       const global int4 *restrict input,
                       local int *restrict lmem, int nquarts) {
  int val = infinity;

  for (int g_id = get_global_id(0); g_id < nquarts;
       g_id += get_global_size(0)) {
    int4 q = input[g_id];
    if (q.x < val)
      val = q.x;
    if (q.y < val)
      val = q.y;
    if (q.z < val)
      val = q.z;
    if (q.w < val)
      val = q.w;
  }

  const int l_id = get_local_id(0);

  lmem[l_id] = val;

  for (int stride = get_local_size(0) / 2; stride > 0; stride /= 2) {
    barrier(CLK_LOCAL_MEM_FENCE);
    if (l_id < stride) {
      if (lmem[l_id + stride] < lmem[l_id])
        lmem[l_id] = lmem[l_id + stride];
    }
  }

  if (l_id == 0)
    output[get_group_id(0)] = lmem[0];
}