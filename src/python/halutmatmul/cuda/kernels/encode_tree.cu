extern "C" __global__ void halut_encode(const float X[],
                                        const float tree_info[],
                                        int group_ids[], int N, int D,
                                        int size) {
  // group_ids [N, C]
  // X [N, D]
  // tree_info [3 * B]
  //    [0:B-1]: dims
  //    [B:2B-1]: thresholds
  //    [2B:3B-1]: classes
  // size = N * C

  // grid_dim = (N // rows_per_block (+ 1), )
  // block_dim = (rows_per_block, C)

  // WILL BE CHANGED
  const int depth = 4;
  const int B = 16;
  const int K = 16;
  const int C = 32;
  // CHANGE END

  // B = leaf elements
  __shared__ float s_tree_info[C * 3 * B];

  // could be optimized be changing threadIdx.x, and threadIdx.y to optimize for
  // warp size
  const int cid = threadIdx.y;
  const int tree_info_offset = cid * 3 * B;
  const int row_offset_X = blockIdx.x * blockDim.x * D + threadIdx.x * D;
  const int tid = blockIdx.x * blockDim.x * blockDim.y +
                  threadIdx.x * blockDim.y + threadIdx.y;
  if (tid < size) {
    // load
    if (threadIdx.x == 0) {
      for (int i = 0; i < 3 * B; ++i) {
        s_tree_info[tree_info_offset + i] = tree_info[tree_info_offset + i];
      }
    }
    __syncthreads();
    int group_id = 0;

    float threshold = 0;
    float x = 0;
    int index_offset_helper = 1;
    for (int i = 0; i < depth; ++i) {
      int index_offset = index_offset_helper - 1;
      index_offset_helper *= 2;
      int dim = __float2int_rn(
          s_tree_info[tree_info_offset + index_offset + group_id]);
      threshold = s_tree_info[tree_info_offset + B + index_offset + group_id];
      x = X[row_offset_X + dim];
      group_id = group_id * 2 + (x > threshold ? 1 : 0);
    }
    int class_ =
        __float2int_rn(s_tree_info[tree_info_offset + 2 * B + group_id]);
    group_ids[tid] = class_ + cid * K;
  }
}