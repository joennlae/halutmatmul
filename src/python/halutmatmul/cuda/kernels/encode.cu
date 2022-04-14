extern "C" __global__ void halut_encode(const float X[],
                                        const float hash_info[],
                                        int group_ids[], int N, int D,
                                        int size) {
  // group_ids [N, C]
  // X [N, D]
  // hash_info [C, num_splits, info_offset + 3]
  // size = N * C

  // grid_dim = (N // rows_per_block (+ 1), )
  // block_dim = (rows_per_block, C)

  // WILL BE CHANGED
  const int num_splits = 4;
  const int C = 32;
  const int info_offset = 8;
  // CHANGE END

  const int K = num_splits * num_splits;
  const int hash_info_x = info_offset + 3;

  __shared__ float s_hash_info[C * num_splits * hash_info_x];

  const int cid = threadIdx.y;
  const int hash_info_offset = cid * num_splits * hash_info_x;
  const int row_offset_X = blockIdx.x * blockDim.x * D + threadIdx.x * D;
  const int tid = blockIdx.x * blockDim.x * blockDim.y +
                  threadIdx.x * blockDim.y + threadIdx.y;
  if (tid < size) {
    // load
    if (threadIdx.x == 0) {
      for (int i = 0; i < hash_info_x * num_splits; ++i) {
        s_hash_info[hash_info_offset + i] = hash_info[hash_info_offset + i];
      }
    }
    __syncthreads();
    int group_id = 0;

    float val = 0;
    float x = 0;
    float offset = 0;
    float scaleby = 0;
    for (int i = 0; i < num_splits; ++i) {
      int dim = __float2int_rn(
          s_hash_info[hash_info_offset + hash_info_x * i + info_offset]);
      val = s_hash_info[hash_info_offset + hash_info_x * i +
                        group_id]; // normally int
      scaleby =
          s_hash_info[hash_info_offset + hash_info_x * i + info_offset + 1];
      offset =
          s_hash_info[hash_info_offset + hash_info_x * i + info_offset + 2];
      x = (X[row_offset_X + dim] - offset) *
          scaleby; // TODO: this random access could be optimized when using
                   // column-major input
      group_id = group_id * 2 + (x > val ? 1 : 0);
    }
    group_ids[tid] = group_id + cid * K;
  }
}
