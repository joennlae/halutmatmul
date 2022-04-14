extern "C" __global__ void halut_read_acc_lut(const float lut[],
                                              const int A_enc[], float result[],
                                              int N, int M) {
  // lut [M, C, K]
  // result [N, M]
  // A_enc [N, C]

  // WILL BE CHANGED
  const int C = 32;
  const int K = 16;
  const int blocks = 8;
  const int rows = 128;
  // CHANGE END

  // load blockDim.y == 1 --> (C * M) + (K + N)
  // load blockDim.y == 8 --> (C * M) + (K * 8 + N / 8)
  // best case when K * blockDim.y ~ B / blockDim.y

  // blockDim.y = 8 = blocks calculate like explained above
  // MAX_THREADS (1080 TI) == 1024
  // grid_dim = (N // blockDim.y + 1, M // blockDim.y + 1 )
  // block_dim = (MAX_THREADS / blockDim.y, blockDim.y)

  // total shared storage = MAX_THREADS / blocks * C * 4
  // + C * K * blocks * 4
  __shared__ float s_lut[C * K * blocks];
  __shared__ int s_enc[rows * C];
  // results[n, m]
  const int m = blockIdx.y * blockDim.y + threadIdx.y;
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N && m < M) {
    if (threadIdx.x == 0) { // could be optimized
      for (int i = 0; i < C * K; i++) {
        s_lut[threadIdx.y * C * K + i] = lut[m * C * K + i];
      }
    }
    if (threadIdx.y == 0) {
      for (int i = 0; i < C; i++) {
        s_enc[threadIdx.x * C + i] = A_enc[n * C + i];
      }
    }
    __syncthreads();
    float res = 0;
    for (int i = 0; i < C; i++) {
      res += s_lut[threadIdx.y * C * K + s_enc[threadIdx.x * C + i]];
    }
    result[n * M + m] = res;
  }
}
