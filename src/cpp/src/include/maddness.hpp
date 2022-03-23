#ifndef __MADDNESS_HPP
#define __MADDNESS_HPP

#include "mithral.hpp"

template <class InputT> struct maddness_amm_task {
  using traits = mithral_input_type_traits<InputT>;
  using scale_t = typename traits::encoding_scales_type;
  using offset_t = typename traits::encoding_offsets_type;
  using output_t = typename traits::output_type;
  static constexpr int scan_block_nrows = 32;
  static constexpr int ncentroids = 16;
  static constexpr int nsplits_per_codebook = 4;
  static constexpr int max_splitvals = 1 << 4;

  maddness_amm_task(int N, int D, int M, int ncodebooks, float lut_work_const)
      : N_padded(N % scan_block_nrows == 0
                     ? N
                     : N + (scan_block_nrows - (N % scan_block_nrows))),
        centroids(ncentroids * ncodebooks, D),
        nsplits(ncodebooks * nsplits_per_codebook), splitdims(nsplits),
        splitvals(max_splitvals, nsplits), encode_scales(nsplits),
        encode_offsets(nsplits),
        nnz_per_centroid(lut_work_const > 0 ? lut_work_const * D / ncodebooks
                                            : D),
        idxs(ncodebooks, nnz_per_centroid),
        amm(N_padded, D, M, ncodebooks, centroids.data(), splitdims.data(),
            splitvals.data(), encode_scales.data(), encode_offsets.data(),
            idxs.data(), nnz_per_centroid),
        X(N_padded, D), Q(D, M) {
    centroids.setRandom();
    splitdims.setRandom();
    for (int i = 0; i < splitdims.size(); i++) {
      splitdims(i) = splitdims(i) % D;
    }
    splitvals.setRandom();
    encode_scales.setRandom();
    encode_offsets.setRandom();

    // randomly initialize idxs, ensuring all are unique and < D
    idxs.setRandom();
    int all_idxs[D];
    for (int i = 0; i < D; i++) {
      all_idxs[i] = i;
    }
    std::random_device rd;
    std::mt19937 g(rd()); // why can't shuffle just create its own...
    for (int c = 0; c < ncodebooks; c++) { // random sequential idxs
      std::shuffle(all_idxs, all_idxs + D, g);
      std::sort(all_idxs, all_idxs + nnz_per_centroid);
      for (int j = 0; j < nnz_per_centroid; j++) {
        idxs(c, j) = all_idxs[j];
      }
    }

    X.setRandom();
    Q.setRandom();
  }

  void encode() { amm.encode(X.data()); }
  void lut() { amm.lut(Q.data()); }
  void scan() { amm.scan(); }

  void run_matmul(bool create_lut = true) {
    encode();
    if (create_lut) {
      lut();
    }
    scan();
  }

  const ColMatrix<output_t> &output() const { return amm.out_mat; }

  // stuff we pass into the amm object (would be learned during training)
  int N_padded;
  ColMatrix<float> centroids;
  int nsplits;
  RowVector<uint32_t> splitdims;
  ColMatrix<int8_t> splitvals;
  RowVector<scale_t> encode_scales;
  RowVector<offset_t> encode_offsets;
  int nnz_per_centroid;
  RowMatrix<int> idxs;

  // amm object
  mithral_amm<InputT> amm;

  // random data
  ColMatrix<InputT> X;
  ColMatrix<float> Q;
};

#endif // __MADDNESS_HPP