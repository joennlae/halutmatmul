//
//  profile_amm.hpp
//  Bolt
//
//  Created by DB on 12/10/19.
//  Copyright © 2019 D Blalock. All rights reserved.
//

#ifndef profile_amm_h
#define profile_amm_h

#include "amm_common.hpp"
#include <Eigen/SparseCore>

struct MatmulTaskShape {
  int N, D, M;
  const char *name;
};
// static constexpr MatmulTaskShape kCaltechTaskShape {49284, 27, 2, "Caltech"};
static constexpr MatmulTaskShape kCaltechTaskShape0{
    (224 - 3 + 1) * (224 - 3 + 1), 3 * (3 * 3), 2, "Caltech3x3"}; // 49284, 27
static constexpr MatmulTaskShape kCaltechTaskShape1{
    (224 - 5 + 1) * (224 - 5 + 1), 3 * (5 * 5), 2, "Caltech5x5"}; // 48400, 75
static constexpr MatmulTaskShape kCifar10TaskShape{10000, 512, 10, "Cifar10"};
static constexpr MatmulTaskShape kCifar100TaskShape{10000, 512, 100,
                                                    "Cifar100"};
// 10000 * 10, 512, 100, "Cifar100"};
// static constexpr MatmulTaskShape kUcrTaskShape {1000, 320, 128, "UCR"};
static constexpr MatmulTaskShape kUcrTaskShape0{1000, 320, 64, "Ucr64"};
static constexpr MatmulTaskShape kUcrTaskShape1{1000, 320, 128, "Ucr128"};
static constexpr MatmulTaskShape kUcrTaskShape2{1000, 320, 256, "Ucr256"};

namespace {
// ================================================================ mithral

template <class InputT> struct mithral_amm_task {
  using traits = mithral_input_type_traits<InputT>;
  using scale_t = typename traits::encoding_scales_type;
  using offset_t = typename traits::encoding_offsets_type;
  using output_t = typename traits::output_type;
  static constexpr int scan_block_nrows = 32;
  static constexpr int ncentroids = 16;
  static constexpr int nsplits_per_codebook = 4;
  static constexpr int max_splitvals = 1 << 4;

  mithral_amm_task(int N, int D, int M, int ncodebooks, float lut_work_const)
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

template <class InputT = float>
void _profile_mithral(const char *dset_name, uint32_t N, uint32_t D, uint32_t M,
                      int ncodebooks, float lut_work_const = 2) {
  if ((lut_work_const > 0) && (lut_work_const > ncodebooks)) {
    return;
  }
  mithral_amm_task<InputT> task(N, D, M, ncodebooks, lut_work_const);

  // mithral_amm_task<InputT> task_dense(N, D, M, ncodebooks, -1);

  std::string msg;
  auto dtype_str = input_type_traits<InputT>{}.name;

  // auto fmt = "%7s, %3s, %22s, N, D, M, C, lut_work_coef:\t"
  //         "%6d, %3d, %3d, %2d, %.1f\t";
  auto fmt_as_cppstring = string_with_format(
      "%s, %-3s, %%-22s, N D M C lut_work_coef:,"
      "%6d, %3d, %3d, %2d, %4.1f,\t",
      dset_name, dtype_str, N, D, M, ncodebooks, lut_work_const);
  auto fmt = fmt_as_cppstring.c_str();
  // printf("fmt string: %s\n", fmt.c_str());
  // fmt = string_with_format()

  if (lut_work_const < 0) { // dense centroids
    msg = string_with_format(fmt, "amm mithral nolut");
    REPEATED_PROFILE_DIST_COMPUTATION(
        kNreps, msg, kNtrials, task.output().data(), task.output().size(),
        task.run_matmul(false));
    msg = string_with_format(fmt, "amm mithral denselut");
    REPEATED_PROFILE_DIST_COMPUTATION(
        kNreps, msg, kNtrials, task.output().data(), task.output().size(),
        task.run_matmul(true));
    msg = string_with_format(fmt, "mithral lut dense");
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, msg, kNtrials,
                                      task.output().data(),
                                      task.output().size(), task.lut());

    // these don't actually have anything to do with the lut_work_const;
    // I'm just putting them in this block so that they only get executed
    // once across all the different lut consts
    msg = string_with_format(fmt, "amm mithral enc");
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, msg, kNtrials,
                                      task.output().data(),
                                      task.output().size(), task.encode());
    msg = string_with_format(fmt, "amm mithral scan", -1.f);
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, msg, kNtrials,
                                      task.output().data(),
                                      task.output().size(), task.scan());
  } else { // sparse centroids
    msg = string_with_format(fmt, "amm mithral sparselut");
    REPEATED_PROFILE_DIST_COMPUTATION(
        kNreps, msg, kNtrials, task.output().data(), task.output().size(),
        task.run_matmul(true));
    msg = string_with_format(fmt, "mithral lut sparse");
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, msg, kNtrials,
                                      task.output().data(),
                                      task.output().size(), task.lut());
  }

  // if (ncodebooks >= lut_work_const) {
  //     msg = string_with_format(fmt, "amm mithral sparselut", lut_work_const);
  //     REPEATED_PROFILE_DIST_COMPUTATION(kNreps, msg, kNtrials,
  //         task.output().data(), task.output().size(),
  //         task.run_matmul(true));
  // }
  // msg = string_with_format( // time if lut already created
  //     "%3s amm mithral nolut      N, D, M, C, lut_work_coef:\t"
  //         "%6d, %3d, %3d, %2d, %.1f\t",
  //     dtype_str, N, D, M, ncodebooks, -1.f);
  // "%3s amm mithral nolut      N, D, M, C:\t\t\t\t\t"
  //     "%6d, %3d, %3d, %2d\t\t",
  // dtype_str, N, D, M, ncodebooks);
  // msg = string_with_format(fmt, dset_name, "amm mithral nolut",
  //     dtype_str, N, D, M, ncodebooks, -1.f);

  // // using dense centroids, which slows down LUT creation
  // auto orig_nnz_per_centroid = task.amm.nnz_per_centroid;
  // task.amm.nnz_per_centroid = -1;
  // msg = string_with_format(fmt, "amm mithral denselut", -1.f);
  // REPEATED_PROFILE_DIST_COMPUTATION(kNreps, msg, kNtrials,
  //     task.output().data(), task.output().size(),
  //     task.run_matmul(true));
  // msg = string_with_format(fmt, "amm mithral lut dense", -1.f);
  // REPEATED_PROFILE_DIST_COMPUTATION(kNreps, msg, kNtrials,
  //     task.output().data(), task.output().size(),
  //     task.lut());
  // task.amm.nnz_per_centroid = orig_nnz_per_centroid;

  // back to sparse centroids
}

template <class InputT = float>
void _profile_mithral(const MatmulTaskShape &shape, std::vector<int> ncodebooks,
                      std::vector<float> lut_work_consts) {
  auto dtype_name = input_type_traits<InputT>{}.name;
  printf("------------------------ %s %s\n", shape.name, dtype_name);
  for (auto c : ncodebooks) {
    printf("---- ncodebooks=%d\n", c);
    for (auto lutconst : lut_work_consts) {
      _profile_mithral<InputT>(shape.name, shape.N, shape.D, shape.M, c,
                               lutconst);
    }
  }
}

} // anonymous namespace
#endif /* profile_amm_h */
