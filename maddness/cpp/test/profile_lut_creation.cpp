//
//  profile_lut_creation.cpp
//  Bolt
//
//  Created by DB on 12/10/19.
//  Copyright © 2019 D Blalock. All rights reserved.
//

#include <catch2/catch_test_macros.hpp>
#include "amm_common.hpp"
#include "bolt.hpp"
// #include "product_quantize.hpp"

void _profile_lut_mithral(int nrows, int ncols, int nbytes,
                          float lut_work_const)
// std::vector<float> lut_work_consts)
{
  int ncodebooks = nbytes * 2;
  // compute nnz to use for idxs
  // float max_const = lut_work_consts[0];
  // for (int i = 0; i < lut_work_consts.size(); i++) {
  //     max_const = MAX(max_const, lut_work_consts[i]);
  // }
  // int nnz = max_const > 0 ? ncols * lut_work_const : ncols;
  int nnz = lut_work_const > 0 ? ncols * (lut_work_const / ncodebooks) : ncols;
  // printf("nnz: %d\n", nnz);

  ColMatrix<float> centroids16(ncodebooks * 16, ncols);
  centroids16.setRandom();
  RowMatrix<float> X(nrows, ncols);
  X.setRandom();
  RowMatrix<uint8_t> lut_out(nrows, ncodebooks * 16);
  lut_out.setRandom();
  RowMatrix<float> lut_f32_out(nrows, ncodebooks * 16);
  lut_f32_out.setRandom();
  RowVector<float> offsets(ncodebooks);
  offsets.setRandom();
  // creating random sorted idxs for sparse centroids is a pain
  RowMatrix<int> idxs(ncodebooks, nnz);
  int all_idxs[ncols];
  for (int i = 0; i < ncols; i++) {
    all_idxs[i] = i;
  }
  std::random_device rd;
  std::mt19937 g(rd()); // why can't shuffle just create its own...
  for (int c = 0; c < ncodebooks; c++) { // random sequential idxs
    std::shuffle(all_idxs, all_idxs + ncols, g);
    std::sort(all_idxs, all_idxs + nnz);
    for (int j = 0; j < nnz; j++) {
      idxs(c, j) = all_idxs[j];
    }
  }

  float offset = 0.;
  float scale = 1.;

  std::string msg;
  auto fmt_as_cppstring = string_with_format(
      "%%-20s, N D C B lut_work_const:, %7d, %3d, %3d, %2d, %%4.1f,\t", nrows,
      ncols, ncodebooks, nbytes, lut_work_const);
  auto fmt = fmt_as_cppstring.c_str();

  // ------------------------ mithral
  if (lut_work_const < 0) {
    msg = string_with_format(fmt, "mithral lut dense", -1.f);
    REPEATED_PROFILE_DIST_COMPUTATION(
        kNreps, msg, kNtrials, lut_out.data(), lut_out.size(),
        (mithral_lut_dense(X.data(), nrows, ncols, ncodebooks,
                           centroids16.data(), offset, scale,
                           lut_f32_out.data(), lut_out.data())));
  } else {
    msg = string_with_format(fmt, "mithral lut sparse", lut_work_const);
    REPEATED_PROFILE_DIST_COMPUTATION(
        kNreps, msg, kNtrials, lut_out.data(), lut_out.size(),
        (mithral_lut_sparse(X.data(), nrows, ncols, ncodebooks,
                            centroids16.data(), idxs.data(), nnz, offset, scale,
                            lut_f32_out.data(), lut_out.data())));
  }
}

TEST_CASE("vq lut timing", "[amm][lut][profile]") {
  //    static constexpr int nrows = 10 * 1024;
  static constexpr int nrows = 1 << 14;

  printf("algo, _0, N, D, C, B, lut_work_const, "
         "_1, latency0, _2, latency1, _3, latency2, _4,"
         "latency3, _5, latency4, _6\n");

  std::vector<int> all_ncols{32, 64, 128, 256, 512, 1024};
  // std::vector<int> all_ncodebooks {16, 32, 64};
  std::vector<int> all_nbytes{8, 16, 32, 64};
  std::vector<float> lut_work_consts{-1.f, 2.f, 4.f};
  // std::vector<float> lut_work_consts {-1};
  // std::vector<float> lut_work_consts {2.f};
  for (auto ncols : all_ncols) {
    // for (auto c : all_ncodebooks) {
    for (auto b : all_nbytes) {
      for (auto w : lut_work_consts) {
        _profile_lut_mithral(nrows, ncols, b, w);
      }
    }
  }
}
