//
//  avx_utils.hpp
//  Dig
//
//  Created by DB on 2017-2-7
//  Copyright (c) 2016 DB. All rights reserved.
//

#ifndef __AVX_UTILS_HPP
#define __AVX_UTILS_HPP

#include "immintrin.h"
#include "testing_utils.hpp"

#ifndef MAX
#define MAX(x, y) ((x) < (y) ? (y) : (x))
#endif
#ifndef MIN
#define MIN(x, y) ((x) > (y) ? (y) : (x))
#endif

static_assert(__AVX2__, "AVX 2 is required! Try --march=native or -mavx2");

// ================================================================ Functions

static inline float pfirst(const __m256 &a) {
  return _mm_cvtss_f32(_mm256_castps256_ps128(a));
}

static inline int32_t pfirst(const __m256i &a) {
  return _mm_cvtsi128_si32(_mm256_castsi256_si128(a));
}

// returns a * b + c, elementwise
inline __m256 fma(__m256 a, __m256 b, __m256 c) {
  __m256 res = c;
  __asm__("vfmadd231ps %[a], %[b], %[c]"
          : [c] "+x"(res)
          : [a] "x"(a), [b] "x"(b));
  return res;
}

inline __m256i avg_epu8(__m256i a, __m256i b) {
  __m256i res = _mm256_undefined_si256();
  __asm__("vpavgb %[a], %[b], %[c]" : [c] "=x"(res) : [a] "x"(a), [b] "x"(b));
  return res;
}

inline uint64_t popcount_u64(uint64_t a) {
  uint64_t res;
  __asm__("popcnt %[in], %[out]" : [out] "=r"(res) : [in] "r"(a));
  return res;
}

// ------------------------------------------------ other avx utils

template <class T> static int8_t msb_idx_u32(T x) {
  // return 8*sizeof(uint32_t) - 1 - __builtin_clzl((uint32_t)x);
  // XXX if sizeof(uinsigned int) != 4, this will break
  static_assert(sizeof(unsigned int) == 4,
                "XXX Need to use different builtin for counting leading zeros");
  return ((uint32_t)31) - __builtin_clz((uint32_t)x);
}

template <class T> static inline __m256i load_si256i(T *ptr) {
  return _mm256_load_si256((__m256i *)ptr);
}
template <class T> static inline __m128i load_si128i(T *ptr) {
  return _mm_load_si128((__m128i *)ptr);
}

template <class T> static inline __m256i stream_load_si256i(T *ptr) {
  return _mm256_stream_load_si256((__m256i *)ptr);
}

static inline __m256i broadcast_min(const __m256i a) {
  // swap upper and lower 128b halves, then min with original
  auto tmp = _mm256_min_epi32(a, _mm256_permute2x128_si256(a, a, 1));
  // swap upper and lower halves within each 128b half, then min
  auto tmp2 =
      _mm256_min_epi32(tmp, _mm256_shuffle_epi32(tmp, _MM_SHUFFLE(1, 0, 3, 2)));
  // alternative elements have min of evens and min of odds, respectively;
  // so swap adjacent pairs of elements within each 128b half
  return _mm256_min_epi32(tmp2,
                          _mm256_shuffle_epi32(tmp2, _MM_SHUFFLE(2, 3, 0, 1)));
}
static inline __m256 broadcast_min(const __m256 a) {
  auto tmp = _mm256_min_ps(a, _mm256_permute2f128_ps(a, a, 1));
  auto tmp2 =
      _mm256_min_ps(tmp, _mm256_shuffle_ps(tmp, tmp, _MM_SHUFFLE(1, 0, 3, 2)));
  return _mm256_min_ps(tmp2,
                       _mm256_shuffle_ps(tmp2, tmp2, _MM_SHUFFLE(2, 3, 0, 1)));
}
static inline __m256 broadcast_max(const __m256 a) {
  auto tmp = _mm256_max_ps(a, _mm256_permute2f128_ps(a, a, 1));
  auto tmp2 =
      _mm256_max_ps(tmp, _mm256_shuffle_ps(tmp, tmp, _MM_SHUFFLE(1, 0, 3, 2)));
  return _mm256_max_ps(tmp2,
                       _mm256_shuffle_ps(tmp2, tmp2, _MM_SHUFFLE(2, 3, 0, 1)));
}

static inline __m256i packed_epu16_to_unpacked_epu8(const __m256i &a,
                                                    const __m256i &b) {
  // indices to undo packus_epi32 followed by packus_epi16, once we've
  // swapped 64-bit blocks 1 and 2
  static const __m256i shuffle_idxs = _mm256_set_epi8(
      31 - 16, 30 - 16, 29 - 16, 28 - 16, 23 - 16, 22 - 16, 21 - 16, 20 - 16,
      27 - 16, 26 - 16, 25 - 16, 24 - 16, 19 - 16, 18 - 16, 17 - 16, 16 - 16,
      15 - 0, 14 - 0, 13 - 0, 12 - 0, 7 - 0, 6 - 0, 5 - 0, 4 - 0, 11 - 0,
      10 - 0, 9 - 0, 8 - 0, 3 - 0, 2 - 0, 1 - 0, 0 - 0);

  auto dists_uint8 = _mm256_packus_epi16(a, b);
  // undo the weird shuffling caused by the pack operations
  auto dists_perm =
      _mm256_permute4x64_epi64(dists_uint8, _MM_SHUFFLE(3, 1, 2, 0));
  return _mm256_shuffle_epi8(dists_perm, shuffle_idxs);
}

// based on https://stackoverflow.com/a/51779212/1153180
template <bool Signed = true, bool SameOrder = true>
static inline __m256i pack_ps_epi8_or_epu8(const __m256 &x0, const __m256 &x1,
                                           const __m256 &x2, const __m256 &x3) {
  __m256i a = _mm256_cvtps_epi32(x0);
  __m256i b = _mm256_cvtps_epi32(x1);
  __m256i c = _mm256_cvtps_epi32(x2);
  __m256i d = _mm256_cvtps_epi32(x3);
  __m256i ab = _mm256_packs_epi32(a, b);
  __m256i cd = _mm256_packs_epi32(c, d);
  __m256i abcd = _mm256_undefined_si256();
  if (Signed) {
    abcd = _mm256_packs_epi16(ab, cd);
  } else {
    abcd = _mm256_packus_epi16(ab, cd);
  }
  // packed to one vector, but in [ a_lo, b_lo, c_lo, d_lo | a_hi, b_hi, c_hi,
  // d_hi ] order if you can deal with that in-memory format (e.g. for later
  // in-lane unpack), great, you're done

  if (!SameOrder) {
    return abcd;
  }

  // but if you need sequential order, then vpermd:
  __m256i lanefix = _mm256_permutevar8x32_epi32(
      abcd, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
  return lanefix;
}

static inline __m256i invert_pack_ps0to255_epu8_perm(const __m256i x) {
  const __m256i shuffle_idxs =
      _mm256_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 0,
                       4, 8, 12, // 2nd half is same as first half
                       1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
  const __m256i vperm_idxs = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
  auto grouped = _mm256_shuffle_epi8(x, shuffle_idxs);
  return _mm256_permutevar8x32_epi32(grouped, vperm_idxs);
}

// this is just here to reduce p5 pressure for the case that the floats
// are known to be in the range [0, 255]; note that this will yield garbage if
// this is not the case
// TODO actually test this function
template <bool Signed = false, bool SameOrder = true>
static inline __m256i
pack_ps0to255_epi8_or_epu8(const __m256 &x0, const __m256 &x1, const __m256 &x2,
                           const __m256 &x3) {
  auto a = _mm256_cvtps_epi32(x0);
  auto b = _mm256_cvtps_epi32(x1);
  auto c = _mm256_cvtps_epi32(x2);
  auto d = _mm256_cvtps_epi32(x3);
  b = _mm256_slli_epi32(b, 8);
  c = _mm256_slli_epi32(c, 16);
  d = _mm256_slli_epi32(d, 24);
  auto ab = _mm256_or_si256(a, b);
  auto cd = _mm256_or_si256(c, d);
  auto abcd = _mm256_or_si256(ab, cd);

  if (Signed) {
    abcd = _mm256_add_epi8(abcd, _mm256_set1_epi8(-128));
  }

  if (!SameOrder) {
    return abcd;
  }
  return invert_pack_ps0to255_epu8_perm(abcd);
}

template <bool Signed = true, bool SameOrder = true>
static inline __m256i load_4xf32_as_32xepi8_or_epu8(const float *x,
                                                    const __m256 &scales,
                                                    const __m256 &offsets) {
  auto x0 = fma(_mm256_loadu_ps(x), scales, offsets);
  auto x1 = fma(_mm256_loadu_ps(x + 8), scales, offsets);
  auto x2 = fma(_mm256_loadu_ps(x + 16), scales, offsets);
  auto x3 = fma(_mm256_loadu_ps(x + 24), scales, offsets);
  return pack_ps_epi8_or_epu8<Signed, SameOrder>(x0, x1, x2, x3);
}

template <bool Signed = true, bool SameOrder = true>
static inline __m256i load_4xf32_as_32xepi8_or_epu8(const float *x,
                                                    const __m256 &scales) {
  auto x0 = _mm256_mul_ps(_mm256_loadu_ps(x), scales);
  auto x1 = _mm256_mul_ps(_mm256_loadu_ps(x + 8), scales);
  auto x2 = _mm256_mul_ps(_mm256_loadu_ps(x + 16), scales);
  auto x3 = _mm256_mul_ps(_mm256_loadu_ps(x + 24), scales);
  return pack_ps_epi8_or_epu8<Signed, SameOrder>(x0, x1, x2, x3);
}

// ================================================================ popcnt gemm

// matmul with xor + popcount; inputs are matrices of bits, but pointers
// are uint64s to make loading them up for popcnt easier; note that A is
// assumed to be rowmajor while B is assumed to be colmajor; out is also
// colmajor
// template<int NReadCols=2, int NWriteCols=2, class OutT>
template <int InRowTileSz = 1, int OutColTileSz = 1, int StaticD = -1,
          class OutT>
void _bgemm(
    const uint64_t *A, const uint64_t *B, int N, int D, int M, OutT *out,
    bool add_to_output = false, int A_row_stride = -1,
    // int B_col_stride=-1, int out_col_stride=-1, int nrows_per_chunk=512)
    int B_col_stride = -1, int out_col_stride = -1, int nrows_per_chunk = -1)
// int B_col_stride=-1, int out_col_stride=-1, int nrows_per_chunk=600 * 100)
// int B_col_stride=-1, int out_col_stride=-1, int nrows_per_chunk=1000*1000)
{
  using dtype = uint64_t;
  using packet_t = uint64_t;
  static const int packet_sz = 1; // 1 uint8 is what popcnt operates on
  D = StaticD > 0 ? StaticD : D;  // allow specifying D at compile time
  if (MIN(N, MIN(D, M)) < 1) {
    return;
  } // nothing to do
  assert(MIN(nrows_per_chunk, N) % InRowTileSz == 0);
  assert(M % OutColTileSz == 0);

  static constexpr int target_chunk_nbytes = 24 * 1024; // most of L1
  int A_row_nbytes = D * sizeof(A[0]);
  int l1_cache_nrows = target_chunk_nbytes / A_row_nbytes;
  nrows_per_chunk =
      nrows_per_chunk > InRowTileSz ? nrows_per_chunk : l1_cache_nrows;

  // stuff for tiling nrows
  int nchunks_N = (N + nrows_per_chunk - 1) / nrows_per_chunk;
  auto N_orig = N;
  N = N < nrows_per_chunk ? N : nrows_per_chunk; // *after* setting strides
  auto A_orig = A;
  auto out_orig = out;
  // A_col_stride = A_col_stride     >= 1 ? A_col_stride   : N_orig;
  A_row_stride = A_row_stride >= 1 ? A_row_stride : D;
  B_col_stride = B_col_stride >= 1 ? B_col_stride : D;
  out_col_stride = out_col_stride >= 1 ? out_col_stride : N_orig;

  // costants derived from matrix / tiling sizes
  // int nstripes_D = D / NReadCols;
  int nstripes_M = M / OutColTileSz;

  // arrays that will all get unrolled and not really exist
  const dtype *a_row_ptrs[InRowTileSz];
  const dtype *b_col_starts[OutColTileSz];
  const dtype *b_col_ptrs[OutColTileSz];
  OutT *out_col_starts[OutColTileSz];
  OutT *out_col_ptrs[OutColTileSz];
  dtype a_vals[InRowTileSz];
  dtype b_vals[OutColTileSz];
  OutT accumulators[InRowTileSz][OutColTileSz];

  // printf("nchunks_N, nstripes_M = %d, %d\n", nchunks_N, nstripes_M);

  for (int chunk = 0; chunk < nchunks_N;
       chunk++) { // for each chunk of input rows
    A = A_orig + (chunk * nrows_per_chunk * A_row_stride);
    out = out_orig + (chunk * nrows_per_chunk);
    if (chunk == (nchunks_N - 1)) { // handle last chunk
      auto N_done_so_far = chunk * nrows_per_chunk;
      N = N_orig - N_done_so_far;
      assert(N % InRowTileSz == 0);
    }
    int nstripes_N = N / InRowTileSz;
    // printf("got to chunk %d / %d\n", chunk, nchunks_N - 1);

    for (int m = 0; m < nstripes_M; m++) { // for each group of output cols
      // set output col start ptrs and current ptrs for simplicity
      for (int mm = 0; mm < OutColTileSz; mm++) {
        auto out_col = (m * OutColTileSz) + mm;
        b_col_starts[mm] = B + (B_col_stride * out_col);
        // out_col_starts[mm] = out + (out_col_stride * out_col);
        out_col_ptrs[mm] = out + (out_col_stride * out_col);

        if (!add_to_output) { // zero this block of output buffer
          for (int i = 0; i < N; i++) {
            // out_col_starts[mm][i] = 0;
            out_col_ptrs[mm][i] = 0;
          }
        }
      }
      // for each group of input cols
      for (int n = 0; n < nstripes_N; n++) {
        // reset ptrs to start of rows of A
        for (int nn = 0; nn < InRowTileSz; nn++) {
          auto row_idx = (n * InRowTileSz) + nn;
          a_row_ptrs[nn] = A + (row_idx * A_row_stride);
        }
        // reset ptrs to start of cols of B
        for (int mm = 0; mm < OutColTileSz; mm++) {
          b_col_ptrs[mm] = b_col_starts[mm];
        }
        // reset accumulators
        for (int nn = 0; nn < InRowTileSz; nn++) {
          for (int mm = 0; mm < OutColTileSz; mm++) {
            accumulators[nn][mm] = 0;
          }
        }

        // TODO uncomment
        // main loop; dot prods of rows of A with cols of B
        for (int j = 0; j < D; j++) {
          // load up b vals to use
          for (int mm = 0; mm < OutColTileSz; mm++) {
            b_vals[mm] = *(b_col_ptrs[mm]);
            b_col_ptrs[mm] += packet_sz;
          }
          // for each a row, for each b col
          for (int nn = 0; nn < InRowTileSz; nn++) {
            a_vals[nn] = *(a_row_ptrs[nn]);
            a_row_ptrs[nn] += packet_sz;
            for (int mm = 0; mm < OutColTileSz; mm++) {
              auto diffs = a_vals[nn] ^ b_vals[mm];
              // we use inline asm so that it doesn't get
              // compiled to a bunch of shuffle instructions,
              // which defeats the point of the profiling; can
              // switch to popcount() func from bit_ops.hpp to
              // maybe get a speedup
              accumulators[nn][mm] += popcount_u64(diffs);

              // slower than inline asm for unclear reasons
              // accumulators[nn][mm] += __builtin_popcountll(
              //     a_vals[nn] ^ b_vals[mm]);
            }
          }
        }
        // write out accumulator values
        OutT nbits_per_row = D * 8;
        for (int mm = 0; mm < OutColTileSz; mm++) {
          for (int nn = 0; nn < InRowTileSz; nn++) {
            auto nbits_same = nbits_per_row - accumulators[nn][mm];
            *(out_col_ptrs[mm] + nn) = nbits_same;
          }
          out_col_ptrs[mm] += InRowTileSz;
        }
      }
    }
  }
}

template <class OutT>
void bgemm(const uint64_t *A, const uint64_t *B, int N, int D, int M,
           OutT *out) {
  switch (D) {
  case 1:
    _bgemm<1, 1, 1>(A, B, N, -1, M, out);
    break;
  case 2:
    _bgemm<1, 1, 2>(A, B, N, -1, M, out);
    break;
  case 3:
    _bgemm<1, 1, 3>(A, B, N, -1, M, out);
    break;
  case 4:
    _bgemm<1, 1, 4>(A, B, N, -1, M, out);
    break;
  case 8:
    _bgemm<1, 1, 8>(A, B, N, -1, M, out);
    break;
  default:
    assert(false); // should only be passing in D in {1,2,4,8}
    _bgemm<1, 1>(A, B, N, D, M, out);
    break;
  }
}

#endif // __AVX_UTILS_HPP
