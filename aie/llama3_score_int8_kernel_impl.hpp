#pragma once

#include "llama3_gqa_group_kernels.hpp"
#include "llama3_score_kernel_common.hpp"

using namespace llama3_score_detail;

template <int Lane>
void llama3_score_int8_impl(input_stream<int32>* q,
                            input_stream<int32>* k,
                            output_stream<float>* score) {
  using Mmul = aie::mmul<kMmulRows, kMmulK, kMmulCols, int8, int8, acc48>;

  alignas(aie::vector_decl_align) static int8 chess_storage(DM_bankA)
      k_cache_lo[kKeyOctets * kGroupOctets * kMmulK * kMmulCols];
  alignas(aie::vector_decl_align) static int8 chess_storage(DM_bankC)
      k_cache_hi[kKeyOctets * kGroupOctets * kMmulK * kMmulCols];
  alignas(aie::vector_decl_align) static float
      k_scales[kKeys * kGroups];
  static uint8 phase = 0;

  alignas(aie::vector_decl_align) int8 q_cache[kMmulRows * kHeadD];
  alignas(aie::vector_decl_align) float q_scales[kRows * kGroups];

  if (phase == 0) {
    read_bf16_group_max_scales(k, k_scales);
    for (int key_octet = 0; key_octet < kKeyOctets; ++key_octet) {
      for (int dim_octet = 0; dim_octet < kDimOctets; ++dim_octet) {
#pragma chess_prepare_for_pipelining
        const auto packed_lo = aie::vector_cast<int8>(readincr_v<8>(k));
        const auto packed_hi = aie::vector_cast<int8>(readincr_v<8>(k));
        const auto values = aie::concat(packed_lo, packed_hi);
        const int local_dim_octet = dim_octet & (kGroupOctets - 1);
        const int offset =
            (key_octet * kGroupOctets + local_dim_octet) * 64;
        if (dim_octet < kGroupOctets)
          aie::store_v(k_cache_lo + offset, values);
        else
          aie::store_v(k_cache_hi + offset, values);
      }
    }
  }

  read_bf16_group_max_scales(q, q_scales);
  for (int row_pair = 0; row_pair < kRows / kMmulRows; ++row_pair) {
    for (int dim_pair = 0; dim_pair < kDimOctets / 2; ++dim_pair) {
#pragma chess_prepare_for_pipelining
      aie::store_v(q_cache + dim_pair * 32,
                   aie::vector_cast<int8>(readincr_v<8>(q)));
    }

    const auto q_scale_lo = aie::concat(
        aie::broadcast<float, 8>(q_scales[4 * row_pair]),
        aie::broadcast<float, 8>(q_scales[4 * row_pair + 2]));
    const auto q_scale_hi = aie::concat(
        aie::broadcast<float, 8>(q_scales[4 * row_pair + 1]),
        aie::broadcast<float, 8>(q_scales[4 * row_pair + 3]));
    const int last_valid_octet = (2 * row_pair + 1) / kMmulCols;
    for (int key_octet = 0; key_octet < kKeyOctets; ++key_octet) {
      auto dequantized = aie::zeros<float, 16>();
      if (key_octet <= last_valid_octet) {
        const int key_base = key_octet * kGroupOctets * 64;
        Mmul accumulator_lo;
        accumulator_lo.mul(aie::load_v<16>(q_cache),
                           aie::load_v<64>(k_cache_lo + key_base));
        for (int dim_octet = 1; dim_octet < kGroupOctets; ++dim_octet) {
#pragma chess_prepare_for_pipelining
          accumulator_lo.mac(
              aie::load_v<16>(q_cache + dim_octet * 16),
              aie::load_v<64>(k_cache_lo + key_base + dim_octet * 64));
        }

        Mmul accumulator_hi;
        accumulator_hi.mul(
            aie::load_v<16>(q_cache + kGroupOctets * 16),
            aie::load_v<64>(k_cache_hi + key_base));
        for (int dim_octet = 1; dim_octet < kGroupOctets; ++dim_octet) {
#pragma chess_prepare_for_pipelining
          accumulator_hi.mac(
              aie::load_v<16>(q_cache + (kGroupOctets + dim_octet) * 16),
              aie::load_v<64>(k_cache_hi + key_base + dim_octet * 64));
        }

        const auto k_scale_interleaved =
            aie::load_v<16>(k_scales + 16 * key_octet);
        const auto k_scale_lo_8 = aie::filter_even(k_scale_interleaved);
        const auto k_scale_hi_8 = aie::filter_odd(k_scale_interleaved);
        const auto k_scale_lo = aie::concat(k_scale_lo_8, k_scale_lo_8);
        const auto k_scale_hi = aie::concat(k_scale_hi_8, k_scale_hi_8);

        auto score_lo = aie::to_float(
            accumulator_lo.template to_vector<int32>(0));
        score_lo = aie::mul(score_lo, k_scale_lo).template to_vector<float>();
        score_lo = aie::mul(score_lo, q_scale_lo).template to_vector<float>();
        auto score_hi = aie::to_float(
            accumulator_hi.template to_vector<int32>(0));
        score_hi = aie::mul(score_hi, k_scale_hi).template to_vector<float>();
        score_hi = aie::mul(score_hi, q_scale_hi).template to_vector<float>();
        dequantized = aie::add(score_lo, score_hi);
      }

      writeincr(score,
                aie::concat(dequantized.template extract<4>(0),
                            dequantized.template extract<4>(2)));
      writeincr(score,
                aie::concat(dequantized.template extract<4>(1),
                            dequantized.template extract<4>(3)));
    }
  }
  phase ^= 1;
}
