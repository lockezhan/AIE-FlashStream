#include "llama3_gqa_group_kernels.hpp"
#include "llama3_kernel_common.hpp"
#ifdef __X86SIM__
#include <cstdio>
#endif

using namespace llama3_detail;

namespace value_detail {

template <int ValueD>
inline __attribute__((always_inline)) void pv_value_load(
    input_stream<uint32>* value, int16* value_cache) {
  constexpr int kDimQuads = ValueD / 4;
  for (int tile = 0; tile < (kKeys / 4) * kDimQuads; ++tile) {
    for (int word = 0; word < 8; ++word) {
#pragma chess_prepare_for_pipelining
      const uint32 packed = readincr(value);
      value_cache[tile * 16 + 2 * word] = static_cast<int16>(packed);
      value_cache[tile * 16 + 2 * word + 1] =
          static_cast<int16>(packed >> 16);
    }
  }
}

inline __attribute__((always_inline)) void pv_probability_read(
    input_stream<float>* probability, float* probabilities) {
  for (int key_quad = 0; key_quad < kKeys / 4; ++key_quad) {
#pragma chess_prepare_for_pipelining
    aie::store_v(probabilities + key_quad * 8,
                 readincr_v<8>(probability));
  }
}

template <int ValueD>
inline __attribute__((always_inline)) void pv_mmul_accumulate(
    const float* probabilities, const int16* value_cache,
    float output_pair[2][ValueD], int row_pair) {
  alignas(aie::vector_decl_align) float output_block[8];
  using Mmul = aie::mmul<2, 4, 4, float, float>;
  constexpr int kDimQuads = ValueD / 4;
  const int last_valid_quad = (2 * row_pair + 1) / 4;
  for (int dim_quad = 0; dim_quad < kDimQuads; ++dim_quad) {
    Mmul accumulator;
    for (int key_quad = 0; key_quad <= last_valid_quad; ++key_quad) {
#pragma chess_prepare_for_pipelining
      const auto probability_block =
          aie::load_v<8>(probabilities + key_quad * 8);
      const auto value_bf16 = aie::load_v<16>(
          value_cache + (key_quad * kDimQuads + dim_quad) * 16);
      const auto value_block = bf16_bits_to_float(value_bf16);
      if (key_quad == 0)
        accumulator.mul(probability_block, value_block);
      else
        accumulator.mac(probability_block, value_block);
    }
    aie::store_v(output_block, accumulator.template to_vector<float>());
    aie::store_v(output_pair[0] + dim_quad * 4,
                 aie::load_v<4>(output_block));
    aie::store_v(output_pair[1] + dim_quad * 4,
                 aie::load_v<4>(output_block + 4));
  }
}

template <int ValueD>
inline __attribute__((always_inline)) void pv_fp32_to_bf16_vector_write(
    const float output_pair[2][ValueD], int16*& output_ptr) {
  alignas(aie::vector_decl_align) float tail_values[8];
  alignas(aie::vector_decl_align) int16 tail_packed[8];
  for (int row = 0; row < 2; ++row) {
    int vector_begin = 0;
    if (ValueD == 28 && row == 1) {
      for (int lane = 0; lane < 4; ++lane) {
#pragma chess_unroll_loop
        tail_values[lane] = output_pair[row][lane];
        tail_values[lane + 4] = 0.0f;
      }
      const auto values = aie::load_v<8>(tail_values);
      const auto bits = aie::vector_cast<int32>(values);
      const auto tie = aie::bit_and(1, aie::downshift(bits, 16));
      const auto rounded = aie::add(aie::add(bits, 0x7fff), tie);
      const auto upper = aie::downshift(rounded, 16);
      const auto packed = aie::filter_even(aie::vector_cast<int16>(upper));
      aie::store_v(tail_packed, packed);
      for (int lane = 0; lane < 4; ++lane) {
#pragma chess_unroll_loop
        output_ptr[lane] = tail_packed[lane];
      }
      output_ptr += 4;
      vector_begin = 4;
    }
    int dim = vector_begin;
    for (; dim + 8 <= ValueD; dim += 8) {
#pragma chess_prepare_for_pipelining
      const auto values = aie::load_v<8>(output_pair[row] + dim);
      const auto bits = aie::vector_cast<int32>(values);
      const auto tie = aie::bit_and(1, aie::downshift(bits, 16));
      const auto rounded = aie::add(aie::add(bits, 0x7fff), tie);
      const auto upper = aie::downshift(rounded, 16);
      const auto packed = aie::filter_even(aie::vector_cast<int16>(upper));
      aie::store_v(output_ptr, packed);
      output_ptr += 8;
    }
    if (dim < ValueD) {
      for (int lane = 0; lane < 4; ++lane) {
#pragma chess_unroll_loop
        tail_values[lane] = output_pair[row][dim + lane];
        tail_values[lane + 4] = 0.0f;
      }
      const auto values = aie::load_v<8>(tail_values);
      const auto bits = aie::vector_cast<int32>(values);
      const auto tie = aie::bit_and(1, aie::downshift(bits, 16));
      const auto rounded = aie::add(aie::add(bits, 0x7fff), tie);
      const auto upper = aie::downshift(rounded, 16);
      const auto packed = aie::filter_even(aie::vector_cast<int16>(upper));
      aie::store_v(tail_packed, packed);
      for (int lane = 0; lane < 4; ++lane) {
#pragma chess_unroll_loop
        output_ptr[lane] = tail_packed[lane];
      }
      output_ptr += 4;
    }
  }
}

template <int ValueD, int Instance>
void llama3_value_kernel_impl(
    input_stream<float>* probability,
    input_stream<uint32>* value, output_window<uint16>* output) {
#ifdef __X86SIM__
#define LLAMA3_VALUE_TILE_LOCAL thread_local static
#else
#define LLAMA3_VALUE_TILE_LOCAL static
#endif
  alignas(aie::vector_decl_align) LLAMA3_VALUE_TILE_LOCAL int16 chess_storage(DM_bankA)
      value_cache[kKeys * ValueD];
  LLAMA3_VALUE_TILE_LOCAL uint8 phase = 0;
#undef LLAMA3_VALUE_TILE_LOCAL
  alignas(aie::vector_decl_align) float probabilities[8 * 8];
  alignas(aie::vector_decl_align) float output_pair[2][ValueD];
  int16* output_ptr = reinterpret_cast<int16*>(output->ptr);

  if (phase == 0) pv_value_load<ValueD>(value, value_cache);
#ifdef __X86SIM__
  std::printf("V20_VALUE_CACHE instance=%p phase=%u d=%d action=%s\n",
              static_cast<void*>(&phase), static_cast<unsigned>(phase), ValueD,
              phase == 0 ? "load" : "reuse-no-read");
#endif

  for (int query = 0; query < kQueriesPerPhase; ++query) {
    for (int row_pair = 0; row_pair < kRows / 2; ++row_pair) {
      pv_probability_read(probability, probabilities);
      pv_mmul_accumulate<ValueD>(probabilities, value_cache, output_pair,
                                     row_pair);
      pv_fp32_to_bf16_vector_write<ValueD>(output_pair, output_ptr);
    }
  }
  phase ^= 1;
}

}  // namespace value_detail

using namespace value_detail;

void llama3_value_d24_lane0_bf16_2q(
    input_stream<float>* probability,
    input_stream<uint32>* value, output_window<uint16>* output) {
  llama3_value_kernel_impl<24, 0>(probability, value, output);
}

void llama3_value_d24_lane1_bf16_2q(
    input_stream<float>* probability,
    input_stream<uint32>* value, output_window<uint16>* output) {
  llama3_value_kernel_impl<24, 1>(probability, value, output);
}

void llama3_value_d16_lane2_bf16_2q(
    input_stream<float>* probability,
    input_stream<uint32>* value, output_window<uint16>* output) {
  llama3_value_kernel_impl<16, 2>(probability, value, output);
}

void llama3_value_d24_lane3_bf16_2q(
    input_stream<float>* probability,
    input_stream<uint32>* value, output_window<uint16>* output) {
  llama3_value_kernel_impl<24, 3>(probability, value, output);
}

void llama3_value_d24_lane4_bf16_2q(
    input_stream<float>* probability,
    input_stream<uint32>* value, output_window<uint16>* output) {
  llama3_value_kernel_impl<24, 4>(probability, value, output);
}

void llama3_value_d16_lane5_bf16_2q(
    input_stream<float>* probability,
    input_stream<uint32>* value, output_window<uint16>* output) {
  llama3_value_kernel_impl<16, 5>(probability, value, output);
}
