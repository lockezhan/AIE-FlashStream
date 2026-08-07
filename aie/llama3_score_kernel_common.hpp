#pragma once

#include "llama3_kernel_common.hpp"

namespace llama3_score_detail {

constexpr int kRows = 32;
constexpr int kKeys = 32;
constexpr int kHeadD = 128;
constexpr int kGroups = 2;
constexpr int kGroupD = 64;
constexpr int kMmulRows = 2;
constexpr int kMmulK = 8;
constexpr int kMmulCols = 8;
constexpr int kDimOctets = kHeadD / kMmulK;
constexpr int kGroupOctets = kGroupD / kMmulK;
constexpr int kKeyOctets = kKeys / kMmulCols;
constexpr float kInvInt8Max = 1.0f / 127.0f;

using llama3_detail::bf16_bits_to_float;

inline void read_bf16_group_max_scales(input_stream<int32>* input,
                                       float* scales) {
  alignas(aie::vector_decl_align) int16 packed[kRows * kGroups];
  for (int word = 0; word < kRows * kGroups / 2; ++word) {
#pragma chess_prepare_for_pipelining
    const uint32 value = static_cast<uint32>(readincr(input));
    packed[2 * word] = static_cast<int16>(value);
    packed[2 * word + 1] = static_cast<int16>(value >> 16);
  }
  for (int block = 0; block < kRows * kGroups / 8; ++block) {
#pragma chess_prepare_for_pipelining
    const auto maximum =
        bf16_bits_to_float(aie::load_v<8>(packed + 8 * block));
    const auto scale =
        aie::mul(maximum, kInvInt8Max).template to_vector<float>();
    aie::store_v(scales + 8 * block, scale);
  }
}

}  // namespace llama3_score_detail
