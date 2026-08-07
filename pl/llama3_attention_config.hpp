#pragma once

#include "ap_int.h"

namespace llama3_attn {

constexpr int kHeadDim = 128;
constexpr int kQueryHeads = 32;
constexpr int kKvHeads = 8;
constexpr int kQueriesPerKvHead = kQueryHeads / kKvHeads;
constexpr int kQueryBlock = 32;
constexpr int kKeyBlock = 64;
constexpr float kAttentionScale = 0.08838834764831843f;  // 1 / sqrt(128)

inline float bf16_to_float(ap_uint<16> value) {
  union {
    unsigned int bits;
    float value;
  } converter;
  converter.bits = static_cast<unsigned int>(value) << 16;
  return converter.value;
}

inline ap_uint<16> float_to_bf16(float value) {
  union {
    unsigned int bits;
    float value;
  } converter;
  converter.value = value;

  // Round-to-nearest-even before dropping the lower FP32 mantissa bits.
  const unsigned int lsb = (converter.bits >> 16) & 1U;
  converter.bits += 0x7fffU + lsb;
  return ap_uint<16>(converter.bits >> 16);
}

inline unsigned long long q_offset(int batch, int head, int token, int dim,
                                   int seq_len) {
  return (((static_cast<unsigned long long>(batch) * kQueryHeads + head) *
           seq_len + token) * kHeadDim + dim);
}

inline unsigned long long kv_offset(int batch, int head, int token, int dim,
                                    int seq_len) {
  return (((static_cast<unsigned long long>(batch) * kKvHeads + head) *
           seq_len + token) * kHeadDim + dim);
}

}  // namespace llama3_attn
