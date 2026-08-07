#pragma once

#include <aie_api/aie.hpp>
#include <aie_api/operators.hpp>
#include <stdint.h>

#include "llama3_runtime_stubs.hpp"

using namespace aie::operators;

namespace llama3_detail {

constexpr int kQueriesPerPhase = 2;
constexpr int kRows = 32;
constexpr int kKeys = 32;
constexpr int kMmulRows = 2;
constexpr int kMmulK = 4;
constexpr int kMmulCols = 4;
constexpr int kValueD = 32;
constexpr float kScale = 0.08838834764831843f;

inline aie::vector<float, 8> bf16_bits_to_float(
    const aie::vector<int16, 8>& values) {
  const aie::accum<acc32, 8> wide(values, 16);
  return aie::vector_cast<float>(wide.template to_vector<int32>());
}

inline aie::vector<float, 16> bf16_bits_to_float(
    const aie::vector<int16, 16>& values) {
  const aie::accum<acc32, 16> wide(values, 16);
  return aie::vector_cast<float>(wide.template to_vector<int32>());
}

inline void unpack_four_words(input_stream<uint32>* input, float* output) {
  alignas(aie::vector_decl_align) int16 packed[8];
  for (int word = 0; word < 4; ++word) {
    const uint32 value = readincr(input);
    packed[2 * word] = static_cast<int16>(value);
    packed[2 * word + 1] = static_cast<int16>(value >> 16);
  }
  aie::store_v(output, bf16_bits_to_float(aie::load_v<8>(packed)));
}

inline aie::vector<float, 8> exp_negative_vector(
    const aie::vector<float, 8>& input) {
  constexpr float kLog2E = 1.4426950408889634f;
  constexpr float kLn2 = 0.6931471805599453f;
  auto x = aie::max(input, -16.0f);
  const auto scaled = aie::mul(x, kLog2E).template to_vector<float>();
  const auto exponent =
      aie::to_fixed<int32>(aie::sub(scaled, 0.999999f));
  const auto exponent_f = aie::to_float(exponent);
  const auto remainder = aie::sub(
      x, aie::mul(exponent_f, kLn2).template to_vector<float>());
  auto polynomial = aie::broadcast<float, 8>(0.0083333338f);
  polynomial = aie::add(
      aie::mul(polynomial, remainder).template to_vector<float>(),
      0.0416666679f);
  polynomial = aie::add(
      aie::mul(polynomial, remainder).template to_vector<float>(),
      0.1666666716f);
  polynomial = aie::add(
      aie::mul(polynomial, remainder).template to_vector<float>(), 0.5f);
  polynomial = aie::add(
      aie::mul(polynomial, remainder).template to_vector<float>(), 1.0f);
  polynomial = aie::add(
      aie::mul(polynomial, remainder).template to_vector<float>(), 1.0f);
  const auto power_bits = aie::upshift(aie::add(exponent, 127), 23);
  return aie::mul(polynomial, aie::vector_cast<float>(power_bits))
      .template to_vector<float>();
}

inline uint16 float_to_bf16(float value) {
  union {
    float value;
    uint32 bits;
  } converter;
  converter.value = value;
  converter.bits += 0x7fffU + ((converter.bits >> 16) & 1U);
  return static_cast<uint16>(converter.bits >> 16);
}

}  // namespace llama3_detail
