#include "llama3_gqa_group_kernels.hpp"
#include "llama3_kernel_common.hpp"

using namespace llama3_detail;

namespace {

inline __attribute__((always_inline)) void softmax_score_read(
    input_stream<float>* score, float scores[2][kKeys]) {
  for (int key_quad = 0; key_quad < kKeys / 4; ++key_quad) {
#pragma chess_prepare_for_pipelining
    const auto reduced =
        aie::mul(readincr_v<8>(score), kScale).template to_vector<float>();
    aie::store_v(scores[0] + key_quad * 4,
                 reduced.template extract<4>(0));
    aie::store_v(scores[1] + key_quad * 4,
                 reduced.template extract<4>(1));
  }
}

inline __attribute__((always_inline)) float softmax_causal_row_max(
    const float* scores, int row) {
  alignas(aie::vector_decl_align) static const float neg_inf_raw[8] = {
      -3.402823466e+38F, -3.402823466e+38F, -3.402823466e+38F, -3.402823466e+38F,
      -3.402823466e+38F, -3.402823466e+38F, -3.402823466e+38F, -3.402823466e+38F};
  alignas(aie::vector_decl_align) static const int32 lane_index[8] = {
      0, 1, 2, 3, 4, 5, 6, 7};
  const auto neg_inf = aie::load_v<8>(neg_inf_raw);
  const auto lanes = aie::load_v<8>(lane_index);

  float row_max = -3.402823466e+38F;

  for (int column = 0; column < kKeys; column += 8) {
#pragma chess_prepare_for_pipelining
    if (column > row) {
      break;
    }
    const auto raw_score = aie::load_v<8>(scores + column);
    if (column + 7 <= row) {
      const float block_max = aie::reduce_max(raw_score);
      row_max = block_max > row_max ? block_max : row_max;
    } else {
      const auto valid = aie::le(lanes, row - column);
      const auto masked = aie::select(neg_inf, raw_score, valid);
      const float block_max = aie::reduce_max(masked);
      row_max = block_max > row_max ? block_max : row_max;
    }
  }
  return row_max;
}

inline __attribute__((always_inline)) float softmax_exp_reduce(
    const float* scores, float* weights, int row, float row_max) {
  alignas(aie::vector_decl_align) static const int32 lane_index[8] = {
      0, 1, 2, 3, 4, 5, 6, 7};
  const auto lanes = aie::load_v<8>(lane_index);
  const auto zero = aie::zeros<float, 8>();
  float row_sum = 0.0f;

  for (int column = 0; column < kKeys; column += 8) {
#pragma chess_prepare_for_pipelining
    if (column > row) {
      aie::store_v(weights + column, zero);
      continue;
    }
    const auto centered = aie::sub(aie::load_v<8>(scores + column), row_max);
    const auto calculated = exp_negative_vector(centered);
    if (column + 7 <= row) {
      row_sum += aie::reduce_add(calculated);
      aie::store_v(weights + column, calculated);
    } else {
      const auto valid = aie::le(lanes, row - column);
      const auto exponent = aie::select(zero, calculated, valid);
      row_sum += aie::reduce_add(exponent);
      aie::store_v(weights + column, exponent);
    }
  }
  return row_sum;
}

inline __attribute__((always_inline)) float softmax_reciprocal(
    float row_sum) {
  float reciprocal = aie::inv(row_sum);
  return reciprocal * (2.0f - row_sum * reciprocal);
}

inline __attribute__((always_inline)) void softmax_normalize(
    float* weights, float reciprocal) {
  for (int column = 0; column < kKeys; column += 8) {
#pragma chess_prepare_for_pipelining
    const auto normalized =
        aie::mul(aie::load_v<8>(weights + column), reciprocal)
            .template to_vector<float>();
    aie::store_v(weights + column, normalized);
  }
}

inline __attribute__((always_inline)) void softmax_stream_output(
    const float weights[2][kKeys], output_stream<float>* probability) {
  for (int key_quad = 0; key_quad < kKeys / 4; ++key_quad) {
#pragma chess_prepare_for_pipelining
    aie::vector<float, 8> tile;
    tile.insert(0, aie::load_v<4>(weights[0] + key_quad * 4));
    tile.insert(1, aie::load_v<4>(weights[1] + key_quad * 4));
    writeincr(probability, tile);
  }
}

void softmax_head(input_stream<float>* score,
                  output_stream<float>* probability) {
  alignas(aie::vector_decl_align) float scores[2][kKeys];
  alignas(aie::vector_decl_align) float weights[2][kKeys];
  for (int row_pair = 0; row_pair < kRows / 2; ++row_pair) {
    softmax_score_read(score, scores);
    float row_reciprocal[2];
    for (int row_in_pair = 0; row_in_pair < 2; ++row_in_pair) {
      const int row = 2 * row_pair + row_in_pair;
      const float row_max =
          softmax_causal_row_max(scores[row_in_pair], row);
      const float row_sum = softmax_exp_reduce(
          scores[row_in_pair], weights[row_in_pair], row, row_max);
      row_reciprocal[row_in_pair] = softmax_reciprocal(row_sum);
      softmax_normalize(weights[row_in_pair],
                        row_reciprocal[row_in_pair]);
    }
    softmax_stream_output(weights, probability);
  }
}

}  // namespace

void llama3_fused_softmax_2lane(
    input_stream<float>* score_lane0, input_stream<float>* score_lane1,
    output_stream<float>* probability) {
  softmax_head(score_lane0, probability);
  softmax_head(score_lane1, probability);
}
