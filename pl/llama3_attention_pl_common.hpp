#pragma once

#include "ap_int.h"
#include "ap_axi_sdata.h"
#include "hls_stream.h"
#include "llama3_attention_config.hpp"
#include "llama3_stream_protocol.hpp"

namespace llama3_attn {

using DdrWord = ap_uint<512>;
using DdrStream = hls::stream<DdrWord>;
using RawStream = hls::stream<ap_uint<128>>;
using AxisWord = ap_axiu<128, 0, 0, 0>;
using AxisStream = hls::stream<AxisWord>;
using ScaleStream = hls::stream<ap_uint<16>>;

constexpr int kCompactSequence = 32;
constexpr unsigned long long kQWordsPerBatch =
    static_cast<unsigned long long>(kQueryHeads) * kCompactSequence * 4;
constexpr unsigned long long kKvWordsPerBatch =
    static_cast<unsigned long long>(kKvHeads) * kCompactSequence * 4;

constexpr unsigned kQuantPacketId[kKvHeads] = {0, 1, 2, 3, 4, 5, 6, 7};
constexpr int kQuantRows = 32;
constexpr int kQuantHeadD = 128;
constexpr int kQuantGroups = 2;
constexpr int kQuantGroupD = 64;
constexpr int kQuantWordsPerHead = 256;
constexpr int kScaleWordsPerHead = 8;
constexpr int kPacketWordsPerHead = kScaleWordsPerHead + kQuantWordsPerHead;

static const ap_uint<16> kReciprocalMantissaQ15[128] = {
    32512, 32260, 32012, 31767, 31527, 31290, 31056, 30826,
    30600, 30376, 30156, 29939, 29725, 29514, 29307, 29102,
    28900, 28700, 28504, 28310, 28118, 27930, 27744, 27560,
    27379, 27200, 27023, 26849, 26677, 26507, 26339, 26173,
    26010, 25848, 25688, 25531, 25375, 25221, 25069, 24919,
    24771, 24624, 24480, 24336, 24195, 24055, 23917, 23780,
    23645, 23512, 23379, 23249, 23120, 22992, 22866, 22741,
    22617, 22495, 22374, 22254, 22136, 22019, 21903, 21788,
    21675, 21562, 21451, 21341, 21232, 21125, 21018, 20912,
    20808, 20704, 20602, 20500, 20400, 20300, 20202, 20104,
    20007, 19912, 19817, 19723, 19630, 19538, 19446, 19356,
    19266, 19178, 19090, 19002, 18916, 18830, 18746, 18662,
    18578, 18496, 18414, 18333, 18252, 18173, 18094, 18015,
    17938, 17861, 17784, 17709, 17634, 17559, 17485, 17412, 17340,
    17268, 17196, 17126, 17055, 16986, 16917, 16848, 16780,
    16713, 16646, 16580, 16514, 16449, 16384, 16320};

inline ap_uint<64> select_bf16_quad(const DdrWord& word, int quad) {
#pragma HLS INLINE
  switch (quad) {
    case 0: return word.range(63, 0);
    case 1: return word.range(127, 64);
    case 2: return word.range(191, 128);
    case 3: return word.range(255, 192);
    case 4: return word.range(319, 256);
    case 5: return word.range(383, 320);
    case 6: return word.range(447, 384);
    default: return word.range(511, 448);
  }
}

inline ap_uint<64> select_global_bf16_quad(const DdrWord words[4],
                                           int global_quad) {
#pragma HLS INLINE
  switch (global_quad >> 3) {
    case 0: return select_bf16_quad(words[0], global_quad & 7);
    case 1: return select_bf16_quad(words[1], global_quad & 7);
    case 2: return select_bf16_quad(words[2], global_quad & 7);
    default: return select_bf16_quad(words[3], global_quad & 7);
  }
}

inline ap_uint<15> bf16_magnitude(ap_uint<16> value) {
#pragma HLS INLINE
  return value.range(14, 0);
}

inline ap_uint<8> bf16_exponent(ap_uint<16> value) {
#pragma HLS INLINE
  return value.range(14, 7);
}

inline ap_uint<8> bf16_mantissa(ap_uint<16> value) {
#pragma HLS INLINE
  ap_uint<8> mantissa = 0;
  mantissa[7] = 1;
  mantissa.range(6, 0) = value.range(6, 0);
  return mantissa;
}

inline ap_int<8> quantize_lut(ap_uint<16> value, ap_uint<16> row_max) {
#pragma HLS INLINE
  const ap_uint<15> magnitude = bf16_magnitude(value);
  const ap_uint<8> value_exp = bf16_exponent(value);
  const ap_uint<8> max_exp = bf16_exponent(row_max);
  if (magnitude == 0 || value_exp == 0 || max_exp == 0) return 0;

  const ap_uint<8> mantissa = bf16_mantissa(value);
  const ap_uint<8> max_mantissa = bf16_mantissa(row_max);
  const ap_uint<16> reciprocal =
      kReciprocalMantissaQ15[max_mantissa.range(6, 0)];
  const ap_uint<24> product = mantissa * reciprocal;
  const ap_uint<9> exponent_delta = max_exp - value_exp;
  const ap_uint<9> shift = 15 + exponent_delta;
  ap_uint<9> quantized = 0;
  if (shift < 24) {
    const ap_uint<24> rounding = ap_uint<24>(1) << (shift - 1);
    quantized = (product + rounding) >> shift;
  }
  if (quantized > 127) quantized = 127;
  ap_int<9> signed_value = ap_int<9>(quantized);
  if (value[15]) signed_value = -signed_value;
  return ap_int<8>(signed_value);
}

inline ap_uint<16> max_abs_bf16_halfword(ap_uint<256> packed) {
#pragma HLS INLINE
  ap_uint<16> maximum = 0;
  for (int lane = 0; lane < 16; ++lane) {
#pragma HLS UNROLL
    const ap_uint<16> raw = packed.range(16 * lane + 15, 16 * lane);
    const ap_uint<16> magnitude = raw & ap_uint<16>(0x7FFF);
    if (magnitude > maximum) {
      maximum = magnitude;
    }
  }
  return maximum;
}

inline void read_q_ddr(const DdrWord* q, int batch, DdrStream lane_words[2]) {
#pragma HLS INLINE off
  const unsigned long long batch_base =
      static_cast<unsigned long long>(batch) * kQWordsPerBatch;
  // Wave-major schedule: Q0 across all groups, then Q1, Q2, and Q3.
  // Each physical lane owns four groups, not one query of every group.
  for (int q_in_group = 0; q_in_group < kQueriesPerKvHead; ++q_in_group) {
    for (int group = 0; group < kKvHeads / 2; ++group) {
      const int q_head = group * kQueriesPerKvHead + q_in_group;
      for (int row = 0; row < kQuantRows; ++row) {
        const unsigned long long row_base =
            batch_base +
            static_cast<unsigned long long>(q_head * kQuantRows + row) * 4;
        for (int word = 0; word < 4; ++word) {
#pragma HLS PIPELINE II=1
          lane_words[0].write(q[row_base + word]);
        }
      }
    }
    for (int group = kKvHeads / 2; group < kKvHeads; ++group) {
      const int q_head = group * kQueriesPerKvHead + q_in_group;
      for (int row = 0; row < kQuantRows; ++row) {
        const unsigned long long row_base =
            batch_base +
            static_cast<unsigned long long>(q_head * kQuantRows + row) * 4;
        for (int word = 0; word < 4; ++word) {
#pragma HLS PIPELINE II=1
          lane_words[1].write(q[row_base + word]);
        }
      }
    }
  }
}

inline void read_k_ddr(const DdrWord* k, int batch, DdrStream& words) {
#pragma HLS INLINE off
  const unsigned long long batch_base =
      static_cast<unsigned long long>(batch) * kKvWordsPerBatch;
  for (int group = 0; group < kKvHeads; ++group) {
    for (int row = 0; row < kQuantRows; ++row) {
      const unsigned long long row_base =
          batch_base +
          static_cast<unsigned long long>(group * kQuantRows + row) * 4;
      for (int word = 0; word < 4; ++word) {
#pragma HLS PIPELINE II=1
        words.write(k[row_base + word]);
      }
    }
  }
}

inline void quantize_q_lane(DdrStream& words, ScaleStream& scales,
                            RawStream& quantized) {
#pragma HLS INLINE off
  DdrWord row_buffer[2][4];
  ap_uint<16> word_max_buffer_g0[2][4];
  ap_uint<16> word_max_buffer_g1[2][4];
  ap_uint<64> q_pair[2][kQuantHeadD / 8];
#pragma HLS ARRAY_PARTITION variable=row_buffer complete dim=1
#pragma HLS ARRAY_PARTITION variable=row_buffer complete dim=2
#pragma HLS ARRAY_PARTITION variable=word_max_buffer_g0 complete
#pragma HLS ARRAY_PARTITION variable=word_max_buffer_g1 complete
#pragma HLS ARRAY_PARTITION variable=q_pair complete dim=1
#pragma HLS BIND_STORAGE variable=q_pair type=ram_2p impl=bram

  for (int head = 0; head < 2 * kKvHeads; ++head) {
    ap_uint<16> row_max[2][kQuantGroups] = {{0, 0}, {0, 0}};
#pragma HLS ARRAY_PARTITION variable=row_max complete
    for (int word = 0; word < 4; ++word) {
#pragma HLS PIPELINE II=1
      const DdrWord packed = words.read();
      row_buffer[0][word] = packed;
      word_max_buffer_g0[0][word] = max_abs_bf16_halfword(packed.range(255, 0));
      word_max_buffer_g1[0][word] = max_abs_bf16_halfword(packed.range(511, 256));
    }
    row_max[0][0] = word_max_buffer_g0[0][0] > word_max_buffer_g0[0][1]
                        ? (word_max_buffer_g0[0][0] > word_max_buffer_g0[0][2]
                               ? (word_max_buffer_g0[0][0] > word_max_buffer_g0[0][3]
                                      ? word_max_buffer_g0[0][0]
                                      : word_max_buffer_g0[0][3])
                               : (word_max_buffer_g0[0][2] > word_max_buffer_g0[0][3]
                                      ? word_max_buffer_g0[0][2]
                                      : word_max_buffer_g0[0][3]))
                        : (word_max_buffer_g0[0][1] > word_max_buffer_g0[0][2]
                               ? (word_max_buffer_g0[0][1] > word_max_buffer_g0[0][3]
                                      ? word_max_buffer_g0[0][1]
                                      : word_max_buffer_g0[0][3])
                               : (word_max_buffer_g0[0][2] > word_max_buffer_g0[0][3]
                                      ? word_max_buffer_g0[0][2]
                                      : word_max_buffer_g0[0][3]));

    row_max[0][1] = word_max_buffer_g1[0][0] > word_max_buffer_g1[0][1]
                        ? (word_max_buffer_g1[0][0] > word_max_buffer_g1[0][2]
                               ? (word_max_buffer_g1[0][0] > word_max_buffer_g1[0][3]
                                      ? word_max_buffer_g1[0][0]
                                      : word_max_buffer_g1[0][3])
                               : (word_max_buffer_g1[0][2] > word_max_buffer_g1[0][3]
                                      ? word_max_buffer_g1[0][2]
                                      : word_max_buffer_g1[0][3]))
                        : (word_max_buffer_g1[0][1] > word_max_buffer_g1[0][2]
                               ? (word_max_buffer_g1[0][1] > word_max_buffer_g1[0][3]
                                      ? word_max_buffer_g1[0][1]
                                      : word_max_buffer_g1[0][3])
                               : (word_max_buffer_g1[0][2] > word_max_buffer_g1[0][3]
                                      ? word_max_buffer_g1[0][2]
                                      : word_max_buffer_g1[0][3]));

    for (int row = 0; row < kQuantRows; ++row) {
      const int current = row & 1;
      const int next = current ^ 1;
      const bool has_next = row + 1 < kQuantRows;
      scales.write(row_max[current][0]);
      scales.write(row_max[current][1]);

      for (int dim_octet = 0; dim_octet < kQuantHeadD / 8; ++dim_octet) {
#pragma HLS PIPELINE II=1
        const int word_index = dim_octet >> 2;
        const int word_element_base = (dim_octet & 3) * 8;
        const DdrWord packed_row = row_buffer[current][word_index];
        const int group = (dim_octet >> 1) & 1;
        ap_uint<64> packed_octet = 0;
        for (int lane = 0; lane < 8; ++lane) {
#pragma HLS UNROLL
          const int word_element = word_element_base + lane;
          const ap_uint<16> value = packed_row.range(
              16 * word_element + 15, 16 * word_element);
          packed_octet.range(8 * lane + 7, 8 * lane) = ap_uint<8>(
              quantize_lut(value, row_max[current][group]));
        }
        q_pair[current][dim_octet] = packed_octet;

        if (has_next && dim_octet < 4) {
          const DdrWord next_word = words.read();
          row_buffer[next][dim_octet] = next_word;
          word_max_buffer_g0[next][dim_octet] =
              max_abs_bf16_halfword(next_word.range(255, 0));
          word_max_buffer_g1[next][dim_octet] =
              max_abs_bf16_halfword(next_word.range(511, 256));
        }
      }

      if (has_next) {
        row_max[next][0] = word_max_buffer_g0[next][0] > word_max_buffer_g0[next][1]
                               ? (word_max_buffer_g0[next][0] > word_max_buffer_g0[next][2]
                                      ? (word_max_buffer_g0[next][0] > word_max_buffer_g0[next][3]
                                             ? word_max_buffer_g0[next][0]
                                             : word_max_buffer_g0[next][3])
                                      : (word_max_buffer_g0[next][2] > word_max_buffer_g0[next][3]
                                             ? word_max_buffer_g0[next][2]
                                             : word_max_buffer_g0[next][3]))
                               : (word_max_buffer_g0[next][1] > word_max_buffer_g0[next][2]
                                      ? (word_max_buffer_g0[next][1] > word_max_buffer_g0[next][3]
                                             ? word_max_buffer_g0[next][1]
                                             : word_max_buffer_g0[next][3])
                                      : (word_max_buffer_g0[next][2] > word_max_buffer_g0[next][3]
                                             ? word_max_buffer_g0[next][2]
                                             : word_max_buffer_g0[next][3]));

        row_max[next][1] = word_max_buffer_g1[next][0] > word_max_buffer_g1[next][1]
                               ? (word_max_buffer_g1[next][0] > word_max_buffer_g1[next][2]
                                      ? (word_max_buffer_g1[next][0] > word_max_buffer_g1[next][3]
                                             ? word_max_buffer_g1[next][0]
                                             : word_max_buffer_g1[next][3])
                                      : (word_max_buffer_g1[next][2] > word_max_buffer_g1[next][3]
                                             ? word_max_buffer_g1[next][2]
                                             : word_max_buffer_g1[next][3]))
                               : (word_max_buffer_g1[next][1] > word_max_buffer_g1[next][2]
                                      ? (word_max_buffer_g1[next][1] > word_max_buffer_g1[next][3]
                                             ? word_max_buffer_g1[next][1]
                                             : word_max_buffer_g1[next][3])
                                      : (word_max_buffer_g1[next][2] > word_max_buffer_g1[next][3]
                                             ? word_max_buffer_g1[next][2]
                                             : word_max_buffer_g1[next][3]));
      }

      if (current == 1) {
        static const int octet_order[16] = {0, 1, 4, 5, 8, 9, 12, 13,
                                            2, 3, 6, 7, 10, 11, 14, 15};
        for (int idx = 0; idx < kQuantHeadD / 8; ++idx) {
#pragma HLS PIPELINE II=1
          const int dim_octet = octet_order[idx];
          ap_uint<128> packed = 0;
          packed.range(63, 0) = q_pair[0][dim_octet];
          packed.range(127, 64) = q_pair[1][dim_octet];
          quantized.write(packed);
        }
      }
    }
  }
}

inline void quantize_k_once(DdrStream& words, ScaleStream& scales0,
                            ScaleStream& scales1, RawStream& quantized0,
                            RawStream& quantized1) {
#pragma HLS INLINE off
  DdrWord row_buffer[2][4];
  ap_uint<16> word_max_buffer_g0[2][4];
  ap_uint<16> word_max_buffer_g1[2][4];
  ap_uint<64> k_octet[8][kQuantHeadD / 8];
#pragma HLS ARRAY_PARTITION variable=row_buffer complete dim=1
#pragma HLS ARRAY_PARTITION variable=row_buffer complete dim=2
#pragma HLS ARRAY_PARTITION variable=word_max_buffer_g0 complete
#pragma HLS ARRAY_PARTITION variable=word_max_buffer_g1 complete
#pragma HLS ARRAY_PARTITION variable=k_octet complete dim=1

  for (int group_index = 0; group_index < kKvHeads; ++group_index) {
    ap_uint<16> row_max[2][kQuantGroups] = {{0, 0}, {0, 0}};
#pragma HLS ARRAY_PARTITION variable=row_max complete
    for (int word = 0; word < 4; ++word) {
#pragma HLS PIPELINE II=1
      const DdrWord packed = words.read();
      row_buffer[0][word] = packed;
      word_max_buffer_g0[0][word] = max_abs_bf16_halfword(packed.range(255, 0));
      word_max_buffer_g1[0][word] = max_abs_bf16_halfword(packed.range(511, 256));
    }
    row_max[0][0] = word_max_buffer_g0[0][0] > word_max_buffer_g0[0][1]
                        ? (word_max_buffer_g0[0][0] > word_max_buffer_g0[0][2]
                               ? (word_max_buffer_g0[0][0] > word_max_buffer_g0[0][3]
                                      ? word_max_buffer_g0[0][0]
                                      : word_max_buffer_g0[0][3])
                               : (word_max_buffer_g0[0][2] > word_max_buffer_g0[0][3]
                                      ? word_max_buffer_g0[0][2]
                                      : word_max_buffer_g0[0][3]))
                        : (word_max_buffer_g0[0][1] > word_max_buffer_g0[0][2]
                               ? (word_max_buffer_g0[0][1] > word_max_buffer_g0[0][3]
                                      ? word_max_buffer_g0[0][1]
                                      : word_max_buffer_g0[0][3])
                               : (word_max_buffer_g0[0][2] > word_max_buffer_g0[0][3]
                                      ? word_max_buffer_g0[0][2]
                                      : word_max_buffer_g0[0][3]));

    row_max[0][1] = word_max_buffer_g1[0][0] > word_max_buffer_g1[0][1]
                        ? (word_max_buffer_g1[0][0] > word_max_buffer_g1[0][2]
                               ? (word_max_buffer_g1[0][0] > word_max_buffer_g1[0][3]
                                      ? word_max_buffer_g1[0][0]
                                      : word_max_buffer_g1[0][3])
                               : (word_max_buffer_g1[0][2] > word_max_buffer_g1[0][3]
                                      ? word_max_buffer_g1[0][2]
                                      : word_max_buffer_g1[0][3]))
                        : (word_max_buffer_g1[0][1] > word_max_buffer_g1[0][2]
                               ? (word_max_buffer_g1[0][1] > word_max_buffer_g1[0][3]
                                      ? word_max_buffer_g1[0][1]
                                      : word_max_buffer_g1[0][3])
                               : (word_max_buffer_g1[0][2] > word_max_buffer_g1[0][3]
                                      ? word_max_buffer_g1[0][2]
                                      : word_max_buffer_g1[0][3]));

    for (int row = 0; row < kQuantRows; ++row) {
      const int current = row & 1;
      const int next = current ^ 1;
      const bool has_next = row + 1 < kQuantRows;
      for (int group = 0; group < kQuantGroups; ++group) {
#pragma HLS UNROLL
        if (group_index < kKvHeads / 2)
          scales0.write(row_max[current][group]);
        else
          scales1.write(row_max[current][group]);
      }

      for (int dim_octet = 0; dim_octet < kQuantHeadD / 8; ++dim_octet) {
#pragma HLS PIPELINE II=1
        const int word_index = dim_octet >> 2;
        const int word_element_base = (dim_octet & 3) * 8;
        const DdrWord packed_row = row_buffer[current][word_index];
        const int group = (dim_octet >> 1) & 1;
        ap_uint<64> packed_octet = 0;
        for (int lane = 0; lane < 8; ++lane) {
#pragma HLS UNROLL
          const int word_element = word_element_base + lane;
          const ap_uint<16> value = packed_row.range(
              16 * word_element + 15, 16 * word_element);
          packed_octet.range(8 * lane + 7, 8 * lane) = ap_uint<8>(
              quantize_lut(value, row_max[current][group]));
        }
        k_octet[row & 7][dim_octet] = packed_octet;

        if (has_next && dim_octet < 4) {
          const DdrWord next_word = words.read();
          row_buffer[next][dim_octet] = next_word;
          word_max_buffer_g0[next][dim_octet] =
              max_abs_bf16_halfword(next_word.range(255, 0));
          word_max_buffer_g1[next][dim_octet] =
              max_abs_bf16_halfword(next_word.range(511, 256));
        }
      }

      if (has_next) {
        row_max[next][0] = word_max_buffer_g0[next][0] > word_max_buffer_g0[next][1]
                               ? (word_max_buffer_g0[next][0] > word_max_buffer_g0[next][2]
                                      ? (word_max_buffer_g0[next][0] > word_max_buffer_g0[next][3]
                                             ? word_max_buffer_g0[next][0]
                                             : word_max_buffer_g0[next][3])
                                      : (word_max_buffer_g0[next][2] > word_max_buffer_g0[next][3]
                                             ? word_max_buffer_g0[next][2]
                                             : word_max_buffer_g0[next][3]))
                               : (word_max_buffer_g0[next][1] > word_max_buffer_g0[next][2]
                                      ? (word_max_buffer_g0[next][1] > word_max_buffer_g0[next][3]
                                             ? word_max_buffer_g0[next][1]
                                             : word_max_buffer_g0[next][3])
                                      : (word_max_buffer_g0[next][2] > word_max_buffer_g0[next][3]
                                             ? word_max_buffer_g0[next][2]
                                             : word_max_buffer_g0[next][3]));

        row_max[next][1] = word_max_buffer_g1[next][0] > word_max_buffer_g1[next][1]
                               ? (word_max_buffer_g1[next][0] > word_max_buffer_g1[next][2]
                                      ? (word_max_buffer_g1[next][0] > word_max_buffer_g1[next][3]
                                             ? word_max_buffer_g1[next][0]
                                             : word_max_buffer_g1[next][3])
                                      : (word_max_buffer_g1[next][2] > word_max_buffer_g1[next][3]
                                             ? word_max_buffer_g1[next][2]
                                             : word_max_buffer_g1[next][3]))
                               : (word_max_buffer_g1[next][1] > word_max_buffer_g1[next][1]
                                      ? (word_max_buffer_g1[next][1] > word_max_buffer_g1[next][3]
                                             ? word_max_buffer_g1[next][1]
                                             : word_max_buffer_g1[next][3])
                                      : (word_max_buffer_g1[next][2] > word_max_buffer_g1[next][3]
                                             ? word_max_buffer_g1[next][2]
                                             : word_max_buffer_g1[next][3]));
      }

      if ((row & 7) == 7) {
        static const int octet_order[16] = {0, 1, 4, 5, 8, 9, 12, 13,
                                            2, 3, 6, 7, 10, 11, 14, 15};
        for (int idx = 0; idx < kQuantHeadD / 8; ++idx) {
          const int dim_octet = octet_order[idx];
          for (int word = 0; word < 4; ++word) {
#pragma HLS PIPELINE II=1
            ap_uint<128> packed = 0;
            for (int byte = 0; byte < 16; ++byte) {
#pragma HLS UNROLL
              const int linear = 16 * word + byte;
              const int dim = linear >> 3;
              const int key = linear & 7;
              packed.range(8 * byte + 7, 8 * byte) =
                  k_octet[key][dim_octet].range(8 * dim + 7, 8 * dim);
            }
            if (group_index < kKvHeads / 2)
              quantized0.write(packed);
            else
              quantized1.write(packed);
          }
        }
      }
    }
  }
}

inline void assemble_quantized_packets(ScaleStream& scales, RawStream& quantized,
                                       RawStream& payload, int packets) {
#pragma HLS INLINE off
  for (int packet = 0; packet < packets; ++packet) {
    for (int scale_word = 0; scale_word < kScaleWordsPerHead; ++scale_word) {
      ap_uint<128> packed = 0;
      for (int scale = 0; scale < 8; ++scale) {
#pragma HLS PIPELINE II=1
        packed.range(16 * scale + 15, 16 * scale) = scales.read();
      }
      payload.write(packed);
    }
    for (int word = 0; word < kQuantWordsPerHead; ++word) {
#pragma HLS PIPELINE II=1
      payload.write(quantized.read());
    }
  }
}

inline ap_uint<32> packet_header(unsigned id) {
#pragma HLS INLINE
  ap_uint<32> header = 0;
  header.range(4, 0) = id;
  header.range(30, 5) = 0;
  header[31] = header.range(30, 0).xor_reduce() ? ap_uint<1>(0)
                                                : ap_uint<1>(1);
  return header;
}

inline void packetize(RawStream& raw, AxisStream& output, int words,
                      unsigned id) {
#pragma HLS INLINE off
  const ap_uint<128> first = raw.read();
  AxisWord packet;
  packet.data.range(31, 0) = packet_header(id);
  packet.data.range(127, 32) = first.range(95, 0);
  packet.keep = -1;
  packet.strb = -1;
  packet.last = 0;
  output.write(packet);
  ap_uint<32> delayed = first.range(127, 96);
  for (int word = 1; word < words; ++word) {
#pragma HLS PIPELINE II=1
    const ap_uint<128> input = raw.read();
    packet.data.range(31, 0) = delayed;
    packet.data.range(127, 32) = input.range(95, 0);
    packet.keep = -1;
    packet.strb = -1;
    packet.last = 0;
    output.write(packet);
    delayed = input.range(127, 96);
  }
  packet.data = 0;
  packet.data.range(31, 0) = delayed;
  packet.keep = 0x000f;
  packet.strb = 0x000f;
  packet.last = 1;
  output.write(packet);
}

inline void packetize_q_lane0(RawStream& payload, AxisStream& packet) {
#pragma HLS INLINE off
  for (int q_in_group = 0; q_in_group < kQueriesPerKvHead; ++q_in_group)
    for (int group = 0; group < kKvHeads / 2; ++group)
      packetize(payload, packet, kPacketWordsPerHead, kQuantPacketId[group]);
}

inline void packetize_q_lane1(RawStream& payload, AxisStream& packet) {
#pragma HLS INLINE off
  for (int q_in_group = 0; q_in_group < kQueriesPerKvHead; ++q_in_group)
    for (int group = kKvHeads / 2; group < kKvHeads; ++group)
      packetize(payload, packet, kPacketWordsPerHead,
                kQuantPacketId[group % (kKvHeads / 2)]);
}

inline void packetize_k_lane0(RawStream& payload, AxisStream& packet) {
#pragma HLS INLINE off
  for (int group = 0; group < kKvHeads / 2; ++group)
    packetize(payload, packet, kPacketWordsPerHead, kQuantPacketId[group]);
}

inline void packetize_k_lane1(RawStream& payload, AxisStream& packet) {
#pragma HLS INLINE off
  for (int group = kKvHeads / 2; group < kKvHeads; ++group)
    packetize(payload, packet, kPacketWordsPerHead,
              kQuantPacketId[group % (kKvHeads / 2)]);
}

}  // namespace llama3_attn
