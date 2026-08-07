#include "llama3_attention_config.hpp"

#include "ap_axi_sdata.h"
#include "hls_stream.h"

using llama3_attn::float_to_bf16;
using llama3_attn::kv_offset;
using llama3_attn::q_offset;
using namespace llama3_attn;

namespace {

// The AIE graph uses 128-bit PLIOs.  A word therefore carries eight BF16
// elements or four FP32 elements, in increasing element order.
using AxisWord = ap_axiu<128, 0, 0, 0>;
using AxisStream = hls::stream<AxisWord>;

constexpr int kBf16PerWord = 8;
constexpr int kFloatPerWord = 4;

void write_word(AxisStream& stream, ap_uint<128> data, bool last) {
#pragma HLS INLINE
  AxisWord word;
  word.data = data;
  word.keep = -1;
  word.strb = -1;
  word.last = last;
  stream.write(word);
}

void send_q_slice(const ap_uint<16>* q_mem, int batch, int q_head,
                  int query_start, int query_count, int seq_len,
                  int dim_base, AxisStream& stream) {
  for (int index = 0; index < kQueryBlock * 64; index += kBf16PerWord) {
#pragma HLS PIPELINE II=1
    ap_uint<128> packed = 0;
    for (int lane = 0; lane < kBf16PerWord; ++lane) {
#pragma HLS UNROLL
      const int element = index + lane;
      const int row = element / 64;
      const int dim = dim_base + element % 64;
      const ap_uint<16> value = row < query_count
                                    ? q_mem[q_offset(batch, q_head,
                                                     query_start + row, dim,
                                                     seq_len)]
                                    : ap_uint<16>(0);
      packed.range(16 * lane + 15, 16 * lane) = value;
    }
    write_word(stream, packed, index + kBf16PerWord == kQueryBlock * 64);
  }
}

void send_k_slice(const ap_uint<16> tile[kKeyBlock][kHeadDim],
                  int dim_base, AxisStream& stream) {
  for (int index = 0; index < kKeyBlock * 64; index += kBf16PerWord) {
#pragma HLS PIPELINE II=1
    ap_uint<128> packed = 0;
    for (int lane = 0; lane < kBf16PerWord; ++lane) {
#pragma HLS UNROLL
      const int element = index + lane;
      packed.range(16 * lane + 15, 16 * lane) =
          tile[element / 64][dim_base + element % 64];
    }
    write_word(stream, packed, index + kBf16PerWord == kKeyBlock * 64);
  }
}

void send_value_slice(const ap_uint<16> tile[kKeyBlock][kHeadDim],
                      int dim_base, AxisStream& stream) {
  for (int index = 0; index < kKeyBlock * 32; index += kBf16PerWord) {
#pragma HLS PIPELINE II=1
    ap_uint<128> packed = 0;
    for (int lane = 0; lane < kBf16PerWord; ++lane) {
#pragma HLS UNROLL
      const int element = index + lane;
      packed.range(16 * lane + 15, 16 * lane) =
          tile[element / 32][dim_base + element % 32];
    }
    write_word(stream, packed, index + kBf16PerWord == kKeyBlock * 32);
  }
}

void send_mask(int query_start, int query_count, int key_start, int key_count,
               AxisStream& stream) {
  for (int index = 0; index < kQueryBlock * kKeyBlock;
       index += kFloatPerWord) {
#pragma HLS PIPELINE II=1
    ap_uint<128> packed = 0;
    for (int lane = 0; lane < kFloatPerWord; ++lane) {
#pragma HLS UNROLL
      const int element = index + lane;
      const int row = element / kKeyBlock;
      const int column = element % kKeyBlock;
      const bool valid = row < query_count && column < key_count &&
                         key_start + column <= query_start + row;
      union {
        float value;
        unsigned int bits;
      } converter;
      converter.value = valid ? 0.0f : -3.402823466e+38F;
      packed.range(32 * lane + 31, 32 * lane) = converter.bits;
    }
    write_word(stream, packed, index + kFloatPerWord == kQueryBlock * kKeyBlock);
  }
}

void send_state(const float state[kQueryBlock][2], AxisStream& stream) {
  for (int index = 0; index < kQueryBlock * 2; index += kFloatPerWord) {
#pragma HLS PIPELINE II=1
    ap_uint<128> packed = 0;
    for (int lane = 0; lane < kFloatPerWord; ++lane) {
#pragma HLS UNROLL
      union {
        float value;
        unsigned int bits;
      } converter;
      const int element = index + lane;
      converter.value = state[element / 2][element % 2];
      packed.range(32 * lane + 31, 32 * lane) = converter.bits;
    }
    write_word(stream, packed, index + kFloatPerWord == kQueryBlock * 2);
  }
}

void send_numerator(const float numerator[kQueryBlock][32], AxisStream& stream) {
  for (int index = 0; index < kQueryBlock * 32; index += kFloatPerWord) {
#pragma HLS PIPELINE II=1
    ap_uint<128> packed = 0;
    for (int lane = 0; lane < kFloatPerWord; ++lane) {
#pragma HLS UNROLL
      union {
        float value;
        unsigned int bits;
      } converter;
      const int element = index + lane;
      converter.value = numerator[element / 32][element % 32];
      packed.range(32 * lane + 31, 32 * lane) = converter.bits;
    }
    write_word(stream, packed, index + kFloatPerWord == kQueryBlock * 32);
  }
}

void receive_state(AxisStream& stream, float state[kQueryBlock][2]) {
  for (int index = 0; index < kQueryBlock * 3; index += kFloatPerWord) {
#pragma HLS PIPELINE II=1
    const ap_uint<128> packed = stream.read().data;
    for (int lane = 0; lane < kFloatPerWord; ++lane) {
#pragma HLS UNROLL
      const int element = index + lane;
      if (element % 3 != 2) {
        union {
          float value;
          unsigned int bits;
        } converter;
        converter.bits = packed.range(32 * lane + 31, 32 * lane);
        state[element / 3][element % 3] = converter.value;
      }
    }
  }
}

void receive_numerator(AxisStream& stream, float numerator[kQueryBlock][32]) {
  for (int index = 0; index < kQueryBlock * 32; index += kFloatPerWord) {
#pragma HLS PIPELINE II=1
    const ap_uint<128> packed = stream.read().data;
    for (int lane = 0; lane < kFloatPerWord; ++lane) {
#pragma HLS UNROLL
      union {
        float value;
        unsigned int bits;
      } converter;
      const int element = index + lane;
      converter.bits = packed.range(32 * lane + 31, 32 * lane);
      numerator[element / 32][element % 32] = converter.value;
    }
  }
}

void load_kv_tile(const ap_uint<16>* k_mem, const ap_uint<16>* v_mem,
                  int batch, int kv_head, int key_start, int key_count,
                  int seq_len, ap_uint<16> k_tile[kKeyBlock][kHeadDim],
                  ap_uint<16> v_tile[kKeyBlock][kHeadDim]) {
  for (int key = 0; key < kKeyBlock; ++key) {
    for (int dim = 0; dim < kHeadDim; ++dim) {
#pragma HLS PIPELINE II=1
      if (key < key_count) {
        const unsigned long long offset =
            kv_offset(batch, kv_head, key_start + key, dim, seq_len);
        k_tile[key][dim] = k_mem[offset];
        v_tile[key][dim] = v_mem[offset];
      } else {
        k_tile[key][dim] = 0;
        v_tile[key][dim] = 0;
      }
    }
  }
}

}  // namespace

// One physical eight-core AIE group is time-multiplexed across all Llama3 GQA
// groups and their four query heads.  All score, softmax and P*V arithmetic
// is performed by the AIE graph; this shell only owns memory, tile transport,
// online-state persistence and final normalization.
extern "C" void llama3_attention_aie(
    const ap_uint<16>* q_mem, const ap_uint<16>* k_mem,
    const ap_uint<16>* v_mem, ap_uint<16>* o_mem, int batch_size, int seq_len,
    AxisStream& q0, AxisStream& q1, AxisStream& k0, AxisStream& k1,
    AxisStream& mask, AxisStream& state_in, AxisStream& v0, AxisStream& v1,
    AxisStream& v2, AxisStream& v3, AxisStream& numerator_in0,
    AxisStream& numerator_in1, AxisStream& numerator_in2,
    AxisStream& numerator_in3, AxisStream& state_out, AxisStream& numerator_out0,
    AxisStream& numerator_out1, AxisStream& numerator_out2,
    AxisStream& numerator_out3) {
#pragma HLS INTERFACE m_axi port=q_mem offset=slave bundle=gmem0 max_read_burst_length=64
#pragma HLS INTERFACE m_axi port=k_mem offset=slave bundle=gmem1 max_read_burst_length=64
#pragma HLS INTERFACE m_axi port=v_mem offset=slave bundle=gmem2 max_read_burst_length=64
#pragma HLS INTERFACE m_axi port=o_mem offset=slave bundle=gmem3 max_write_burst_length=64
#pragma HLS INTERFACE axis port=q0
#pragma HLS INTERFACE axis port=q1
#pragma HLS INTERFACE axis port=k0
#pragma HLS INTERFACE axis port=k1
#pragma HLS INTERFACE axis port=mask
#pragma HLS INTERFACE axis port=state_in
#pragma HLS INTERFACE axis port=v0
#pragma HLS INTERFACE axis port=v1
#pragma HLS INTERFACE axis port=v2
#pragma HLS INTERFACE axis port=v3
#pragma HLS INTERFACE axis port=numerator_in0
#pragma HLS INTERFACE axis port=numerator_in1
#pragma HLS INTERFACE axis port=numerator_in2
#pragma HLS INTERFACE axis port=numerator_in3
#pragma HLS INTERFACE axis port=state_out
#pragma HLS INTERFACE axis port=numerator_out0
#pragma HLS INTERFACE axis port=numerator_out1
#pragma HLS INTERFACE axis port=numerator_out2
#pragma HLS INTERFACE axis port=numerator_out3
#pragma HLS INTERFACE s_axilite port=q_mem bundle=control
#pragma HLS INTERFACE s_axilite port=k_mem bundle=control
#pragma HLS INTERFACE s_axilite port=v_mem bundle=control
#pragma HLS INTERFACE s_axilite port=o_mem bundle=control
#pragma HLS INTERFACE s_axilite port=batch_size bundle=control
#pragma HLS INTERFACE s_axilite port=seq_len bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  if (batch_size <= 0 || seq_len <= 0) return;

  for (int batch = 0; batch < batch_size; ++batch) {
    for (int kv_head = 0; kv_head < kKvHeads; ++kv_head) {
      for (int q_in_group = 0; q_in_group < kQueriesPerKvHead; ++q_in_group) {
        const int q_head = kv_head * kQueriesPerKvHead + q_in_group;
        for (int query_start = 0; query_start < seq_len;
             query_start += kQueryBlock) {
          const int query_count = (seq_len - query_start) < kQueryBlock
                                      ? seq_len - query_start
                                      : kQueryBlock;
          float state[kQueryBlock][2];
          float numerator[4][kQueryBlock][32];
#pragma HLS BIND_STORAGE variable=numerator type=RAM_2P impl=URAM
#pragma HLS ARRAY_PARTITION variable=numerator complete dim=1

          for (int row = 0; row < kQueryBlock; ++row) {
            state[row][0] = -3.402823466e+38F;
            state[row][1] = 0.0f;
          }
          // Initialize each independently banked D=32 numerator lane. This
          // preserves II=1 without asking a single URAM to accept four writes.
          for (int lane = 0; lane < 4; ++lane) {
            for (int row = 0; row < kQueryBlock; ++row) {
              for (int dim = 0; dim < 32; ++dim) {
#pragma HLS PIPELINE II=1
                numerator[lane][row][dim] = 0.0f;
              }
            }
          }

          for (int key_start = 0; key_start < seq_len; key_start += kKeyBlock) {
            const int key_count = (seq_len - key_start) < kKeyBlock
                                      ? seq_len - key_start
                                      : kKeyBlock;
            ap_uint<16> k_tile[kKeyBlock][kHeadDim];
            ap_uint<16> v_tile[kKeyBlock][kHeadDim];
#pragma HLS BIND_STORAGE variable=k_tile type=RAM_2P impl=URAM
#pragma HLS BIND_STORAGE variable=v_tile type=RAM_2P impl=URAM
            load_kv_tile(k_mem, v_mem, batch, kv_head, key_start, key_count,
                         seq_len, k_tile, v_tile);

            send_q_slice(q_mem, batch, q_head, query_start, query_count,
                         seq_len, 0, q0);
            send_q_slice(q_mem, batch, q_head, query_start, query_count,
                         seq_len, 64, q1);
            send_k_slice(k_tile, 0, k0);
            send_k_slice(k_tile, 64, k1);
            send_mask(query_start, query_count, key_start, key_count, mask);
            send_state(state, state_in);
            send_value_slice(v_tile, 0, v0);
            send_value_slice(v_tile, 32, v1);
            send_value_slice(v_tile, 64, v2);
            send_value_slice(v_tile, 96, v3);
            send_numerator(numerator[0], numerator_in0);
            send_numerator(numerator[1], numerator_in1);
            send_numerator(numerator[2], numerator_in2);
            send_numerator(numerator[3], numerator_in3);

            receive_state(state_out, state);
            receive_numerator(numerator_out0, numerator[0]);
            receive_numerator(numerator_out1, numerator[1]);
            receive_numerator(numerator_out2, numerator[2]);
            receive_numerator(numerator_out3, numerator[3]);
          }

          for (int row = 0; row < query_count; ++row) {
            for (int lane = 0; lane < 4; ++lane) {
              for (int dim = 0; dim < 32; ++dim) {
#pragma HLS PIPELINE II=1
                o_mem[q_offset(batch, q_head, query_start + row,
                               lane * 32 + dim, seq_len)] =
                    float_to_bf16(numerator[lane][row][dim] / state[row][1]);
              }
            }
          }
        }
      }
    }
  }
}
