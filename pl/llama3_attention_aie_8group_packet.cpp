// PL kernel: 24+24+16+24+24+16 Value tiles and four-plane output.
// ============================================================================

#include "llama3_attention_pl_common.hpp"
#include "llama3_attention_packet_ids.hpp"

namespace {

using namespace llama3_attn;

constexpr int kSliceCount = 6;
constexpr int kSliceWidths[kSliceCount] = {24, 24, 16, 24, 24, 16};
constexpr int kSliceOffsets[kSliceCount] = {0, 24, 48, 64, 88, 112};
constexpr int kValueWordsPerGroup[kSliceCount] = {
    96, 96, 64, 96, 96, 64};
static_assert(24 + 24 + 16 + 24 + 24 + 16 == 128, "PV slices must cover D=128");
static_assert(24 == 0 + 24 && 48 == 24 + 24 && 64 == 48 + 16 &&
              88 == 64 + 24 && 112 == 88 + 24 && 128 == 112 + 16,
              "PV slice offsets must be contiguous");
constexpr int kQueriesPerPacket = 2;
constexpr int kPacketsPerGroup = kQueriesPerKvHead / kQueriesPerPacket;
constexpr int kRowsPerPacket = kQueriesPerPacket * kCompactSequence;
constexpr int kRowsPerBatch = kQueryHeads * kCompactSequence;

struct SliceRow {
  ap_uint<448> data;
  ap_uint<10> row_index;
};

template <int ValueD>
struct SlicePair {
  ap_uint<2 * ValueD * 16> data;
  ap_uint<10> first_row_index;
};

using SliceRowStream = hls::stream<SliceRow>;

template <int SliceOffset, int ValueD>
void emit_v_slice_tiles(const DdrWord rows[4][4], RawStream& raw) {
#pragma HLS INLINE
  static_assert((SliceOffset % 4) == 0, "slice offset must be quad aligned");
  static_assert((ValueD % 4) == 0, "slice width must be quad aligned");
  for (int dim_quad = 0; dim_quad < ValueD / 4; ++dim_quad) {
#pragma HLS PIPELINE II=2
    ap_uint<256> tile = 0;
    const int global_quad = SliceOffset / 4 + dim_quad;
    for (int key = 0; key < 4; ++key) {
#pragma HLS UNROLL
      tile.range(key * 64 + 63, key * 64) =
          select_global_bf16_quad(rows[key], global_quad);
    }
    raw.write(tile.range(127, 0));
    raw.write(tile.range(255, 128));
  }
}

void route_v_bf16(const DdrWord* v, int batch, RawStream raw[6]) {
#pragma HLS INLINE off
  const unsigned long long batch_base =
      static_cast<unsigned long long>(batch) * kKvWordsPerBatch;
  for (int group = 0; group < kKvHeads; ++group) {
    for (int key_quad = 0; key_quad < kCompactSequence / 4; ++key_quad) {
      DdrWord rows[4][4];
#pragma HLS ARRAY_PARTITION variable=rows complete
      const unsigned long long key_base =
          batch_base +
          static_cast<unsigned long long>(group * kCompactSequence +
                                          key_quad * 4) *
              4;
      for (int key = 0; key < 4; ++key) {
        for (int word = 0; word < 4; ++word) {
#pragma HLS PIPELINE II=1
          rows[key][word] = v[key_base + key * 4 + word];
        }
      }
      emit_v_slice_tiles<0, 24>(rows, raw[0]);
      emit_v_slice_tiles<24, 24>(rows, raw[1]);
      emit_v_slice_tiles<48, 16>(rows, raw[2]);
      emit_v_slice_tiles<64, 24>(rows, raw[3]);
      emit_v_slice_tiles<88, 24>(rows, raw[4]);
      emit_v_slice_tiles<112, 16>(rows, raw[5]);
    }
  }
}

void packetize_v(RawStream& raw, AxisStream& packet,
                 int words_per_group,
                 const unsigned packet_ids[kKvHeads]) {
#pragma HLS INLINE off
  for (int group = 0; group < kKvHeads; ++group)
    packetize(raw, packet, words_per_group, packet_ids[group]);
}

template <int ValueD>
void drain_output_pairs(
    AxisStream& packet, hls::stream<SlicePair<ValueD>>& pairs,
    const unsigned id_to_group[32]) {
#pragma HLS INLINE off
  constexpr int kWordsPerPair = 2 * ValueD * 2 / 16;
  unsigned packets_seen[kKvHeads];
#pragma HLS ARRAY_PARTITION variable=packets_seen complete
  for (int group = 0; group < kKvHeads; ++group) packets_seen[group] = 0;

  for (int packet_number = 0;
       packet_number < kKvHeads * kPacketsPerGroup; ++packet_number) {
    AxisWord previous = packet.read();
    const unsigned packet_id = previous.data.range(4, 0);
    const unsigned group = id_to_group[packet_id];
    const unsigned invocation_phase = packets_seen[group]++;
    for (int row_pair = 0; row_pair < kRowsPerPacket / 2; ++row_pair) {
      ap_uint<2 * ValueD * 16> pair_data = 0;
      for (int word = 0; word < kWordsPerPair; ++word) {
#pragma HLS PIPELINE II=1
        const AxisWord current = packet.read();
        ap_uint<128> payload = 0;
        payload.range(95, 0) = previous.data.range(127, 32);
        payload.range(127, 96) = current.data.range(31, 0);
        pair_data >>= 128;
        pair_data.range(2 * ValueD * 16 - 1, 2 * ValueD * 16 - 128) =
            payload;
        previous = current;
      }

      const int row_in_packet = 2 * row_pair;
      const int q_in_packet = row_in_packet / kCompactSequence;
      const int sequence = row_in_packet % kCompactSequence;
      const int q_head = group * kQueriesPerKvHead +
                         invocation_phase * kQueriesPerPacket +
                         q_in_packet;
      SlicePair<ValueD> output_pair;
      output_pair.data = pair_data;
      output_pair.first_row_index = q_head * kCompactSequence + sequence;
      pairs.write(output_pair);
    }
  }
}

template <int ValueD>
void split_output_pairs(
    hls::stream<SlicePair<ValueD>>& pairs, SliceRowStream& rows) {
#pragma HLS INLINE off
  for (int pair_index = 0; pair_index < kRowsPerBatch / 2; ++pair_index) {
#pragma HLS PIPELINE II=2
    const SlicePair<ValueD> input_pair = pairs.read();
    SliceRow row0;
    SliceRow row1;
    row0.data = 0;
    row1.data = 0;
    row0.data.range(ValueD * 16 - 1, 0) =
        input_pair.data.range(ValueD * 16 - 1, 0);
    row1.data.range(ValueD * 16 - 1, 0) =
        input_pair.data.range(2 * ValueD * 16 - 1, ValueD * 16);
    row0.row_index = input_pair.first_row_index;
    row1.row_index = input_pair.first_row_index + 1;
    rows.write(row0);
    rows.write(row1);
  }
}

void fanout_slice_rows(SliceRowStream& input,
                       SliceRowStream& output_a,
                       SliceRowStream& output_b) {
#pragma HLS INLINE off
  for (int row = 0; row < kRowsPerBatch; ++row) {
#pragma HLS PIPELINE II=1
    const SliceRow value = input.read();
    output_a.write(value);
    output_b.write(value);
  }
}

template <int LeftStart, int LeftWidth, int RightStart, int RightWidth>
void assemble_output_plane(SliceRowStream& left,
                           SliceRowStream& right, DdrWord* output,
                           int batch) {
#pragma HLS INLINE off
  static_assert(LeftWidth + RightWidth == 32,
                "each host output plane must contain exactly 32 BF16 values");
  ap_uint<LeftWidth * 16> left_rows[kRowsPerBatch];
  ap_uint<RightWidth * 16> right_rows[kRowsPerBatch];
#pragma HLS BIND_STORAGE variable=left_rows type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=right_rows type=ram_2p impl=bram

  int left_count = 0;
  int right_count = 0;
  while (left_count < kRowsPerBatch || right_count < kRowsPerBatch) {
#pragma HLS PIPELINE II=1
    SliceRow left_row;
    SliceRow right_row;
    if (left_count < kRowsPerBatch && left.read_nb(left_row)) {
      left_rows[left_row.row_index] =
          left_row.data.range((LeftStart + LeftWidth) * 16 - 1,
                              LeftStart * 16);
      ++left_count;
    }
    if (right_count < kRowsPerBatch && right.read_nb(right_row)) {
      right_rows[right_row.row_index] =
          right_row.data.range((RightStart + RightWidth) * 16 - 1,
                               RightStart * 16);
      ++right_count;
    }
  }

  const unsigned long long batch_base =
      static_cast<unsigned long long>(batch) * kRowsPerBatch;
  for (int row = 0; row < kRowsPerBatch; ++row) {
#pragma HLS PIPELINE II=1
    DdrWord packed = 0;
    packed.range(LeftWidth * 16 - 1, 0) = left_rows[row];
    packed.range(511, LeftWidth * 16) = right_rows[row];
    output[batch_base + row] = packed;
  }
}

void run_attention_dataflow(
    const DdrWord* q, const DdrWord* k, const DdrWord* v,
    DdrWord* o0, DdrWord* o1, DdrWord* o2, DdrWord* o3, int batch,
    AxisStream& packet_q0, AxisStream& packet_q1, AxisStream& packet_k0, AxisStream& packet_k1,
    AxisStream& packet_v0, AxisStream& packet_v1, AxisStream& packet_v2, AxisStream& packet_v3,
    AxisStream& packet_v4, AxisStream& packet_v5, AxisStream& packet_o0,
    AxisStream& packet_o1, AxisStream& packet_o2, AxisStream& packet_o3,
    AxisStream& packet_o4, AxisStream& packet_o5) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW
  DdrStream q_ddr[2];
  DdrStream k_ddr;
  ScaleStream q_scales[2];
  ScaleStream k_scales[2];
  RawStream q_quantized[2];
  RawStream k_quantized[2];
  RawStream q_payload[2];
  RawStream k_payload[2];
  RawStream v_raw[6];
  hls::stream<SlicePair<24>> output_pairs24[4];
  hls::stream<SlicePair<16>> output_pairs16[2];
  SliceRowStream slice_rows[6];
  SliceRowStream slice1_left, slice1_right;
  SliceRowStream slice4_left, slice4_right;
#pragma HLS ARRAY_PARTITION variable=q_ddr complete
#pragma HLS ARRAY_PARTITION variable=q_scales complete
#pragma HLS ARRAY_PARTITION variable=k_scales complete
#pragma HLS ARRAY_PARTITION variable=q_quantized complete
#pragma HLS ARRAY_PARTITION variable=k_quantized complete
#pragma HLS ARRAY_PARTITION variable=q_payload complete
#pragma HLS ARRAY_PARTITION variable=k_payload complete
#pragma HLS ARRAY_PARTITION variable=v_raw complete
#pragma HLS ARRAY_PARTITION variable=output_pairs24 complete
#pragma HLS ARRAY_PARTITION variable=output_pairs16 complete
#pragma HLS ARRAY_PARTITION variable=slice_rows complete
#pragma HLS STREAM variable=q_ddr depth=64
#pragma HLS STREAM variable=k_ddr depth=64
#pragma HLS STREAM variable=q_scales depth=128
#pragma HLS STREAM variable=k_scales depth=128
#pragma HLS STREAM variable=q_quantized depth=512
#pragma HLS STREAM variable=k_quantized depth=512
#pragma HLS STREAM variable=q_payload depth=64
#pragma HLS STREAM variable=k_payload depth=64
#pragma HLS STREAM variable=v_raw depth=256
#pragma HLS STREAM variable=output_pairs24 depth=8
#pragma HLS STREAM variable=output_pairs16 depth=8
#pragma HLS STREAM variable=slice_rows depth=32
#pragma HLS STREAM variable=slice1_left depth=32
#pragma HLS STREAM variable=slice1_right depth=32
#pragma HLS STREAM variable=slice4_left depth=32
#pragma HLS STREAM variable=slice4_right depth=32

  read_q_ddr(q, batch, q_ddr);
  read_k_ddr(k, batch, k_ddr);
  quantize_q_lane(q_ddr[0], q_scales[0], q_quantized[0]);
  quantize_q_lane(q_ddr[1], q_scales[1], q_quantized[1]);
  quantize_k_once(k_ddr, k_scales[0], k_scales[1], k_quantized[0],
                  k_quantized[1]);
  assemble_quantized_packets(q_scales[0], q_quantized[0], q_payload[0],
                             2 * kKvHeads);
  assemble_quantized_packets(q_scales[1], q_quantized[1], q_payload[1],
                             2 * kKvHeads);
  assemble_quantized_packets(k_scales[0], k_quantized[0], k_payload[0],
                             kKvHeads / 2);
  assemble_quantized_packets(k_scales[1], k_quantized[1], k_payload[1],
                             kKvHeads / 2);
  packetize_q_lane0(q_payload[0], packet_q0);
  packetize_q_lane1(q_payload[1], packet_q1);
  packetize_k_lane0(k_payload[0], packet_k0);
  packetize_k_lane1(k_payload[1], packet_k1);

  route_v_bf16(v, batch, v_raw);
  packetize_v(v_raw[0], packet_v0, kValueWordsPerGroup[0], kVPacketId[0]);
  packetize_v(v_raw[1], packet_v1, kValueWordsPerGroup[1], kVPacketId[1]);
  packetize_v(v_raw[2], packet_v2, kValueWordsPerGroup[2], kVPacketId[2]);
  packetize_v(v_raw[3], packet_v3, kValueWordsPerGroup[3], kVPacketId[3]);
  packetize_v(v_raw[4], packet_v4, kValueWordsPerGroup[4], kVPacketId[4]);
  packetize_v(v_raw[5], packet_v5, kValueWordsPerGroup[5], kVPacketId[5]);

  drain_output_pairs<24>(packet_o0, output_pairs24[0], kOIdToGroup[0]);
  drain_output_pairs<24>(packet_o1, output_pairs24[1], kOIdToGroup[1]);
  drain_output_pairs<16>(packet_o2, output_pairs16[0], kOIdToGroup[2]);
  drain_output_pairs<24>(packet_o3, output_pairs24[2], kOIdToGroup[3]);
  drain_output_pairs<24>(packet_o4, output_pairs24[3], kOIdToGroup[4]);
  drain_output_pairs<16>(packet_o5, output_pairs16[1], kOIdToGroup[5]);
  split_output_pairs<24>(output_pairs24[0], slice_rows[0]);
  split_output_pairs<24>(output_pairs24[1], slice_rows[1]);
  split_output_pairs<16>(output_pairs16[0], slice_rows[2]);
  split_output_pairs<24>(output_pairs24[2], slice_rows[3]);
  split_output_pairs<24>(output_pairs24[3], slice_rows[4]);
  split_output_pairs<16>(output_pairs16[1], slice_rows[5]);

  fanout_slice_rows(slice_rows[1], slice1_left, slice1_right);
  fanout_slice_rows(slice_rows[4], slice4_left, slice4_right);

  assemble_output_plane<0, 24, 0, 8>(slice_rows[0], slice1_left, o0, batch);
  assemble_output_plane<8, 16, 0, 16>(slice1_right, slice_rows[2], o1, batch);
  assemble_output_plane<0, 24, 0, 8>(slice_rows[3], slice4_left, o2, batch);
  assemble_output_plane<8, 16, 0, 16>(slice4_right, slice_rows[5], o3, batch);
}

}  // namespace

extern "C" void llama3_attention_aie8_packet(
    const DdrWord* q, const DdrWord* k, const DdrWord* v,
    DdrWord* o0, DdrWord* o1, DdrWord* o2, DdrWord* o3,
    int batch_size, int seq_len,
    AxisStream& packet_q0, AxisStream& packet_q1,
    AxisStream& packet_k0, AxisStream& packet_k1,
    AxisStream& packet_v0, AxisStream& packet_v1,
    AxisStream& packet_v2, AxisStream& packet_v3,
    AxisStream& packet_v4, AxisStream& packet_v5,
    AxisStream& packet_o0, AxisStream& packet_o1,
    AxisStream& packet_o2, AxisStream& packet_o3,
    AxisStream& packet_o4, AxisStream& packet_o5) {
#pragma HLS INTERFACE m_axi port=q offset=slave bundle=gmem0 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=k offset=slave bundle=gmem1 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=v offset=slave bundle=gmem2 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=o0 offset=slave bundle=gmem3 max_write_burst_length=64 num_write_outstanding=16
#pragma HLS INTERFACE m_axi port=o1 offset=slave bundle=gmem4 max_write_burst_length=64 num_write_outstanding=16
#pragma HLS INTERFACE m_axi port=o2 offset=slave bundle=gmem5 max_write_burst_length=64 num_write_outstanding=16
#pragma HLS INTERFACE m_axi port=o3 offset=slave bundle=gmem6 max_write_burst_length=64 num_write_outstanding=16
#pragma HLS INTERFACE axis port=packet_q0
#pragma HLS INTERFACE axis port=packet_q1
#pragma HLS INTERFACE axis port=packet_k0
#pragma HLS INTERFACE axis port=packet_k1
#pragma HLS INTERFACE axis port=packet_v0
#pragma HLS INTERFACE axis port=packet_v1
#pragma HLS INTERFACE axis port=packet_v2
#pragma HLS INTERFACE axis port=packet_v3
#pragma HLS INTERFACE axis port=packet_v4
#pragma HLS INTERFACE axis port=packet_v5
#pragma HLS INTERFACE axis port=packet_o0
#pragma HLS INTERFACE axis port=packet_o1
#pragma HLS INTERFACE axis port=packet_o2
#pragma HLS INTERFACE axis port=packet_o3
#pragma HLS INTERFACE axis port=packet_o4
#pragma HLS INTERFACE axis port=packet_o5
#pragma HLS INTERFACE s_axilite port=q bundle=control
#pragma HLS INTERFACE s_axilite port=k bundle=control
#pragma HLS INTERFACE s_axilite port=v bundle=control
#pragma HLS INTERFACE s_axilite port=o0 bundle=control
#pragma HLS INTERFACE s_axilite port=o1 bundle=control
#pragma HLS INTERFACE s_axilite port=o2 bundle=control
#pragma HLS INTERFACE s_axilite port=o3 bundle=control
#pragma HLS INTERFACE s_axilite port=batch_size bundle=control
#pragma HLS INTERFACE s_axilite port=seq_len bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  if (batch_size <= 0 || seq_len != kCompactSequence) return;
  for (int batch = 0; batch < batch_size; ++batch)
    run_attention_dataflow(
        q, k, v, o0, o1, o2, o3, batch,
        packet_q0, packet_q1, packet_k0, packet_k1,
        packet_v0, packet_v1, packet_v2, packet_v3,
        packet_v4, packet_v5, packet_o0, packet_o1, packet_o2,
        packet_o3, packet_o4, packet_o5);
}
