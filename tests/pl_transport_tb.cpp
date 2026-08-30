#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "llama3_attention_aie_8group_packet.cpp"

namespace {

uint16_t marker(int key, int dim) {
  return static_cast<uint16_t>(key * 256 + dim);
}

void fail(const char* message) {
  std::cerr << "PL C-sim FAILED: " << message << "\n";
  std::exit(1);
}

uint16_t lane16(const ap_uint<128>& word, int lane) {
  return word.range(16 * lane + 15, 16 * lane).to_uint();
}

uint16_t lane16(const DdrWord& word, int lane) {
  return word.range(16 * lane + 15, 16 * lane).to_uint();
}

void fill_value_ddr(std::vector<DdrWord>& value) {
  for (int group = 0; group < kKvHeads; ++group) {
    for (int key = 0; key < kCompactSequence; ++key) {
      for (int word = 0; word < 4; ++word) {
        DdrWord packed = 0;
        for (int lane = 0; lane < 32; ++lane) {
          const int dim = 32 * word + lane;
          packed.range(16 * lane + 15, 16 * lane) = marker(key, dim);
        }
        const unsigned long long index =
            static_cast<unsigned long long>(group * kCompactSequence + key) *
                4 + word;
        value[index] = packed;
      }
    }
  }
}

std::vector<ap_uint<128>> read_packet(AxisStream& stream, int words,
                                      unsigned expected_id) {
  AxisWord previous = stream.read();
  if (previous.data.range(4, 0).to_uint() != expected_id)
    fail("packet ID mismatch");
  std::vector<ap_uint<128>> payload;
  payload.reserve(words);
  for (int word = 0; word < words; ++word) {
    const AxisWord current = stream.read();
    ap_uint<128> reconstructed = 0;
    reconstructed.range(95, 0) = previous.data.range(127, 32);
    reconstructed.range(127, 96) = current.data.range(31, 0);
    payload.push_back(reconstructed);
    if (static_cast<bool>(current.last) != (word + 1 == words))
      fail("TLAST mismatch");
    previous = current;
  }
  return payload;
}

void verify_value_payload(const std::vector<ap_uint<128>>& payload,
                          int slice_offset, int width) {
  int payload_word = 0;
  for (int key_quad = 0; key_quad < kCompactSequence / 4; ++key_quad) {
    for (int dim_quad = 0; dim_quad < width / 4; ++dim_quad) {
      const ap_uint<128> low = payload[payload_word++];
      const ap_uint<128> high = payload[payload_word++];
      for (int element = 0; element < 16; ++element) {
        const int key_lane = element / 4;
        const int dim_lane = element % 4;
        const uint16_t actual =
            element < 8 ? lane16(low, element) : lane16(high, element - 8);
        const uint16_t expected =
            marker(4 * key_quad + key_lane,
                   slice_offset + 4 * dim_quad + dim_lane);
        if (actual != expected) fail("key/dim tile order mismatch");
      }
    }
  }
  if (payload_word != static_cast<int>(payload.size()))
    fail("Value payload length mismatch");
}

void verify_value_input_transport(const std::vector<DdrWord>& value) {
  RawStream raw[6];
  route_v_bf16(value.data(), 0, raw);
  for (int slice = 0; slice < kSliceCount; ++slice) {
    for (int group = 0; group < kKvHeads; ++group) {
      std::vector<ap_uint<128>> payload;
      payload.reserve(kValueWordsPerGroup[slice]);
      for (int word = 0; word < kValueWordsPerGroup[slice]; ++word)
        payload.push_back(raw[slice].read());
      verify_value_payload(payload, kSliceOffsets[slice],
                           kSliceWidths[slice]);
    }
    if (!raw[slice].empty()) fail("unexpected Value input token");
  }

  RawStream packet_raw[6];
  AxisStream packet_axis[6];
  route_v_bf16(value.data(), 0, packet_raw);
  for (int slice = 0; slice < kSliceCount; ++slice) {
    packetize_v(packet_raw[slice], packet_axis[slice],
                kValueWordsPerGroup[slice],
                kVPacketId[slice]);
    for (int group = 0; group < kKvHeads; ++group) {
      const auto payload = read_packet(packet_axis[slice],
                                       kValueWordsPerGroup[slice],
                                       kVPacketId[slice][group]);
      verify_value_payload(payload, kSliceOffsets[slice],
                           kSliceWidths[slice]);
    }
    if (!packet_axis[slice].empty()) fail("unexpected Value packet beat");
  }
}

std::array<int, 8> group_order(int slice) {
  if (slice == 1) return {7, 6, 5, 4, 3, 2, 1, 0};
  if (slice == 2) return {0, 2, 4, 6, 1, 3, 5, 7};
  if (slice == 3) return {1, 3, 5, 7, 0, 2, 4, 6};
  if (slice == 4) return {4, 5, 6, 7, 0, 1, 2, 3};
  return {0, 1, 2, 3, 4, 5, 6, 7};
}

unsigned output_packet_id(int slice, int group) {
  for (unsigned id = 0; id < 32; ++id)
    if (kOIdToGroup[slice][id] == static_cast<unsigned>(group)) return id;
  fail("missing output packet ID");
  return 0;
}

void append_output_packet(AxisStream& axis, int slice, int group, int phase) {
  const int width = kSliceWidths[slice];
  const int offset = kSliceOffsets[slice];
  RawStream payload;
  ap_uint<128> word = 0;
  int lane = 0;
  for (int query = 0; query < kQueriesPerPacket; ++query) {
    const int q_head = group * kQueriesPerKvHead +
                       phase * kQueriesPerPacket + query;
    for (int row = 0; row < kCompactSequence; ++row) {
      const int selected_key = (7 * q_head + row) % kCompactSequence;
      for (int dim = 0; dim < width; ++dim) {
        word.range(16 * lane + 15, 16 * lane) =
            marker(selected_key, offset + dim);
        if (++lane == 8) {
          payload.write(word);
          word = 0;
          lane = 0;
        }
      }
    }
  }
  if (lane != 0) fail("output packet is not 128-bit aligned");
  const int words = 2 * kCompactSequence * width * 2 / 16;
  packetize(payload, axis, words, output_packet_id(slice, group));
  if (!payload.empty()) fail("output packet payload count mismatch");
}

void verify_output_reorder() {
  AxisStream output_axis[6];
  for (int slice = 0; slice < kSliceCount; ++slice) {
    const auto order = group_order(slice);
    for (int phase = 0; phase < kPacketsPerGroup; ++phase)
      for (int group : order)
        append_output_packet(output_axis[slice], slice, group, phase);
  }

  hls::stream<SlicePair<24>> pairs24[4];
  hls::stream<SlicePair<16>> pairs16[2];
  SliceRowStream rows[6];
  SliceRowStream slice1_left, slice1_right, slice4_left, slice4_right;
  drain_output_pairs<24>(output_axis[0], pairs24[0], kOIdToGroup[0]);
  drain_output_pairs<24>(output_axis[1], pairs24[1], kOIdToGroup[1]);
  drain_output_pairs<16>(output_axis[2], pairs16[0], kOIdToGroup[2]);
  drain_output_pairs<24>(output_axis[3], pairs24[2], kOIdToGroup[3]);
  drain_output_pairs<24>(output_axis[4], pairs24[3], kOIdToGroup[4]);
  drain_output_pairs<16>(output_axis[5], pairs16[1], kOIdToGroup[5]);
  split_output_pairs<24>(pairs24[0], rows[0]);
  split_output_pairs<24>(pairs24[1], rows[1]);
  split_output_pairs<16>(pairs16[0], rows[2]);
  split_output_pairs<24>(pairs24[2], rows[3]);
  split_output_pairs<24>(pairs24[3], rows[4]);
  split_output_pairs<16>(pairs16[1], rows[5]);
  fanout_slice_rows(rows[1], slice1_left, slice1_right);
  fanout_slice_rows(rows[4], slice4_left, slice4_right);

  std::vector<DdrWord> plane0(kRowsPerBatch);
  std::vector<DdrWord> plane1(kRowsPerBatch);
  std::vector<DdrWord> plane2(kRowsPerBatch);
  std::vector<DdrWord> plane3(kRowsPerBatch);
  assemble_output_plane<0, 24, 0, 8>(rows[0], slice1_left, plane0.data(), 0);
  assemble_output_plane<8, 16, 0, 16>(slice1_right, rows[2], plane1.data(), 0);
  assemble_output_plane<0, 24, 0, 8>(rows[3], slice4_left, plane2.data(), 0);
  assemble_output_plane<8, 16, 0, 16>(slice4_right, rows[5], plane3.data(), 0);

  const DdrWord* planes[4] = {plane0.data(), plane1.data(), plane2.data(), plane3.data()};
  for (int q_head = 0; q_head < kQueryHeads; ++q_head) {
    for (int row = 0; row < kCompactSequence; ++row) {
      const int row_index = q_head * kCompactSequence + row;
      const int selected_key = (7 * q_head + row) % kCompactSequence;
      for (int dim = 0; dim < 128; ++dim) {
        const uint16_t actual = lane16(planes[dim / 32][row_index], dim % 32);
        if (actual != marker(selected_key, dim))
          fail("six-slice to four-plane reorder mismatch");
      }
    }
  }
  for (auto& axis : output_axis)
    if (!axis.empty()) fail("unexpected output packet beat");
}

void fill_payload(RawStream& payload, int packets) {
  for (int word = 0; word < packets * kPacketWordsPerHead; ++word)
    payload.write(ap_uint<128>(word));
}

void verify_qk_partition_and_schedule() {
  RawStream q_payload[2];
  AxisStream q_axis[2];
  fill_payload(q_payload[0], 2 * kKvHeads);
  fill_payload(q_payload[1], 2 * kKvHeads);
  packetize_q_lane0(q_payload[0], q_axis[0]);
  packetize_q_lane1(q_payload[1], q_axis[1]);

  for (int q_in_group = 0; q_in_group < kQueriesPerKvHead; ++q_in_group) {
    for (int local_group = 0; local_group < kKvHeads / 2; ++local_group) {
      read_packet(q_axis[0], kPacketWordsPerHead, local_group);
      read_packet(q_axis[1], kPacketWordsPerHead, local_group);
    }
  }
  if (!q_payload[0].empty() || !q_payload[1].empty() ||
      !q_axis[0].empty() || !q_axis[1].empty())
    fail("Q wave-major packet count mismatch");

  RawStream k_payload[2];
  AxisStream k_axis[2];
  fill_payload(k_payload[0], kKvHeads / 2);
  fill_payload(k_payload[1], kKvHeads / 2);
  packetize_k_lane0(k_payload[0], k_axis[0]);
  packetize_k_lane1(k_payload[1], k_axis[1]);
  for (int local_group = 0; local_group < kKvHeads / 2; ++local_group) {
    read_packet(k_axis[0], kPacketWordsPerHead, local_group);
    read_packet(k_axis[1], kPacketWordsPerHead, local_group);
  }
  if (!k_payload[0].empty() || !k_payload[1].empty() ||
      !k_axis[0].empty() || !k_axis[1].empty())
    fail("K partition packet count mismatch");
}

}  // namespace

int main() {
  std::vector<DdrWord> value(kKvHeads * kCompactSequence * 4);
  fill_value_ddr(value);
  verify_qk_partition_and_schedule();
  verify_value_input_transport(value);
  verify_output_reorder();
  std::cout << "PL C-sim PASS: strict 96/64 input and 192/128 output words\n";
  std::cout << "PL C-sim PASS: Q wave-major 16/lane, K partitioned 4/lane\n";
  std::cout << "PL C-sim PASS: 6 slices -> 4 contiguous 512-bit DDR planes\n";
  return 0;
}
