#include <adf.h>

#include <string>

#include "llama3_gqa_group_graph.hpp"

using namespace adf;

class Llama3Gqa8GroupPacketGraph : public graph {
 public:
  Llama3GqaGroupGraph group[8];
  input_plio q[2];
  input_plio k[2];
  input_plio value[6];
  output_plio output[6];
  pktsplit<4> q_split[2];
  pktsplit<4> k_split[2];
  pktsplit<8> value_split[6];
  pktmerge<8> output_merge[6];

  Llama3Gqa8GroupPacketGraph() {
    for (int lane = 0; lane < 2; ++lane) {
      const std::string suffix = std::to_string(lane);
      q[lane] = input_plio::create("packet_q" + suffix, plio_128_bits);
      k[lane] = input_plio::create("packet_k" + suffix, plio_128_bits);
      q_split[lane] = pktsplit<4>::create();
      k_split[lane] = pktsplit<4>::create();
      connect<pktstream>(q[lane].out[0], q_split[lane].in[0]);
      connect<pktstream>(k[lane].out[0], k_split[lane].in[0]);
    }
    for (int lane = 0; lane < 6; ++lane) {
      const std::string suffix = std::to_string(lane);
      value[lane] = input_plio::create(
          "packet_v" + suffix, plio_128_bits);
      output[lane] = output_plio::create(
          "packet_o" + suffix, plio_128_bits);
      value_split[lane] = pktsplit<8>::create();
      output_merge[lane] = pktmerge<8>::create();
      connect<pktstream>(value[lane].out[0], value_split[lane].in[0]);
      connect<pktstream>(output_merge[lane].out[0], output[lane].in[0]);
    }

    for (int gqa = 0; gqa < 8; ++gqa) {
      const int col_base = 4 * gqa;
      // V21 row 1: Score, PV5, PV4, Softmax.
      location<kernel>(group[gqa].score)    = tile(col_base + 0, 1);
      location<kernel>(group[gqa].value[5]) = tile(col_base + 1, 1);
      location<kernel>(group[gqa].value[4]) = tile(col_base + 2, 1);
      location<kernel>(group[gqa].softmax)  = tile(col_base + 3, 1);

      // Row 2 placement
      location<kernel>(group[gqa].value[3]) = tile(col_base + 0, 2);
      location<kernel>(group[gqa].value[2]) = tile(col_base + 1, 2);
      location<kernel>(group[gqa].value[1]) = tile(col_base + 2, 2);
      location<kernel>(group[gqa].value[0]) = tile(col_base + 3, 2);

      const int qk_lane = gqa / 4;
      const int lane_group = gqa % 4;
      connect<pktstream>(q_split[qk_lane].out[lane_group], group[gqa].q);
      connect<pktstream>(k_split[qk_lane].out[lane_group], group[gqa].k);
      for (int lane = 0; lane < 6; ++lane) {
        connect<pktstream>(value_split[lane].out[gqa],
                           group[gqa].value_slice[lane]);
      }
      connect<window<3072>, pktstream>(group[gqa].output_slice[0], output_merge[0].in[gqa]);
      connect<window<3072>, pktstream>(group[gqa].output_slice[1], output_merge[1].in[gqa]);
      connect<window<2048>, pktstream>(group[gqa].output_slice[2], output_merge[2].in[gqa]);
      connect<window<3072>, pktstream>(group[gqa].output_slice[3], output_merge[3].in[gqa]);
      connect<window<3072>, pktstream>(group[gqa].output_slice[4], output_merge[4].in[gqa]);
      connect<window<2048>, pktstream>(group[gqa].output_slice[5], output_merge[5].in[gqa]);
    }
  }
};

Llama3Gqa8GroupPacketGraph llama3_gqa_8group_packet;

#if defined(__AIESIM__) || defined(__X86SIM__)
int main() {
  llama3_gqa_8group_packet.init();
  llama3_gqa_8group_packet.run(2);
  llama3_gqa_8group_packet.end();
  return 0;
}
#endif
