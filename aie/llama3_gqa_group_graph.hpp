#pragma once

#include <adf.h>

using namespace adf;

void llama3_score_int8_2q(input_stream<int32>*, input_stream<int32>*,
                          output_stream<float>*);
void llama3_fused_softmax_1lane_2q(input_stream<float>*,
                                   output_stream<float>*);
void llama3_value_d24_lane0_bf16_2q(
    input_stream<float>*, input_stream<uint32>*, output_window<uint16>*);
void llama3_value_d24_lane1_bf16_2q(
    input_stream<float>*, input_stream<uint32>*, output_window<uint16>*);
void llama3_value_d16_lane2_bf16_2q(
    input_stream<float>*, input_stream<uint32>*, output_window<uint16>*);
void llama3_value_d24_lane3_bf16_2q(
    input_stream<float>*, input_stream<uint32>*, output_window<uint16>*);
void llama3_value_d24_lane4_bf16_2q(
    input_stream<float>*, input_stream<uint32>*, output_window<uint16>*);
void llama3_value_d16_lane5_bf16_2q(
    input_stream<float>*, input_stream<uint32>*, output_window<uint16>*);

class Llama3GqaGroupGraph : public graph {
 public:
  port<input> q;
  port<input> k;
  port<input> value_slice[6];
  port<output> output_slice[6];

  kernel score;
  kernel softmax;
  kernel value[6];

  Llama3GqaGroupGraph() {
    score = kernel::create(llama3_score_int8_2q);
    source(score) = "llama3_score_int8_2q_kernel.cpp";
    runtime<ratio>(score) = 1;
    stack_size(score) = 2048;
    connect<stream>(q, score.in[0]);
    connect<stream>(k, score.in[1]);

    softmax = kernel::create(llama3_fused_softmax_1lane_2q);
    source(softmax) = "llama3_fused_softmax_kernel.cpp";
    runtime<ratio>(softmax) = 1;
    stack_size(softmax) = 2048;
    connect<stream>(score.out[0], softmax.in[0]);

    value[0] = kernel::create(llama3_value_d24_lane0_bf16_2q);
    value[1] = kernel::create(llama3_value_d24_lane1_bf16_2q);
    value[2] = kernel::create(llama3_value_d16_lane2_bf16_2q);
    value[3] = kernel::create(llama3_value_d24_lane3_bf16_2q);
    value[4] = kernel::create(llama3_value_d24_lane4_bf16_2q);
    value[5] = kernel::create(llama3_value_d16_lane5_bf16_2q);
    for (int i = 0; i < 6; ++i) {
      source(value[i]) = "llama3_value_kernel.cpp";
      runtime<ratio>(value[i]) = 1;
      connect<stream>(softmax.out[0], value[i].in[0]);
      connect<stream>(value_slice[i], value[i].in[1]);
    }

    connect<window<3072>>(value[0].out[0], output_slice[0]);
    connect<window<3072>>(value[1].out[0], output_slice[1]);
    connect<window<2048>>(value[2].out[0], output_slice[2]);
    connect<window<3072>>(value[3].out[0], output_slice[3]);
    connect<window<3072>>(value[4].out[0], output_slice[4]);
    connect<window<2048>>(value[5].out[0], output_slice[5]);
  }
};
