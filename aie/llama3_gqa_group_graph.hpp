#pragma once

#include <adf.h>

using namespace adf;

void llama3_score_int8_lane0_1q(input_stream<int32>*,
                                input_stream<int32>*,
                                output_stream<float>*);
void llama3_score_int8_lane1_1q(input_stream<int32>*,
                                input_stream<int32>*,
                                output_stream<float>*);
void llama3_fused_softmax_2lane(input_stream<float>*,
                                input_stream<float>*,
                                output_stream<float>*);
void llama3_value_d28_lane0_bf16_2q(
    input_stream<float>*, input_stream<uint32>*, output_window<uint16>*);
void llama3_value_d28_lane1_bf16_2q(
    input_stream<float>*, input_stream<uint32>*, output_window<uint16>*);
void llama3_value_d24_lane2_bf16_2q(
    input_stream<float>*, input_stream<uint32>*, output_window<uint16>*);
void llama3_value_d24_lane3_bf16_2q(
    input_stream<float>*, input_stream<uint32>*, output_window<uint16>*);
void llama3_value_d24_lane4_bf16_2q(
    input_stream<float>*, input_stream<uint32>*, output_window<uint16>*);

class Llama3GqaGroupGraph : public graph {
 public:
  port<input> q[2];
  port<input> k[2];
  port<input> value_slice[5];
  port<output> output_slice[5];

  kernel score[2];
  kernel softmax;
  kernel value[5];

  Llama3GqaGroupGraph() {
    score[0] = kernel::create(llama3_score_int8_lane0_1q);
    score[1] = kernel::create(llama3_score_int8_lane1_1q);
    source(score[0]) = "llama3_score_int8_lane0_kernel.cpp";
    source(score[1]) = "llama3_score_int8_lane1_kernel.cpp";
    for (int lane = 0; lane < 2; ++lane) {
      runtime<ratio>(score[lane]) = 1;
      connect<stream>(q[lane], score[lane].in[0]);
      connect<stream>(k[lane], score[lane].in[1]);
    }

    softmax = kernel::create(llama3_fused_softmax_2lane);
    source(softmax) = "llama3_fused_softmax_kernel.cpp";
    runtime<ratio>(softmax) = 1;
    connect<stream>(score[0].out[0], softmax.in[0]);
    connect<stream>(score[1].out[0], softmax.in[1]);

    value[0] = kernel::create(llama3_value_d28_lane0_bf16_2q);
    value[1] = kernel::create(llama3_value_d28_lane1_bf16_2q);
    value[2] = kernel::create(llama3_value_d24_lane2_bf16_2q);
    value[3] = kernel::create(llama3_value_d24_lane3_bf16_2q);
    value[4] = kernel::create(llama3_value_d24_lane4_bf16_2q);
    for (int i = 0; i < 5; ++i) {
      source(value[i]) = "llama3_value_kernel.cpp";
      runtime<ratio>(value[i]) = 1;
      connect<stream>(softmax.out[0], value[i].in[0]);
      connect<stream>(value_slice[i], value[i].in[1]);
    }

    connect<window<3584>>(value[0].out[0], output_slice[0]);
    connect<window<3584>>(value[1].out[0], output_slice[1]);
    connect<window<3072>>(value[2].out[0], output_slice[2]);
    connect<window<3072>>(value[3].out[0], output_slice[3]);
    connect<window<3072>>(value[4].out[0], output_slice[4]);
  }
};
