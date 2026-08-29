#pragma once

#include <adf.h>

void llama3_score_int8_2q(input_stream<int32>* q,
                          input_stream<int32>* k,
                          output_stream<float>* score);
void llama3_fused_softmax_1lane_2q(
    input_stream<float>* score,
    output_stream<float>* exponent_and_reciprocal);
void llama3_value_d28_lane0_bf16_2q(input_stream<float>*,
                                     input_stream<uint32>*,
                                     output_window<uint16>*);
void llama3_value_d28_lane1_bf16_2q(input_stream<float>*,
                                     input_stream<uint32>*,
                                     output_window<uint16>*);
void llama3_value_d24_lane2_bf16_2q(input_stream<float>*,
                                     input_stream<uint32>*,
                                     output_window<uint16>*);
void llama3_value_d24_lane3_bf16_2q(input_stream<float>*,
                                     input_stream<uint32>*,
                                     output_window<uint16>*);
void llama3_value_d24_lane4_bf16_2q(input_stream<float>*,
                                     input_stream<uint32>*,
                                     output_window<uint16>*);
