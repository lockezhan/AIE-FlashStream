#include "llama3_score_int8_kernel_impl.hpp"

void llama3_score_int8_lane1_1q(input_stream<int32>* q,
                                 input_stream<int32>* k,
                                 output_stream<float>* score) {
  llama3_score_int8_impl<1>(q, k, score);
}
