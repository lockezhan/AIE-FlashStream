#pragma once

#include "ap_int.h"

namespace llama3_attn {

// A fixed 128-bit stream word carries eight BF16 values or four FP32 values.
// The packet header permits the PL shell to multiplex all eight GQA groups
// over a bounded number of AIE shim channels.
enum class TileKind : unsigned char {
  kQSlice = 0,
  kKSlice = 1,
  kValueSlice = 2,
  kMask = 3,
  kOnlineState = 4,
  kNumerator = 5,
  kStateNext = 6,
  kNumeratorNext = 7,
};

struct TileHeader {
  ap_uint<4> kind;
  ap_uint<3> gqa_group;
  ap_uint<2> lane;
  ap_uint<12> query_block;
  ap_uint<12> key_block;
  ap_uint<1> last;
};

// m/l is kept in PL URAM between key blocks; alpha is produced by AIE and is
// consumed inside every value lane, so PL only forwards these state windows.
struct OnlineState {
  float max_value;
  float denominator;
};

}  // namespace llama3_attn
