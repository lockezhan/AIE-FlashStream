// Packetized 64-AIE Llama3 GQA shell.  The 19 physical AXIS ports are
// demultiplexed by pktsplit/pktmerge in the AIE graph using the KV-head ID.
#define llama3_attention_aie llama3_attention_aie_unused_reference
#include "llama3_attention_aie_shell.cpp"
#undef llama3_attention_aie

namespace {
using RawStream = hls::stream<ap_uint<128>>;

ap_uint<32> packet_header(unsigned id) {
#pragma HLS INLINE
  ap_uint<32> h = 0;
  h(4, 0) = id;
  h(11, 5) = 0;
  h(14, 12) = 0;
  h[15] = 0;
  // Match the Vitis 2022.2 packet_header(id, 0) encoding for PLIO input.
  h(20, 16) = 0;
  h(27, 21) = 0;
  h(30, 28) = 0;
  h[31] = h(30, 0).xor_reduce() ? ap_uint<1>(0) : ap_uint<1>(1);
  return h;
}

// The packet header consumes the first 32 bits.  The final beat supplies the
// delayed fourth 32-bit word and marks TLAST, matching the 2022.2 packet API.
void packetize(RawStream& raw, AxisStream& out, int words, unsigned id) {
  ap_uint<128> first = raw.read();
  AxisWord p;
  p.data.range(31, 0) = packet_header(id);
  p.data.range(127, 32) = first.range(95, 0);
  p.keep = -1; p.strb = -1; p.last = 0;
  out.write(p);
  ap_uint<32> delayed = first.range(127, 96);
  for (int i = 1; i < words; ++i) {
#pragma HLS PIPELINE II=1
    ap_uint<128> in = raw.read();
    p.data.range(31, 0) = delayed;
    p.data.range(127, 32) = in.range(95, 0);
    p.keep = -1; p.strb = -1; p.last = 0;
    out.write(p);
    delayed = in.range(127, 96);
  }
  p.data = 0;
  p.data.range(31, 0) = delayed;
  p.keep = 0x000f; p.strb = 0x000f; p.last = 1;
  out.write(p);
}

void depacketize(AxisStream& in, RawStream& raw, int words) {
  AxisWord previous = in.read(); // Header is in lane zero.
  for (int i = 0; i < words; ++i) {
#pragma HLS PIPELINE II=1
    AxisWord current = in.read();
    ap_uint<128> payload;
    payload.range(95, 0) = previous.data.range(127, 32);
    payload.range(127, 96) = current.data.range(31, 0);
    raw.write(payload);
    previous = current;
  }
}

void send_q_raw(const ap_uint<16>* m, int b, int h, int qs, int qc, int s,
                int d, RawStream& raw) {
  for (int index = 0; index < kQueryBlock * 64; index += 8) {
#pragma HLS PIPELINE II=1
    ap_uint<128> x = 0;
    for (int lane = 0; lane < 8; ++lane) {
#pragma HLS UNROLL
      int e = index + lane; int row = e / 64; int dim = d + e % 64;
      x.range(16*lane+15,16*lane) = row < qc ? m[q_offset(b,h,qs+row,dim,s)] : ap_uint<16>(0);
    }
    raw.write(x);
  }
}
void send_k_raw(const ap_uint<16> t[kKeyBlock][kHeadDim], int d, RawStream& raw) {
  for (int index = 0; index < kKeyBlock * 64; index += 8) {
#pragma HLS PIPELINE II=1
    ap_uint<128> x = 0; for (int lane = 0; lane < 8; ++lane) {
#pragma HLS UNROLL
      int e=index+lane; x.range(16*lane+15,16*lane)=t[e/64][d+e%64]; }
    raw.write(x);
  }
}
void send_mask_raw(int qs, int qc, int ks, int kc, RawStream& raw) {
  for (int index = 0; index < kQueryBlock*kKeyBlock; index += 4) {
#pragma HLS PIPELINE II=1
    ap_uint<128> x=0; for (int lane=0;lane<4;++lane) {
#pragma HLS UNROLL
      int e=index+lane, row=e/kKeyBlock, col=e%kKeyBlock; union { float f; unsigned u; } c;
      c.f=(row<qc && col<kc && ks+col<=qs+row)?0.0F:-3.402823466e+38F;
      x.range(32*lane+31,32*lane)=c.u; } raw.write(x);
  }
}
void send_floats_raw(const float* source, int count, RawStream& raw) {
  for (int index=0; index<count; index+=4) {
#pragma HLS PIPELINE II=1
    ap_uint<128> x=0; for(int lane=0;lane<4;++lane) { union { float f; unsigned u; } c; c.f=source[index+lane]; x.range(32*lane+31,32*lane)=c.u; } raw.write(x);
  }
}
void send_value_raw(const ap_uint<16> t[kKeyBlock][kHeadDim], int d, RawStream& raw) {
  for (int index=0;index<kKeyBlock*32;index+=8) {
#pragma HLS PIPELINE II=1
    ap_uint<128> x=0; for(int lane=0;lane<8;++lane) {
#pragma HLS UNROLL
      int e=index+lane; x.range(16*lane+15,16*lane)=t[e/32][d+e%32]; } raw.write(x);
  }
}
void receive_state_raw(RawStream& raw, float x[kQueryBlock][2]) {
  for(int index=0;index<kQueryBlock*3;index+=4) { ap_uint<128> a=raw.read(); for(int lane=0;lane<4;++lane) {
#pragma HLS PIPELINE II=1
    int e=index+lane; if(e%3!=2) { union { float f; unsigned u; } c; c.u=a.range(32*lane+31,32*lane); x[e/3][e%3]=c.f; } } }
}
void receive_numerator_raw(RawStream& raw, float x[kQueryBlock][32]) {
  for(int index=0;index<kQueryBlock*32;index+=4) { ap_uint<128> a=raw.read(); for(int lane=0;lane<4;++lane) {
#pragma HLS PIPELINE II=1
    union { float f; unsigned u; } c; c.u=a.range(32*lane+31,32*lane); int e=index+lane; x[e/32][e%32]=c.f; } }
}
void send_q_packet(const ap_uint<16>* m, int b, int h, int qs, int qc, int s,
                   int d, AxisStream& p, unsigned id) {
  RawStream raw;
#pragma HLS STREAM variable=raw depth=512
  send_q_raw(m, b, h, qs, qc, s, d, raw); packetize(raw, p, 512, id);
}
void send_k_packet(const ap_uint<16> t[kKeyBlock][kHeadDim], int d,
                   AxisStream& p, unsigned id) {
  RawStream raw;
#pragma HLS STREAM variable=raw depth=512
  send_k_raw(t, d, raw); packetize(raw, p, 512, id);
}
void send_mask_packet(int qs, int qc, int ks, int kc, AxisStream& p, unsigned id) {
  RawStream raw;
#pragma HLS STREAM variable=raw depth=512
  send_mask_raw(qs, qc, ks, kc, raw); packetize(raw, p, 512, id);
}
void send_state_packet(const float x[kQueryBlock][2], AxisStream& p, unsigned id) {
  RawStream raw;
#pragma HLS STREAM variable=raw depth=16
  send_floats_raw(&x[0][0], kQueryBlock*2, raw); packetize(raw, p, 16, id);
}
void send_value_packet(const ap_uint<16> t[kKeyBlock][kHeadDim], int d,
                       AxisStream& p, unsigned id) {
  RawStream raw;
#pragma HLS STREAM variable=raw depth=256
  send_value_raw(t, d, raw); packetize(raw, p, 256, id);
}
void send_numerator_packet(const float x[kQueryBlock][32], AxisStream& p, unsigned id) {
  RawStream raw;
#pragma HLS STREAM variable=raw depth=256
  send_floats_raw(&x[0][0], kQueryBlock*32, raw); packetize(raw, p, 256, id);
}
void receive_state_packet(AxisStream& p, float x[kQueryBlock][2]) {
  RawStream raw;
#pragma HLS STREAM variable=raw depth=24
  depacketize(p, raw, 24); receive_state_raw(raw, x);
}
void receive_numerator_packet(AxisStream& p, float x[kQueryBlock][32]) {
  RawStream raw;
#pragma HLS STREAM variable=raw depth=256
  depacketize(p, raw, 256); receive_numerator_raw(raw, x);
}

struct PacketPorts {
  AxisStream& q0; AxisStream& q1; AxisStream& k0; AxisStream& k1;
  AxisStream& mask; AxisStream& state_in; AxisStream& v0; AxisStream& v1;
  AxisStream& v2; AxisStream& v3; AxisStream& n0; AxisStream& n1;
  AxisStream& n2; AxisStream& n3; AxisStream& state_out; AxisStream& no0;
  AxisStream& no1; AxisStream& no2; AxisStream& no3;
};

void run_packet_group(const ap_uint<16>* q, const ap_uint<16>* k,
                      const ap_uint<16>* v, ap_uint<16>* o, int batch,
                      int kv_head, int seq_len, PacketPorts p) {
  for (int qi = 0; qi < kQueriesPerKvHead; ++qi) {
    const int q_head = kv_head * kQueriesPerKvHead + qi;
    for (int qs = 0; qs < seq_len; qs += kQueryBlock) {
      const int qc = (seq_len - qs < kQueryBlock) ? seq_len - qs : kQueryBlock;
      float state[kQueryBlock][2]; float numerator[4][kQueryBlock][32];
#pragma HLS BIND_STORAGE variable=numerator type=RAM_2P impl=URAM
#pragma HLS ARRAY_PARTITION variable=numerator complete dim=1
      for (int r = 0; r < kQueryBlock; ++r) { state[r][0] = -3.402823466e+38F; state[r][1] = 0.0F; }
      for (int l = 0; l < 4; ++l) for (int r = 0; r < kQueryBlock; ++r)
        for (int d = 0; d < 32; ++d) {
#pragma HLS PIPELINE II=1
          numerator[l][r][d] = 0.0F;
        }
      for (int ks = 0; ks < seq_len; ks += kKeyBlock) {
        const int kc = (seq_len - ks < kKeyBlock) ? seq_len - ks : kKeyBlock;
        ap_uint<16> kt[kKeyBlock][kHeadDim], vt[kKeyBlock][kHeadDim];
#pragma HLS BIND_STORAGE variable=kt type=RAM_2P impl=URAM
#pragma HLS BIND_STORAGE variable=vt type=RAM_2P impl=URAM
        load_kv_tile(k, v, batch, kv_head, ks, kc, seq_len, kt, vt);
        send_q_packet(q,batch,q_head,qs,qc,seq_len,0,p.q0,kv_head);
        send_q_packet(q,batch,q_head,qs,qc,seq_len,64,p.q1,kv_head);
        send_k_packet(kt,0,p.k0,kv_head); send_k_packet(kt,64,p.k1,kv_head);
        send_mask_packet(qs,qc,ks,kc,p.mask,kv_head); send_state_packet(state,p.state_in,kv_head);
        send_value_packet(vt,0,p.v0,kv_head); send_value_packet(vt,32,p.v1,kv_head);
        send_value_packet(vt,64,p.v2,kv_head); send_value_packet(vt,96,p.v3,kv_head);
        send_numerator_packet(numerator[0],p.n0,kv_head); send_numerator_packet(numerator[1],p.n1,kv_head);
        send_numerator_packet(numerator[2],p.n2,kv_head); send_numerator_packet(numerator[3],p.n3,kv_head);
        receive_state_packet(p.state_out,state); receive_numerator_packet(p.no0,numerator[0]);
        receive_numerator_packet(p.no1,numerator[1]); receive_numerator_packet(p.no2,numerator[2]); receive_numerator_packet(p.no3,numerator[3]);
      }
      for (int r = 0; r < qc; ++r) for (int l = 0; l < 4; ++l) for (int d = 0; d < 32; ++d) {
#pragma HLS PIPELINE II=1
        o[q_offset(batch,q_head,qs+r,l*32+d,seq_len)] = float_to_bf16(numerator[l][r][d] / state[r][1]);
      }
    }
  }
}

// pktmerge<8> emits a round of packets only after every input group has
// produced one.  Keep the online state for all KV heads in PL and issue a
// complete eight-group wave before reading any merged result.
void run_packet_wave(const ap_uint<16>* q, const ap_uint<16>* k,
                     const ap_uint<16>* v, int batch, int q_in_group,
                     int query_start, int query_count, int key_start,
                     int key_count, int seq_len, PacketPorts p,
                     float state[8][kQueryBlock][2],
                     float numerator[8][4][kQueryBlock][32]) {
  // pktmerge<8> cannot make progress when one group receives all fourteen
  // inputs before the other groups receive their first packet.  Stage K/V for
  // the complete wave, then issue every physical input port to all groups.
  // This makes each pktsplit and pktmerge observe the same group ordering.
  ap_uint<16> kt[8][kKeyBlock][kHeadDim];
  ap_uint<16> vt[8][kKeyBlock][kHeadDim];
#pragma HLS BIND_STORAGE variable=kt type=RAM_2P impl=URAM
#pragma HLS BIND_STORAGE variable=vt type=RAM_2P impl=URAM
  for (int group = 0; group < kKvHeads; ++group) {
    load_kv_tile(k, v, batch, group, key_start, key_count, seq_len,
                 kt[group], vt[group]);
  }
  for (int group = 0; group < kKvHeads; ++group) {
    const int q_head = group * kQueriesPerKvHead + q_in_group;
    send_q_packet(q,batch,q_head,query_start,query_count,seq_len,0,p.q0,group);
  }
  for (int group = 0; group < kKvHeads; ++group) {
    const int q_head = group * kQueriesPerKvHead + q_in_group;
    send_q_packet(q,batch,q_head,query_start,query_count,seq_len,64,p.q1,group);
  }
  for (int group = 0; group < kKvHeads; ++group)
    send_k_packet(kt[group],0,p.k0,group);
  for (int group = 0; group < kKvHeads; ++group)
    send_k_packet(kt[group],64,p.k1,group);
  for (int group = 0; group < kKvHeads; ++group)
    send_mask_packet(query_start,query_count,key_start,key_count,p.mask,group);
  for (int group = 0; group < kKvHeads; ++group)
    send_state_packet(state[group],p.state_in,group);
  for (int group = 0; group < kKvHeads; ++group)
    send_value_packet(vt[group],0,p.v0,group);
  for (int group = 0; group < kKvHeads; ++group)
    send_value_packet(vt[group],32,p.v1,group);
  for (int group = 0; group < kKvHeads; ++group)
    send_value_packet(vt[group],64,p.v2,group);
  for (int group = 0; group < kKvHeads; ++group)
    send_value_packet(vt[group],96,p.v3,group);
  for (int group = 0; group < kKvHeads; ++group)
    send_numerator_packet(numerator[group][0],p.n0,group);
  for (int group = 0; group < kKvHeads; ++group)
    send_numerator_packet(numerator[group][1],p.n1,group);
  for (int group = 0; group < kKvHeads; ++group)
    send_numerator_packet(numerator[group][2],p.n2,group);
  for (int group = 0; group < kKvHeads; ++group)
    send_numerator_packet(numerator[group][3],p.n3,group);
  for (int group = 0; group < kKvHeads; ++group) {
    receive_state_packet(p.state_out,state[group]);
    receive_numerator_packet(p.no0,numerator[group][0]);
    receive_numerator_packet(p.no1,numerator[group][1]);
    receive_numerator_packet(p.no2,numerator[group][2]);
    receive_numerator_packet(p.no3,numerator[group][3]);
  }
}
} // namespace

extern "C" void llama3_aie8_pkt(
    const ap_uint<16>* q, const ap_uint<16>* k, const ap_uint<16>* v, ap_uint<16>* o,
    int batch_size, int seq_len, AxisStream& packet_q0, AxisStream& packet_q1,
    AxisStream& packet_k0, AxisStream& packet_k1, AxisStream& packet_mask,
    AxisStream& packet_state_in, AxisStream& packet_v0, AxisStream& packet_v1,
    AxisStream& packet_v2, AxisStream& packet_v3, AxisStream& packet_numerator_in0,
    AxisStream& packet_numerator_in1, AxisStream& packet_numerator_in2, AxisStream& packet_numerator_in3,
    AxisStream& packet_state_out, AxisStream& packet_numerator_out0, AxisStream& packet_numerator_out1,
    AxisStream& packet_numerator_out2, AxisStream& packet_numerator_out3) {
#pragma HLS INTERFACE m_axi port=q offset=slave bundle=gmem0 max_read_burst_length=64
#pragma HLS INTERFACE m_axi port=k offset=slave bundle=gmem1 max_read_burst_length=64
#pragma HLS INTERFACE m_axi port=v offset=slave bundle=gmem2 max_read_burst_length=64
#pragma HLS INTERFACE m_axi port=o offset=slave bundle=gmem3 max_write_burst_length=64
#pragma HLS INTERFACE axis port=packet_q0
#pragma HLS INTERFACE axis port=packet_q1
#pragma HLS INTERFACE axis port=packet_k0
#pragma HLS INTERFACE axis port=packet_k1
#pragma HLS INTERFACE axis port=packet_mask
#pragma HLS INTERFACE axis port=packet_state_in
#pragma HLS INTERFACE axis port=packet_v0
#pragma HLS INTERFACE axis port=packet_v1
#pragma HLS INTERFACE axis port=packet_v2
#pragma HLS INTERFACE axis port=packet_v3
#pragma HLS INTERFACE axis port=packet_numerator_in0
#pragma HLS INTERFACE axis port=packet_numerator_in1
#pragma HLS INTERFACE axis port=packet_numerator_in2
#pragma HLS INTERFACE axis port=packet_numerator_in3
#pragma HLS INTERFACE axis port=packet_state_out
#pragma HLS INTERFACE axis port=packet_numerator_out0
#pragma HLS INTERFACE axis port=packet_numerator_out1
#pragma HLS INTERFACE axis port=packet_numerator_out2
#pragma HLS INTERFACE axis port=packet_numerator_out3
#pragma HLS INTERFACE s_axilite port=q bundle=control
#pragma HLS INTERFACE s_axilite port=k bundle=control
#pragma HLS INTERFACE s_axilite port=v bundle=control
#pragma HLS INTERFACE s_axilite port=o bundle=control
#pragma HLS INTERFACE s_axilite port=batch_size bundle=control
#pragma HLS INTERFACE s_axilite port=seq_len bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
  if (batch_size <= 0 || seq_len <= 0) return;
  PacketPorts p{packet_q0,packet_q1,packet_k0,packet_k1,packet_mask,packet_state_in,
                packet_v0,packet_v1,packet_v2,packet_v3,packet_numerator_in0,packet_numerator_in1,
                packet_numerator_in2,packet_numerator_in3,packet_state_out,packet_numerator_out0,
                packet_numerator_out1,packet_numerator_out2,packet_numerator_out3};
  for (int b = 0; b < batch_size; ++b) {
    for (int qi = 0; qi < kQueriesPerKvHead; ++qi) {
      for (int qs = 0; qs < seq_len; qs += kQueryBlock) {
        const int qc = (seq_len - qs < kQueryBlock) ? seq_len - qs : kQueryBlock;
        float state[8][kQueryBlock][2];
        float numerator[8][4][kQueryBlock][32];
#pragma HLS BIND_STORAGE variable=numerator type=RAM_2P impl=URAM
#pragma HLS ARRAY_PARTITION variable=numerator complete dim=2
        for (int g = 0; g < kKvHeads; ++g) {
          for (int r = 0; r < kQueryBlock; ++r) {
            state[g][r][0] = -3.402823466e+38F;
            state[g][r][1] = 0.0F;
          }
          for (int lane = 0; lane < 4; ++lane) for (int r = 0; r < kQueryBlock; ++r)
            for (int d = 0; d < 32; ++d) {
#pragma HLS PIPELINE II=1
              numerator[g][lane][r][d] = 0.0F;
            }
        }
        for (int ks = 0; ks < seq_len; ks += kKeyBlock) {
          const int kc = (seq_len - ks < kKeyBlock) ? seq_len - ks : kKeyBlock;
          run_packet_wave(q,k,v,b,qi,qs,qc,ks,kc,seq_len,p,state,numerator);
        }
        for (int g = 0; g < kKvHeads; ++g) {
          const int q_head = g * kQueriesPerKvHead + qi;
          for (int r = 0; r < qc; ++r) for (int lane = 0; lane < 4; ++lane)
            for (int d = 0; d < 32; ++d) {
#pragma HLS PIPELINE II=1
              o[q_offset(b,q_head,qs+r,lane*32+d,seq_len)] =
                  float_to_bf16(numerator[g][lane][r][d] / state[g][r][1]);
            }
        }
      }
    }
  }
}
