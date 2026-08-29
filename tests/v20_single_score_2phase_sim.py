#!/usr/bin/env python3
"""Generate and verify full-graph V20 packet-stream simulation data."""

import argparse
import json
import math
import random
import struct
from pathlib import Path

ROWS = 32
HEAD_D = 128
GROUPS = 8
QUERIES_PER_GROUP = 4
SLICES = ((0, 28), (28, 28), (56, 24), (80, 24), (104, 24))
OCTET_ORDER = (0, 1, 4, 5, 8, 9, 12, 13,
               2, 3, 6, 7, 10, 11, 14, 15)
QUANT_GROUP_DIMS = tuple(
    tuple(8 * octet + lane
          for octet in OCTET_ORDER[group * 8:(group + 1) * 8]
          for lane in range(8))
    for group in range(2))
INV_SQRT_D = 0.08838834764831843


def float_to_bf16_bits(value):
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    bits += 0x7FFF + ((bits >> 16) & 1)
    return (bits >> 16) & 0xFFFF


def bf16_bits_to_float(bits):
    return struct.unpack("<f", struct.pack("<I", (bits & 0xFFFF) << 16))[0]


def to_s32(value):
    value &= 0xFFFFFFFF
    return value if value < 0x80000000 else value - 0x100000000


def packet_header(packet_id):
    # UG1079 simulator convention for a packet originating outside the ADF
    # graph: source row/column are -1/-1, giving the 0x0fff0000 base.
    lower = 0x0FFF0000 | (packet_id & 0x1F)
    parity = 0 if bin(lower).count("1") & 1 else 1
    return lower | (parity << 31)


def pack_u16(values):
    assert len(values) % 2 == 0
    return [values[index] | (values[index + 1] << 16)
            for index in range(0, len(values), 2)]


def pack_i8(values):
    assert len(values) % 4 == 0
    packed = []
    for index in range(0, len(values), 4):
        word = 0
        for lane, value in enumerate(values[index:index + 4]):
            word |= (value & 0xFF) << (8 * lane)
        packed.append(word)
    return packed


def quantize_matrix(matrix):
    """Match the two 64-dimension group-max quantizers in the PL shell."""
    bf16 = [[float_to_bf16_bits(value) for value in row] for row in matrix]
    quantized = []
    maxima = []
    scales = []
    reciprocal = [round((127 << 15) / mantissa)
                  for mantissa in range(128, 256)]
    for row_bits in bf16:
        row_q = [0] * HEAD_D
        for group in range(2):
            dimensions = QUANT_GROUP_DIMS[group]
            values = [row_bits[dim] for dim in dimensions]
            maximum = max(value & 0x7FFF for value in values)
            maxima.append(maximum)
            scales.append(bf16_bits_to_float(maximum) / 127.0)
            max_exp = (maximum >> 7) & 0xFF
            max_mantissa = 128 | (maximum & 0x7F)
            for dim, bits in zip(dimensions, values):
                magnitude = bits & 0x7FFF
                exponent = (bits >> 7) & 0xFF
                mantissa = 128 | (bits & 0x7F)
                if magnitude == 0 or exponent == 0 or max_exp == 0:
                    value = 0
                else:
                    shift = 15 + max_exp - exponent
                    product = mantissa * reciprocal[max_mantissa - 128]
                    value = 0 if shift >= 24 else (
                        product + (1 << (shift - 1))) >> shift
                    value = min(value, 127)
                row_q[dim] = -value if bits & 0x8000 else value
        quantized.append(row_q)
    return bf16, quantized, maxima, scales


def q_payload(quantized, maxima):
    words = pack_u16(maxima)
    for row_pair in range(ROWS // 2):
        row0 = quantized[2 * row_pair]
        row1 = quantized[2 * row_pair + 1]
        for dim_octet in OCTET_ORDER:
            base = 8 * dim_octet
            words.extend(pack_i8(row0[base:base + 8] +
                                 row1[base:base + 8]))
    assert len(words) == 1056
    return words


def k_payload(quantized, maxima):
    words = pack_u16(maxima)
    for key_octet in range(ROWS // 8):
        for dim_octet in OCTET_ORDER:
            tiled = []
            for dim in range(8):
                for key in range(8):
                    tiled.append(
                        quantized[key_octet * 8 + key][dim_octet * 8 + dim])
            words.extend(pack_i8(tiled))
    assert len(words) == 1056
    return words


def v_payload(value_bits, offset, width):
    words = []
    for key_quad in range(ROWS // 4):
        for dim_quad in range(width // 4):
            tile = []
            for key_lane in range(4):
                for dim_lane in range(4):
                    tile.append(value_bits[4 * key_quad + key_lane]
                                [offset + 4 * dim_quad + dim_lane])
            words.extend(pack_u16(tile))
    assert len(words) == ROWS * width // 2
    return words


def write_packet_file(path, packets):
    """Write a 32-bit packet PLIO using AMD's header/TLAST text format."""
    with path.open("w", encoding="ascii") as output:
        for packet_id, payload in packets:
            output.write(str(packet_header(packet_id)) + "\n")
            for index, value in enumerate(payload):
                if index + 1 == len(payload):
                    output.write("TLAST\n")
                output.write(str(to_s32(value)) + "\n")


def reference_attention(q_quantized, q_scales, k_quantized, k_scales,
                        value):
    result = []
    for row in range(ROWS):
        scores = []
        for key in range(row + 1):
            dot = 0.0
            for group in range(2):
                integer_dot = sum(
                    q_quantized[row][dim] * k_quantized[key][dim]
                    for dim in QUANT_GROUP_DIMS[group])
                dot += integer_dot * q_scales[2 * row + group] * \
                    k_scales[2 * key + group]
            scores.append(dot * INV_SQRT_D)
        maximum = max(scores)
        weights = [math.exp(score - maximum) for score in scores]
        denominator = sum(weights)
        result.append([
            sum(weights[key] * value[key][dim]
                for key in range(row + 1)) / denominator
            for dim in range(HEAD_D)
        ])
    return result


def generate(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20260829)
    queries = []
    keys = []
    values = []
    q_payloads = []
    k_payloads = []
    expected = [None] * (GROUPS * QUERIES_PER_GROUP)

    for group in range(GROUPS):
        key = [[rng.uniform(-0.5, 0.5) for _ in range(HEAD_D)]
               for _ in range(ROWS)]
        # An identity prefix makes the first 32 output dimensions expose the
        # complete causal probability row, so the end-to-end test checks the
        # score/softmax path directly as well as all five PV slices.
        value = [[1.0 if key == dim else 0.0 for dim in range(HEAD_D)]
                 for key in range(ROWS)]
        value_bits = [[float_to_bf16_bits(item) for item in row]
                      for row in value]
        value_bf16 = [[bf16_bits_to_float(item) for item in row]
                      for row in value_bits]
        _, key_q, key_max, key_scales = quantize_matrix(key)
        keys.append((key_q, key_scales))
        values.append((value_bits, value_bf16))
        k_payloads.append(k_payload(key_q, key_max))

        group_queries = []
        for query in range(QUERIES_PER_GROUP):
            matrix = [[rng.uniform(-0.5, 0.5) for _ in range(HEAD_D)]
                      for _ in range(ROWS)]
            _, query_q, query_max, query_scales = quantize_matrix(matrix)
            q_payloads.append(q_payload(query_q, query_max))
            group_queries.append((query_q, query_scales))
            q_head = group * QUERIES_PER_GROUP + query
            expected[q_head] = reference_attention(
                query_q, query_scales, key_q, key_scales, value_bf16)
        queries.append(group_queries)

    for lane in range(2):
        q_packets = []
        for query in range(QUERIES_PER_GROUP):
            for local_group in range(4):
                group = lane * 4 + local_group
                q_packets.append((local_group,
                                  q_payloads[group * 4 + query]))
        write_packet_file(output_dir / f"packet_q{lane}.txt", q_packets)
        write_packet_file(
            output_dir / f"packet_k{lane}.txt",
            [(local_group, k_payloads[lane * 4 + local_group])
             for local_group in range(4)])

    for slice_index, (offset, width) in enumerate(SLICES):
        packets = []
        for group in range(GROUPS):
            packets.append((group,
                            v_payload(values[group][0], offset, width)))
        write_packet_file(output_dir / f"packet_v{slice_index}.txt", packets)

    (output_dir / "expected.json").write_text(
        json.dumps(expected), encoding="ascii")
    print("generated_q_packets_per_lane=16")
    print("generated_k_packets_per_lane=4")
    print("generated_v_packets_per_slice=8")


def read_packets(path):
    packets = []
    current = []
    next_is_last = False
    for raw_line in path.read_text(encoding="ascii").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.lower() == "tlast":
            next_is_last = True
            continue
        for token in line.split():
            current.append(int(token, 0) & 0xFFFFFFFF)
        if next_is_last:
            packets.append(current)
            current = []
            next_is_last = False
    if current:
        raise RuntimeError(f"unterminated output packet in {path}")
    return packets


def verify(output_dir):
    expected = json.loads((output_dir / "expected.json").read_text(
        encoding="ascii"))
    output_candidates = (
        output_dir,
        output_dir / "x86simulator_output",
        output_dir / "aiesimulator_output",
    )
    packet_dir = next(
        (candidate for candidate in output_candidates
         if (candidate / "packet_o0.txt").is_file()),
        None)
    if packet_dir is None:
        raise SystemExit(f"no simulator packet outputs below {output_dir}")
    actual = [[[None for _ in range(HEAD_D)] for _ in range(ROWS)]
              for _ in range(GROUPS * QUERIES_PER_GROUP)]
    packets_seen = [[0] * GROUPS for _ in SLICES]

    for slice_index, (offset, width) in enumerate(SLICES):
        packets = read_packets(packet_dir / f"packet_o{slice_index}.txt")
        if len(packets) != 16:
            raise SystemExit(
                f"slice {slice_index}: expected 16 packets, got {len(packets)}")
        payload_words = 2 * ROWS * width // 2
        for packet in packets:
            group = packet[0] & 0x1F
            if group >= GROUPS:
                raise SystemExit(f"slice {slice_index}: invalid group ID {group}")
            phase = packets_seen[slice_index][group]
            packets_seen[slice_index][group] += 1
            if phase >= 2:
                raise SystemExit(
                    f"slice {slice_index}: more than two packets for G{group}")
            payload = packet[1:1 + payload_words]
            if len(payload) != payload_words:
                raise SystemExit(f"slice {slice_index}: short output packet")
            bf16 = []
            for word in payload:
                bf16.append(word & 0xFFFF)
                bf16.append((word >> 16) & 0xFFFF)
            cursor = 0
            for query in range(2):
                q_head = group * 4 + phase * 2 + query
                for row in range(ROWS):
                    for dim in range(width):
                        actual[q_head][row][offset + dim] = \
                            bf16_bits_to_float(bf16[cursor])
                        cursor += 1

    for slice_index, seen in enumerate(packets_seen):
        if any(count != 2 for count in seen):
            raise SystemExit(
                f"slice {slice_index}: per-group packet counts are {seen}")

    maximum = 0.0
    maximum_location = None
    total = 0.0
    count = 0
    head_maximum = [0.0] * 32
    head_total = [0.0] * 32
    head_count = [0] * 32
    nan_count = 0
    inf_count = 0
    for q_head in range(32):
        for row in range(ROWS):
            for dim in range(HEAD_D):
                got = actual[q_head][row][dim]
                want = expected[q_head][row][dim]
                if got is None:
                    raise SystemExit(
                        f"missing output h={q_head} row={row} dim={dim}")
                nan_count += int(math.isnan(got))
                inf_count += int(math.isinf(got))
                if math.isfinite(got):
                    error = abs(got - want)
                    if error > maximum:
                        maximum = error
                        maximum_location = (q_head, row, dim, got, want)
                    total += error
                    count += 1
                    head_maximum[q_head] = max(head_maximum[q_head], error)
                    head_total[q_head] += error
                    head_count[q_head] += 1
    mean = total / count
    print(f"verified_elements={count}")
    print(f"nan_count={nan_count}")
    print(f"inf_count={inf_count}")
    print(f"max_abs_error={maximum:.9g}")
    print(f"mean_abs_error={mean:.9g}")
    if nan_count or inf_count or maximum > 0.008 or mean > 0.001:
        for q_head in range(32):
            print(f"head={q_head:02d} max={head_maximum[q_head]:.9g} "
                  f"mean={head_total[q_head] / head_count[q_head]:.9g}")
        print(f"maximum_location={maximum_location}")
        raise SystemExit("V20 full simulation numerical gate failed")
    print("V20 single-Score two-phase full simulation PASS")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("generate", "verify"))
    parser.add_argument("--dir", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "generate":
        generate(args.dir)
    else:
        verify(args.dir)


if __name__ == "__main__":
    main()
