#!/usr/bin/env python3
"""Dependency-free online-softmax and causal-GQA correctness test.

The production kernel uses Llama3-8B dimensions (32 Q heads, 8 KV heads,
D=128). This small test keeps the same GQA/causal semantics and checks the
blockwise recurrence against a naive exact implementation.
"""

import math
import random


def dot(left, right):
    return sum(a * b for a, b in zip(left, right))


def naive_attention(q, k, v, group_size):
    output = []
    scale = 1.0 / math.sqrt(len(q[0][0]))
    for q_head, q_rows in enumerate(q):
        kv_head = q_head // group_size
        output_rows = []
        for row, q_row in enumerate(q_rows):
            scores = [dot(q_row, k[kv_head][col]) * scale if col <= row else -math.inf
                      for col in range(len(q_rows))]
            maximum = max(scores)
            weights = [math.exp(score - maximum) if score != -math.inf else 0.0
                       for score in scores]
            denominator = sum(weights)
            output_rows.append([
                sum(weights[col] * v[kv_head][col][dim] for col in range(len(weights))) / denominator
                for dim in range(len(q_row))
            ])
        output.append(output_rows)
    return output


def online_attention(q, k, v, group_size, key_block):
    output = []
    scale = 1.0 / math.sqrt(len(q[0][0]))
    sequence = len(q[0])
    for q_head, q_rows in enumerate(q):
        kv_head = q_head // group_size
        output_rows = []
        for row, q_row in enumerate(q_rows):
            row_max = -math.inf
            row_sum = 0.0
            numerator = [0.0] * len(q_row)
            for key_start in range(0, sequence, key_block):
                key_end = min(key_start + key_block, sequence)
                scores = [dot(q_row, k[kv_head][col]) * scale if col <= row else -math.inf
                          for col in range(key_start, key_end)]
                tile_max = max(scores)
                next_max = max(row_max, tile_max)
                history_scale = 0.0 if row_max == -math.inf else math.exp(row_max - next_max)
                probabilities = [math.exp(score - next_max) if score != -math.inf else 0.0
                                 for score in scores]
                numerator = [
                    history_scale * numerator[dim] +
                    sum(probabilities[col - key_start] * v[kv_head][col][dim]
                        for col in range(key_start, key_end))
                    for dim in range(len(q_row))
                ]
                row_sum = history_scale * row_sum + sum(probabilities)
                row_max = next_max
            output_rows.append([value / row_sum for value in numerator])
        output.append(output_rows)
    return output


def main():
    random.seed(7)
    q_heads, kv_heads, sequence, dim = 4, 1, 11, 8
    group_size = q_heads // kv_heads
    sample = lambda: random.uniform(-1.0, 1.0)
    q = [[[sample() for _ in range(dim)] for _ in range(sequence)] for _ in range(q_heads)]
    k = [[[sample() for _ in range(dim)] for _ in range(sequence)] for _ in range(kv_heads)]
    v = [[[sample() for _ in range(dim)] for _ in range(sequence)] for _ in range(kv_heads)]

    expected = naive_attention(q, k, v, group_size)
    actual = online_attention(q, k, v, group_size, key_block=4)
    error = max(abs(expected[head][row][col] - actual[head][row][col])
                for head in range(q_heads)
                for row in range(sequence)
                for col in range(dim))
    print(f"max_abs_error={error:.12f}")
    assert error < 1e-10


if __name__ == "__main__":
    main()
