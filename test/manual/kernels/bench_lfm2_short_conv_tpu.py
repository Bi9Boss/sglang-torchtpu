# SPDX-License-Identifier: Apache-2.0
"""Compare compiled Pallas and torch LFM2 ShortConv, including cache writes.

  python test/manual/kernels/bench_lfm2_short_conv_tpu.py --channels 1536 \
      --batch 8 --length 1024 --mode both

Each sample measures host dispatch + device completion. Compilation, warmup
and state reset are excluded. This is wall time, not xprof device-only time.
"""

import argparse
import statistics
import time
from typing import Optional

import torch

from sglang_tpu.kernels.mamba.short_conv import build_short_conv_op


# Historical torch baseline, retained only for performance comparisons.
def _write_history(
    conv_state: torch.Tensor,
    slots: torch.Tensor,
    history: torch.Tensor,
    active: torch.Tensor,
) -> None:
    # Several padded requests can share slot 0. A batched index_copy_ would
    # have duplicate destinations, so serialize the fixed-size batch rows.
    # Inactive rows preserve their destination, including a dirty null slot.
    history = history.unsqueeze(2).to(conv_state.dtype)
    for row in range(slots.shape[0]):
        index = slots[row : row + 1]
        previous = conv_state.index_select(0, index)
        update = torch.where(
            active[row : row + 1].reshape(1, 1, 1, 1),
            history[row : row + 1],
            previous,
        )
        conv_state.index_copy_(0, index, update)


def short_conv_decode(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    conv_state: torch.Tensor,
    slots: torch.Tensor,
) -> torch.Tensor:
    """One token per request, including padded requests mapped to slot 0."""
    history = conv_state.index_select(0, slots).squeeze(2)
    window = torch.cat((history, x.float().unsqueeze(1)), dim=1)
    compute_weight = weight.float()
    output = torch.zeros_like(x, dtype=torch.float32)
    for k in range(weight.shape[1]):
        output = output + window[:, k] * compute_weight[:, k]
    if bias is not None:
        output = output + bias.float()

    active = slots > 0
    _write_history(conv_state, slots, window[:, 1:], active)
    return torch.where(active[:, None], output, 0).to(x.dtype)


def short_conv_prefill(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    conv_state: torch.Tensor,
    slots: torch.Tensor,
    query_start_loc: torch.Tensor,
    has_initial_state: torch.Tensor,
    token_request_indices: torch.Tensor,
    valid_tokens: torch.Tensor,
) -> torch.Tensor:
    """Packed prefill, with zero-length requests and trailing token padding.

    The caller builds token_request_indices once per forward, shared by all
    layers. Invalid tokens use request 0 and are masked before returning.
    """
    history = conv_state.index_select(0, slots).squeeze(2)
    history = torch.where(has_initial_state[:, None, None], history, 0)
    batch_size, history_len, channels = history.shape
    history_rows = batch_size * history_len
    bank = torch.cat((history.reshape(-1, channels), x.float()), dim=0)
    starts = query_start_loc[:-1]
    lengths = query_start_loc[1:] - starts
    positions = torch.arange(x.shape[0], dtype=torch.int64, device=x.device)
    positions = torch.where(valid_tokens, positions, 0)
    token_starts = starts.index_select(0, token_request_indices)
    compute_weight = weight.float()
    output = torch.zeros_like(x, dtype=torch.float32)

    for k in range(weight.shape[1]):
        relative = positions - token_starts - history_len + k
        indices = torch.where(
            relative < 0,
            token_request_indices * history_len + relative + history_len,
            history_rows + positions - history_len + k,
        )
        output = output + bank.index_select(0, indices) * compute_weight[:, k]
    if bias is not None:
        output = output + bias.float()

    tail_positions = (
        lengths[:, None]
        - history_len
        + torch.arange(history_len, dtype=torch.int64, device=x.device)
    )
    request_base = (
        torch.arange(batch_size, dtype=torch.int64, device=x.device)[:, None]
        * history_len
    )
    tail_indices = torch.where(
        tail_positions < 0,
        request_base + tail_positions + history_len,
        history_rows + starts[:, None] + tail_positions,
    )
    next_history = bank.index_select(0, tail_indices.reshape(-1)).reshape(
        batch_size, history_len, channels
    )
    _write_history(conv_state, slots, next_history, (slots > 0) & (lengths > 0))
    active_tokens = valid_tokens & (slots.index_select(0, token_request_indices) > 0)
    return torch.where(active_tokens[:, None], output, 0).to(x.dtype)


def _sync(tensors):
    from torch_tpu._internal.sync.sync import synchronize

    synchronize(tensors, wait=True)


def _time(fn, initial, iters):
    samples = []
    for _ in range(iters):
        state = initial.to("tpu")
        _sync([state])
        begin = time.perf_counter()
        out = fn(state)
        _sync([out, state])
        samples.append((time.perf_counter() - begin) * 1e6)
    return statistics.median(samples)


def bench(args, decode):
    device = torch.device("tpu")
    torch.manual_seed(0)
    length = 1 if decode else args.length
    tokens = args.batch * length
    x = torch.randn(tokens, args.channels).bfloat16().to(device)
    weight = torch.randn(args.channels, args.kernel_size).bfloat16().to(device)
    bias = torch.randn(args.channels).bfloat16().to(device)
    initial = torch.randn(args.slots, args.kernel_size - 1, 1, args.channels)
    slots = torch.arange(1, args.batch + 1, dtype=torch.int64, device=device)
    starts = torch.arange(0, tokens + 1, length, dtype=torch.int64, device=device)
    prefix = torch.ones(args.batch, dtype=torch.bool, device=device)
    requests = torch.arange(args.batch, device=device).repeat_interleave(length)
    valid = torch.ones(tokens, dtype=torch.bool, device=device)
    op = build_short_conv_op()

    def pallas(state):
        return op(x, weight, bias, state, slots, starts, prefix, decode)

    def reference(state):
        if decode:
            return short_conv_decode(x, weight, bias, state, slots)
        return short_conv_prefill(
            x, weight, bias, state, slots, starts, prefix, requests, valid
        )

    pallas_fn = torch.compile(pallas, backend="tpu", fullgraph=True, dynamic=False)
    reference_fn = torch.compile(
        reference, backend="tpu", fullgraph=True, dynamic=False
    )
    # Compare outputs and mutations before timing, and materialize every input
    # before donation. Independent host snapshots avoid lazy-clone aliasing.
    states = [initial.to(device), initial.to(device)]
    _sync([x, weight, bias, starts, slots, prefix, requests, valid, *states])
    actual, expected = pallas_fn(states[0]), reference_fn(states[1])
    _sync([actual, expected, *states])
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(states[0].cpu(), states[1].cpu(), rtol=0, atol=0)
    for fn in (pallas_fn, reference_fn):
        for _ in range(3):
            state = initial.to(device)
            _sync([state])
            out = fn(state)
            _sync([out, state])
    pallas_us = _time(pallas_fn, initial, args.iters)
    reference_us = _time(reference_fn, initial, args.iters)
    print(
        f"{'decode' if decode else 'prefill'} B={args.batch} T={tokens} "
        f"C={args.channels} K={args.kernel_size} slots={args.slots}: "
        f"Pallas={pallas_us:.2f} us, torch={reference_us:.2f} us, "
        f"speedup={reference_us / pallas_us:.2f}x (median, BF16, compiled)"
    )


def main():
    import torch_tpu  # noqa: F401

    parser = argparse.ArgumentParser()
    parser.add_argument("--channels", type=int, default=1536)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--slots", type=int, default=1024)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--mode", choices=("prefill", "decode", "both"), default="both")
    args = parser.parse_args()
    assert 0 < args.batch < args.slots
    for decode in (False, True):
        if args.mode == "both" or args.mode == ("decode" if decode else "prefill"):
            bench(args, decode)


if __name__ == "__main__":
    main()
