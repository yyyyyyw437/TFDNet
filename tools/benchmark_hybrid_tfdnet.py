"""Microbenchmark the frequency, patch, and channel components."""
import argparse
import os
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from models.TFDNetHybrid import Model


MODES = (
    ("frequency", 1, 0, 0),
    ("patch", 0, 1, 0),
    ("frequency_patch", 1, 1, 0),
    ("frequency_patch_channel", 1, 1, 1),
)


def make_cfg(args, freq, patch, channel):
    return SimpleNamespace(
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        enc_in=args.enc_in,
        dec_in=args.enc_in,
        c_out=args.enc_in,
        n_fft=args.n_fft,
        kernel_num=args.kernel_num,
        individual_factor=args.individual_factor,
        mode=args.mode,
        output_attention=False,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        use_freq_branch=freq,
        use_patch_branch=patch,
        use_channel_fusion=channel,
        patch_len=args.patch_len,
        patch_stride=args.patch_stride,
        patch_d_model=args.patch_d_model,
        patch_n_heads=args.patch_n_heads,
        patch_e_layers=args.patch_e_layers,
        patch_d_ff=args.patch_d_ff,
        channel_rank=args.channel_rank,
    )


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_peak_memory(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_mb(device):
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)


def benchmark(model, x, device, warmup, repeats):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        synchronize(device)
        reset_peak_memory(device)
        start = time.perf_counter()
        for _ in range(repeats):
            model(x)
        synchronize(device)
        forward_ms = (time.perf_counter() - start) * 1000.0 / repeats
    forward_memory = peak_memory_mb(device)

    model.train()
    for _ in range(warmup):
        model.zero_grad(set_to_none=True)
        model(x).square().mean().backward()
    synchronize(device)
    reset_peak_memory(device)
    start = time.perf_counter()
    for _ in range(repeats):
        model.zero_grad(set_to_none=True)
        model(x).square().mean().backward()
    synchronize(device)
    step_ms = (time.perf_counter() - start) * 1000.0 / repeats
    return {
        "params": sum(parameter.numel() for parameter in model.parameters()),
        "forward_ms": forward_ms,
        "step_ms": step_ms,
        "forward_peak_mb": forward_memory,
        "step_peak_mb": peak_memory_mb(device),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cuda", "cpu"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=336)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--enc_in", type=int, default=7)
    parser.add_argument("--n_fft", nargs="+", type=int, default=[8, 16, 32])
    parser.add_argument("--kernel_num", type=int, default=4)
    parser.add_argument("--individual_factor", type=int, default=7)
    parser.add_argument("--kernel_size", type=int, default=25)
    parser.add_argument("--mode", default="IK", choices=["IK", "MK"])
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--patch_len", type=int, default=16)
    parser.add_argument("--patch_stride", type=int, default=8)
    parser.add_argument("--patch_d_model", type=int, default=32)
    parser.add_argument("--patch_n_heads", type=int, default=4)
    parser.add_argument("--patch_e_layers", type=int, default=1)
    parser.add_argument("--patch_d_ff", type=int, default=64)
    parser.add_argument("--channel_rank", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2021)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    x = torch.randn(
        args.batch_size, args.seq_len, args.enc_in, device=device)
    print("device={} batch={} seq_len={} pred_len={}".format(
        device, args.batch_size, args.seq_len, args.pred_len))
    for name, freq, patch, channel in MODES:
        torch.manual_seed(args.seed)
        model = Model(make_cfg(args, freq, patch, channel)).float().to(device)
        result = benchmark(
            model, x, device, args.warmup, args.repeats)
        print(
            "{} params={} forward_ms={:.3f} step_ms={:.3f} "
            "forward_peak_mb={:.1f} step_peak_mb={:.1f}".format(
                name, result["params"], result["forward_ms"],
                result["step_ms"], result["forward_peak_mb"],
                result["step_peak_mb"]))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
