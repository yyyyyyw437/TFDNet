"""Microbenchmark dense and multi-scale convolution TFB TFDNet."""
import argparse
import os
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from models.TFDNet import Model as DenseModel
from models.TFDNetConv import Model as ConvModel


def make_cfg(args):
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
        tf_conv_kernels=args.tf_conv_kernels,
        tf_conv_dilations=args.tf_conv_dilations,
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
        forward_seconds = (time.perf_counter() - start) / repeats
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
    step_seconds = (time.perf_counter() - start) / repeats
    step_memory = peak_memory_mb(device)
    return {
        "forward_ms": forward_seconds * 1000.0,
        "step_ms": step_seconds * 1000.0,
        "forward_peak_mb": forward_memory,
        "step_peak_mb": step_memory,
        "params": sum(parameter.numel() for parameter in model.parameters()),
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
    parser.add_argument("--mode", default="MK", choices=["IK", "MK"])
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--tf_conv_kernels", nargs="+", type=int,
                        default=[3, 5, 9])
    parser.add_argument("--tf_conv_dilations", nargs="+", type=int,
                        default=[1, 2, 4])
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

    torch.manual_seed(args.seed)
    x = torch.randn(
        args.batch_size, args.seq_len, args.enc_in, device=device)
    cfg = make_cfg(args)
    results = {}
    for name, model_class in (
            ("dense", DenseModel), ("conv", ConvModel)):
        torch.manual_seed(args.seed)
        model = model_class(cfg).float().to(device)
        results[name] = benchmark(
            model, x, device, args.warmup, args.repeats)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    dense = results["dense"]
    conv = results["conv"]
    print("device={} batch={} seq_len={} pred_len={} mode={}".format(
        device, args.batch_size, args.seq_len, args.pred_len, args.mode))
    for name in ("dense", "conv"):
        values = results[name]
        print(
            "{} params={} forward_ms={:.3f} step_ms={:.3f} "
            "forward_peak_mb={:.1f} step_peak_mb={:.1f}".format(
                name, values["params"], values["forward_ms"],
                values["step_ms"], values["forward_peak_mb"],
                values["step_peak_mb"]))
    print("ratios conv/dense: params={:.4f} forward={:.4f} step={:.4f}".format(
        conv["params"] / dense["params"],
        conv["forward_ms"] / dense["forward_ms"],
        conv["step_ms"] / dense["step_ms"]))


if __name__ == "__main__":
    main()
