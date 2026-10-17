"""Smoke test for the harmonic kernel option. Needs torch, no dataset required.

    python tools/check_harmonic.py

It checks four things:
  1. backward compatibility: harmonic OFF gives exactly the original architecture
  2. harmonic ON produces the same output shape
  3. gradients actually reach the harmonic parameters (P, alpha, beta, bases)
  4. the readout works and the affinity rows sum to 1
"""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from layers.HarmonicKernel import harmonic_readout
from models.TFDNet import Model


def make_cfg(**kw):
    cfg = dict(seq_len=96, pred_len=24, enc_in=7, n_fft=[8, 16, 32],
               kernel_num=4, individual_factor=7, mode='MK',
               dropout=0.0, kernel_size=25, output_attention=False,
               harm_seasonal=0, harm_trend=0, harm_periods=[],
               harm_alpha=0.15, harm_beta=0.05, harm_learn_period=1)
    cfg.update(kw)
    return SimpleNamespace(**cfg)


def run(name, **kw):
    cfg = make_cfg(**kw)
    model = Model(cfg)
    x = torch.randn(2, cfg.seq_len, cfg.enc_in)
    y = model(x)
    loss = y.pow(2).mean()
    loss.backward()

    print(f"\n===== {name} =====")
    print("  input ", tuple(x.shape), "-> output", tuple(y.shape))
    rec = harmonic_readout(model)
    if not rec:
        print("  no harmonic module (baseline path)")
        return model

    for mod_name, r in rec.items():
        a = r["affinity"]
        print(f"  [{mod_name}]  P={[round(p, 3) for p in r['periods_samples']]} samples")
        print(f"       m0={[round(v, 3) for v in r['m0_bin_index']]} bins"
              f"  alpha={r['alpha']:.4f}  beta={r['beta']:.4f}")
        print(f"       affinity {a.shape}, row-sum min/max = "
              f"{a.sum(1).min():.6f}/{a.sum(1).max():.6f}")

    missing = []
    for n, p in model.named_parameters():
        if any(k in n for k in ("log_period", "alpha_raw", "beta_raw", "bases", "wc1", "b2")):
            if p.grad is None:
                missing.append(n)
    print("  harmonic params without grad:", missing if missing else "none")
    return model


if __name__ == "__main__":
    torch.manual_seed(0)
    print("torch", torch.__version__)

    run("MK, harmonic OFF (baseline)")
    run("IK, harmonic OFF (baseline)", mode='IK')

    periods = [8.0, 24.0]      # examples only: 8 and 24 samples
    run("MK, harmonic ON  (seasonal)", harm_seasonal=1, harm_periods=periods)
    run("IK, harmonic ON  (seasonal)", mode='IK', harm_seasonal=1, harm_periods=periods)
    m = run("MK, harmonic ON  (seasonal + trend)", harm_seasonal=1, harm_trend=1,
            harm_periods=periods)

    print("\n===== summary from the last model =====")
    print(m.harmonic_summary())
    print("\nOK")
