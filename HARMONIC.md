# Harmonic prior on the time-frequency kernels

This fork adds an **optional** harmonic-structure prior to the TFB kernels.
Default is OFF, so the original model is reproduced bit-for-bit.

## What changes

Original TFB: `M` independent `N x N` complex kernels per bank, `M*N^2` params.

With the prior: each bank is generated from `R+1` shared basis kernels

```
a[m, j]  = sum_h exp(-(m - h*m0_j)^2 / (2*(alpha*m0_j)^2))   j = 1..R   (harmonic families)
a[m, 0]  = beta                                              (free slot / fallback)
a[m, :] /= a[m, :].sum()                                     (row-normalised)
W[m]     = sum_j a[m, j] * B[j]                              (M, N, N)
m0_j     = n_fft / P_j                                       (P_j in SAMPLES)
```

Only `{B_j, P_j, alpha, beta}` are learned; the affinity `a` is computed.
`M*N^2` -> `(R+1)*N^2` parameters per bank.

It works for all three TFB variants:

| variant | how the prior is applied |
|---|---|
| Trend-TFB | one bank: `W[m] = sum_j a[m,j] B_j` |
| Seasonal-IK | `W_ind(i)[m] = sum_f W1[i,f] * sum_j a[m,j] B2[f,j]` (rank over channels x harmonic over bins) |
| Seasonal-MK | each of the `k` banks gets its own bases, but **shares `a`** (the harmonic structure is a property of the data, not of expert `k`) |

`a` is shared across channels and across the `k` MK dictionaries, and `P_j` is in
samples so one value serves all scales `S in {8,16,32}`.

This does **not** create cross-bin information flow: `Q[m]` still depends only on
`x_m`.  Cross-bin mixing remains the job of Frequency-FFN.

## Flags

| flag | default | meaning |
|---|---|---|
| `--harm_seasonal` | `0` | `1` = harmonic prior on the seasonal TFB (MK and IK) |
| `--harm_trend` | `0` | `1` = harmonic prior on the trend TFB |
| `--harm_periods` | `[]` | initial fundamental periods **in samples** |
| `--harm_alpha` | `0.15` | comb width relative to harmonic spacing; small = sharp |
| `--harm_beta` | `0.05` | free-slot baseline weight (fallback for non-harmonic bins) |
| `--harm_learn_period` | `1` | learn `P_j` end-to-end (`0` = freeze) |

### Initial periods (`--harm_periods`), in samples

| dataset | sampling | daily | weekly |
|---|---|---|---|
| ETTh1 / ETTh2 | 1 h | 24 | 168 |
| ETTm1 / ETTm2 | 15 min | 96 | 672 |
| Electricity / Traffic | 1 h | 24 | 168 |
| Weather | 10 min | 144 | 1008 |
| ILI | 1 week | 52 (yearly) | — |

If `--harm_periods` is empty while harmonic is on, the model warns and falls back
to a single shared kernel (only the free slot exists) — useful as an extreme
baseline, not as the intended setting.

## Quick start

```bash
# 1) smoke test (no dataset needed, ~seconds)
python tools/check_harmonic.py

# 2) paired comparison: same settings, only the flag differs
bash scripts/harmonic_compare.sh
```

## Reading the result

```python
print(model.harmonic_summary())
# [TFDNet] encoder_list.0.block_seasonal.harm_bank.aff:
#   P=[24.31, 169.07] samples | m0=[1.32, 0.19] bins | alpha=0.142 | beta=0.031
```

Three read-outs, all falsifiable:

* `P_j` -> convert to a physical period and compare with known cycles
  (24 h / 168 h / 52 weeks).  A match means the model found real periodicity.
* `alpha` -> how sharp the comb is.  Large `alpha` means the data has no usable
  harmonic structure (it degrades towards one shared kernel).
* `beta` -> how many bins fall back to the free slot, i.e. no harmonic to bind.

`layers.HarmonicKernel.harmonic_readout(model)` returns the full `(M, R+1)`
affinity matrix if you want to plot the comb.

## Caveats

* The STFT in this repo uses `n_fft = S` (no zero-padding) and a rectangular
  window.  Periods shorter than `S` are resolvable; longer ones fall below bin 1
  and are treated as trend.  So the prior can only bite on in-window oscillations.
* The gate of MK is unchanged (still a scalar per channel and kernel), so the
  harmonic prior and the gate are independent knobs.
