# CrabNet-Mineral v2

Predict mineral assemblage and per-mineral chemistry from bulk ICP/XRF data, without requiring temperature, pressure, or spot analyses.

## What it does (one line)

Bulk oxide wt% in, calibrated posterior over assemblages out, with per-mineral oxide compositions, solid-solution endmembers, IUGS rock class, and assemblage Gibbs energy.

## Status

Pre-alpha. See `docs/POSITION_C.md` for the design and `CLAUDE.md` for the development context.

## Install

```bash
conda env update -f environment.yml -n alignn_fix
conda activate alignn_fix
pip install -e .
```
