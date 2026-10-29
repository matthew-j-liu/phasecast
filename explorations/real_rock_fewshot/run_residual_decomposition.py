"""Is the recalibrator's remaining per-group residual width, or location?

Fits the vector and matrix tiers on the committed train roster, then asks per
group how far a pure width knob (mean-preserving Beta precision scale) and a
width+location pair (precision scale plus a logit shift on the Beta mean) can
push 80% coverage.  A group whose width-only ceiling sits below nominal while
the 2-d oracle reaches it is location-limited: HPO on concentration-space
knobs cannot fix it, a mean-shifting term could.

    conda run -n mineral python explorations/real_rock_fewshot/run_residual_decomposition.py

Fit on train, diagnose on train and val; the test slice (spent 2026-08-03) is not touched.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.special import betaincinv, expit, logit
from scipy.stats import spearmanr

from cnmineral.data.phase_vocab import GROUPS
from cnmineral.eval.latent import bulk_latents, load_encoder_from_checkpoint
from cnmineral.eval.mode_head import beta_pit, conditional_mean
from cnmineral.eval.real_rock import (
    load_benchmark,
    share_interval,
    split_mask,
    unit_bootstrap,
)
from cnmineral.training import pick_device
from cnmineral.training.joint_heads import JointHeadCheckpoint
from cnmineral.training.real_rock_recalibrator import (
    build_group_predictions,
    fit_recalibrator,
    score_recalibrator,
)

RESULTS = Path(__file__).resolve().parent / "results"

# the values selected on val per tier; see run_curve.WEIGHT_DECAY
WEIGHT_DECAY = {"vector": 0.1, "matrix": 1.0}

TIERS_FIT = ("vector", "matrix")
NOMINAL_Q = (0.1, 0.9)
NOMINAL = 0.8

K_GRID = np.unique(np.concatenate([np.geomspace(1e-3, 30.0, 61), [1.0]]))
DELTA_GRID = np.linspace(-2.0, 2.0, 41)
DELTA_GRID[20] = 0.0

DEFAULT_CHECKPOINTS = (
    Path("checkpoints/joint_heads_regen_tuned_seed0/heads_best.pt"),
    Path("checkpoints/joint_heads_regen_seed0/heads_best.pt"),
)


def per_group_offsets(recalibrator) -> dict[str, dict[str, float]]:
    """The fitted constants, per group, for reading against the measured ones.

    Only meaningful for the vector tier, where a group's correction is a
    constant; the higher tiers couple groups and their offsets are not
    separable this way.
    """
    with torch.no_grad():
        return {
            group: {
                "presence_scale": float(recalibrator.presence_scale[i]),
                "presence_offset": float(recalibrator.presence_offset[i]),
                "magnitude_offset": float(recalibrator.magnitude_offset[i]),
            }
            for i, group in enumerate(GROUPS)
        }


def shifted_beta_quantiles(a, b, *, k=1.0, delta=0.0, q=NOMINAL_Q):
    """Quantiles of the width/location-modified Beta marginal.

    Mean-precision reparameterization: `m = a/(a+b)`, `nu = a+b`; the modified
    marginal is `Beta(k*nu*m', k*nu*(1-m'))` with `m' = sigmoid(logit(m) +
    delta)`.  `k` scales precision at fixed mean (pure width) and `delta`
    shifts the mean at fixed precision (pure location), so the two knobs are
    exactly orthogonal -- which the recalibrator's own alpha offsets are not.
    The `delta == 0` branch keeps the identity point bit-exact with
    `share_interval` rather than a logit round-trip away from it.
    """
    if delta != 0.0:
        nu = a + b
        m = expit(logit(np.clip(a / nu, 1e-12, 1.0 - 1e-12)) + delta)
        a, b = nu * m, nu * (1.0 - m)
    a2 = np.clip(k * a, 1e-9, None)
    b2 = np.clip(k * b, 1e-9, None)
    return tuple(betaincinv(a2, b2, p) for p in q)


def interval_coverage(a, b, x, *, k=1.0, delta=0.0):
    lo, hi = shifted_beta_quantiles(a, b, k=k, delta=delta)
    return float(((x >= lo) & (x <= hi)).mean())


def oracle_ceiling(a, b, x, *, k_grid=K_GRID, delta_grid=(0.0,)) -> dict:
    """Max coverage over the (k, delta) grid; `delta_grid=(0.0,)` is width-only.

    The oracle modifies one group's marginal at a time; the result corresponds
    to no coherent Dirichlet across groups.  That is fine for a ceiling -- the
    question is what any per-group knob of this shape could reach, not what one
    alpha vector realizes.
    """
    k = np.asarray(k_grid, dtype=float)[:, None, None]
    d = np.asarray(delta_grid, dtype=float)[None, :, None]
    nu = a + b
    m = expit(logit(np.clip(a / nu, 1e-12, 1.0 - 1e-12))[None, None, :] + d)
    a_eff = np.where(d != 0.0, nu * m, a)
    b_eff = np.where(d != 0.0, nu * (1.0 - m), b)
    a2 = np.clip(k * a_eff, 1e-9, None)
    b2 = np.clip(k * b_eff, 1e-9, None)
    lo = betaincinv(a2, b2, NOMINAL_Q[0])
    hi = betaincinv(a2, b2, NOMINAL_Q[1])
    cov = ((x >= lo) & (x <= hi)).mean(-1)
    i, j = np.unravel_index(np.argmax(cov), cov.shape)
    return {
        "cov": float(cov[i, j]),
        "k": float(np.asarray(k_grid)[i]),
        "delta": float(np.asarray(delta_grid)[j]),
    }


def _marginal_params(alpha: np.ndarray, numeric: np.ndarray, g: int):
    on = alpha * numeric
    a = on[:, g]
    b = np.clip(on.sum(-1) - a, 1e-9, None)
    return a, b


def _coverage_matrix(alpha, numeric, magnitude, eligible):
    """[N, 7] float: 1/0 covered by the nominal interval, NaN off-eligible."""
    out = np.full(alpha.shape, np.nan)
    for g in range(alpha.shape[1]):
        m = eligible[:, g]
        if not m.any():
            continue
        a, b = _marginal_params(alpha[m], numeric[m], g)
        lo, hi = shifted_beta_quantiles(a, b)
        x = magnitude[m, g]
        out[m, g] = ((x >= lo) & (x <= hi)).astype(float)
    return out


def _boot_cov(covered: np.ndarray, units: np.ndarray):
    def stat(rows):
        with np.errstate(invalid="ignore"):
            return np.nanmean(covered[rows], axis=0)
    return unit_bootstrap(stat, units)


def decompose(gp, alpha: np.ndarray) -> list[dict]:
    """Per-group residual stats for one (tier, slice), on the corrected alpha."""
    numeric = gp.numeric.numpy()
    magnitude = gp.magnitude.numpy()
    sio2 = gp.sio2.numpy()
    numeric_b = numeric.astype(bool)

    # rows with a single quantified group are free hits: the Beta marginal
    # collapses to [1, 1] and always covers, so they carry no width information
    eligible = numeric_b & (numeric_b.sum(1, keepdims=True) >= 2)

    covered = _coverage_matrix(alpha, numeric, magnitude, eligible)
    cov_lo, cov_hi = _boot_cov(covered, gp.unit)
    cond = conditional_mean(alpha, numeric)
    pit = beta_pit(alpha, numeric, magnitude)
    alpha0 = (alpha * numeric).sum(-1)

    rows = []
    for g, group in enumerate(GROUPS):
        m = eligible[:, g]
        n = int(m.sum())
        if n < 3:
            rows.append({"group": group, "n": n})
            continue
        a, b = _marginal_params(alpha[m], numeric[m], g)
        x = magnitude[m, g]
        cov80 = float(np.nanmean(covered[m, g]))
        width = oracle_ceiling(a, b, x)
        both = oracle_ceiling(a, b, x, delta_grid=DELTA_GRID)
        assert width["cov"] >= cov80 - 1e-12 and both["cov"] >= width["cov"] - 1e-12

        # sharpness at each optimum: a ceiling reached only by inflating the
        # interval toward (0, 1) is vacuous, and the coverage number alone
        # cannot tell that apart from a genuine fix
        lo1, hi1 = shifted_beta_quantiles(a, b)
        lo_k, hi_k = shifted_beta_quantiles(a, b, k=width["k"])
        lo_b, hi_b = shifted_beta_quantiles(a, b, k=both["k"], delta=both["delta"])
        width["median_width"] = float(np.median(hi_k - lo_k))
        both["median_width"] = float(np.median(hi_b - lo_b))

        # CI on the width ceiling at the argmax chosen on the full slice, so
        # the bootstrap is not re-optimizing inside each resample
        w_covered = np.full((len(alpha), 1), np.nan)
        w_covered[m, 0] = ((x >= lo_k) & (x <= hi_k)).astype(float)
        w_lo, w_hi = _boot_cov(w_covered, gp.unit)

        ratio = np.log2(np.maximum(cond[m, g], 1e-9)
                        / np.maximum(x, 1e-9))
        resid = x - cond[m, g]
        rows.append({
            "group": group,
            "n": n,
            "cov80": cov80,
            "cov80_lo": float(cov_lo[g]),
            "cov80_hi": float(cov_hi[g]),
            "median_log2_ratio": float(np.median(ratio)),
            "mean_pit": float(np.nanmean(pit[m, g])),
            "median_width_identity": float(np.median(hi1 - lo1)),
            "width_only": {
                "cov": width["cov"], "k": width["k"],
                "median_width": width["median_width"],
                "cov_lo": float(w_lo[0]), "cov_hi": float(w_hi[0]),
            },
            "width_location": both,
            "spearman_resid_sio2": float(spearmanr(resid, sio2[m]).statistic),
            "spearman_resid_alpha0": float(spearmanr(resid, alpha0[m]).statistic),
            "location_limited": bool(
                float(w_hi[0]) < NOMINAL and both["cov"] >= 0.75
            ),
        })
    return rows


def _corrected_alpha(recalibrator, gp) -> np.ndarray:
    if recalibrator is None:
        return gp.alpha7.numpy()
    with torch.no_grad():
        _, alpha = recalibrator(gp.logits7, gp.alpha7)
    return alpha.numpy()


def _identity_check(gp) -> None:
    """The identity-tier coverage must equal `share_interval`'s directly."""
    alpha = gp.alpha7.numpy()
    numeric = gp.numeric.numpy()
    magnitude = gp.magnitude.numpy()
    eligible = numeric.astype(bool) & (numeric.astype(bool).sum(1, keepdims=True) >= 2)
    lo, hi = share_interval(alpha, numeric)
    direct = (magnitude >= lo) & (magnitude <= hi)
    ours = _coverage_matrix(alpha, numeric, magnitude, eligible)
    m = ~np.isnan(ours)
    assert np.array_equal(ours[m].astype(bool), direct[m])


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def plot_decomposition(tiers: dict, out_path: Path, subtitle: str) -> None:
    """Coverage ceilings per tier (left) and location stats across tiers (right)."""
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6))
    x = np.arange(len(GROUPS))

    for row, tier in enumerate(TIERS_FIT):
        ax = axes[row, 0]
        per = {r["group"]: r for r in tiers[tier]["val"]}
        idn = {r["group"]: r for r in tiers["identity"]["val"]}
        bars = [
            ("identity", "0.7", lambda r, i: i.get("cov80")),
            ("after fit", "tab:orange", lambda r, i: r.get("cov80")),
            ("width-only ceiling", "tab:blue", lambda r, i: (r.get("width_only") or {}).get("cov")),
            ("width+location", "tab:green", lambda r, i: (r.get("width_location") or {}).get("cov")),
        ]
        w = 0.2
        for j, (label, color, get) in enumerate(bars):
            vals = [get(per[g], idn[g]) for g in GROUPS]
            ax.bar(x + (j - 1.5) * w, [v if v is not None else 0 for v in vals],
                   width=w, color=color, label=label if row == 0 else None)
        errs = [(per[g].get("cov80"), per[g].get("cov80_lo"), per[g].get("cov80_hi"))
                for g in GROUPS]
        for i, (c, lo, hi) in enumerate(errs):
            if c is not None:
                ax.plot([x[i] - 0.5 * w] * 2, [lo, hi], color="0.2", lw=1.0)
        # the k that buys the ceiling, so a bar at 1.0 reached only by a
        # 100x precision cut reads as the vacuity it is
        for i, g in enumerate(GROUPS):
            wo = per[g].get("width_only")
            if wo:
                ax.annotate(f"k {wo['k']:.2g}", (x[i] + 0.5 * w, 0.02),
                            rotation=90, fontsize=6.5, ha="center", va="bottom",
                            color="white")
        ax.axhline(NOMINAL, ls="--", lw=1.0, color="0.35")
        ax.set_xticks(x, [f"{g}\nn={per[g]['n']}" for g in GROUPS], fontsize=8)
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("coverage at nominal 80%", fontsize=9)
        ax.set_title(f"{tier} tier, val slice", fontsize=10)

    series = [("identity", "0.5", "o"), ("vector", "tab:orange", "s"),
              ("matrix", "tab:blue", "^")]

    ax = axes[0, 1]
    for tier, color, marker in series:
        per = {r["group"]: r for r in tiers[tier]["val"]}
        vals = [per[g].get("median_log2_ratio") for g in GROUPS]
        ax.plot(x, [np.nan if v is None else v for v in vals], marker,
                color=color, ms=5, label=tier)
    ax.axhline(0, lw=1.0, color="0.35")
    ax.set_xticks(x, GROUPS, fontsize=8)
    ax.set_ylabel("median log2(predicted / measured)", fontsize=9)
    ax.set_title("location: signed magnitude bias", fontsize=10)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    for tier, color, marker in series:
        per = {r["group"]: r for r in tiers[tier]["val"]}
        vals = [per[g].get("mean_pit") for g in GROUPS]
        ax.plot(x, [np.nan if v is None else v for v in vals], marker,
                color=color, ms=5, label=tier)
    ax.axhline(0.5, lw=1.0, color="0.35")
    ax.set_xticks(x, GROUPS, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("mean PIT", fontsize=9)
    ax.set_title("location: where measured shares sit in the predicted Beta",
                 fontsize=10)

    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"width vs location after the recalibrator fit -- {subtitle}\n"
                 "fit on train, shown on val; test not used", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def run_arm(checkpoint: Path) -> None:
    ckpt = JointHeadCheckpoint.load(checkpoint)
    device = pick_device()
    encoder, _ = load_encoder_from_checkpoint(ckpt.encoder_checkpoint)
    encoder.to(device)
    benchmark = load_benchmark()
    latents = bulk_latents(encoder, benchmark, device)
    gp = build_group_predictions(ckpt.build_mode_head().to(device),
                                 benchmark, latents)

    clean = benchmark.clean & benchmark.has_magnitude
    slices = {
        name: gp.subset(split_mask(benchmark, name) & clean)
        for name in ("train", "val")
    }
    for name, data in slices.items():
        _identity_check(data)
        numeric = data.numeric.numpy().astype(bool)
        n_deg = int((numeric.sum(1) == 1).sum())
        print(f"{name}: {len(data)} rocks / {len(np.unique(data.unit))} units, "
              f"{n_deg} single-numeric rows excluded from coverage")

    identity = {
        name: decompose(data, _corrected_alpha(None, data))
        for name, data in slices.items()
    }
    zero_shot = score_recalibrator(None, slices["val"])
    identity["val_joint_nll"] = float(zero_shot["joint"].mean())
    identity["val_dirichlet_nll"] = float(zero_shot["dirichlet"].mean())
    tiers = {"identity": identity}
    for tier in TIERS_FIT:
        recalibrator = fit_recalibrator(
            slices["train"], tier=tier, weight_decay=WEIGHT_DECAY[tier],
        )
        entry = {
            name: decompose(data, _corrected_alpha(recalibrator, data))
            for name, data in slices.items()
        }
        val_score = score_recalibrator(recalibrator, slices["val"])
        entry["val_joint_nll"] = float(val_score["joint"].mean())
        entry["val_dirichlet_nll"] = float(val_score["dirichlet"].mean())
        if tier == "vector":
            entry["offsets"] = per_group_offsets(recalibrator)
        tiers[tier] = entry
        for r in entry["val"]:
            if "cov80" not in r:
                continue
            w, b = r["width_only"], r["width_location"]
            verdict = " -> location-limited" if r["location_limited"] else ""
            print(f"  {tier} {r['group']}: cov80 {r['cov80']:.2f} "
                  f"[{r['cov80_lo']:.2f}, {r['cov80_hi']:.2f}], "
                  f"width ceiling {w['cov']:.2f} [{w['cov_lo']:.2f}, "
                  f"{w['cov_hi']:.2f}] at k {w['k']:.3g} "
                  f"(interval {r['median_width_identity']:.2f} -> "
                  f"{w['median_width']:.2f}), width+loc {b['cov']:.2f}{verdict}")

    tag = checkpoint.parent.name
    out_dir = RESULTS / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint": str(checkpoint),
        "tag": tag,
        "benchmark": "eruption-v2",
        "groups": list(GROUPS),
        "weight_decay": {t: WEIGHT_DECAY[t] for t in TIERS_FIT},
        "grids": {"k": [float(k) for k in K_GRID],
                  "delta": [float(d) for d in DELTA_GRID]},
        "nominal": NOMINAL,
        "splits": {
            name: {
                "rocks": len(data),
                "units": int(len(np.unique(data.unit))),
                "n_degenerate": int(
                    (data.numeric.numpy().astype(bool).sum(1) == 1).sum()
                ),
            }
            for name, data in slices.items()
        },
        "tiers": tiers,
    }
    (out_dir / "residual_decomposition.json").write_text(
        json.dumps(_jsonable(payload), indent=2)
    )
    print(f"saved {out_dir / 'residual_decomposition.json'}")

    subtitle = f"{tag}, {len(slices['val'])} val rocks"
    plot_decomposition(tiers, out_dir / "residual_decomposition.png", subtitle)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, nargs="+",
                        default=list(DEFAULT_CHECKPOINTS))
    args = parser.parse_args()
    for checkpoint in args.checkpoint:
        print(f"== {checkpoint.parent.name}")
        run_arm(checkpoint)


if __name__ == "__main__":
    main()
