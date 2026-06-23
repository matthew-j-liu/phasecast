"""
Throwaway audit for issue #6: does the trace gate's traces-only XGBoost tuning
objective bias its verdict?

`run_gate._tune_shared` tunes XGBoost on the 13 traces only, then reuses those
params for the major final fits too.  This checks two assumptions by re-tuning
under majors-only and all-22 objectives on the same frozen latent:
  (a) verdict stability -- trace R2s and the 1a/1b verdict don't move under
      best_all vs best_traces;
  (b) majors handicap -- major R2s don't materially improve under best_majors
      vs the trace-tuned best_traces.

Encodes each masked-token latent once and caches it, so only XGBoost re-tunes
and re-fits across the three objectives -- no re-encoding.  Read-only: nothing
trains or mutates the encoder.  The durable record is the JOURNAL entry.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_WARNINGS", "off")

import argparse
import json
import sys
from pathlib import Path

# runnable as a plain file (`python explorations/.../audit_tuning_objective.py`):
# put the repo root on the path so `from explorations...` resolves.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import optuna
import torch
from sklearn.model_selection import train_test_split

from cnmineral.data.tokens import VOCAB
from cnmineral.eval.imputation import _regression_metrics
from cnmineral.eval.latent import load_encoder_from_checkpoint
from cnmineral.training import checkpoint_tag, create_dataset, pick_device, reserve_test

from explorations.trace_signal_in_latent.run_gate import (
    RESULTS_ROOT,
    _N_MAJORS,
    _TRACE_POSITIONS,
    _build_gate,
    _collect_frac,
    _encode_with_masked_token,
    _fit_predict_xgb,
    _log_feature_matrix,
    _tune_shared,
)

MATERIALITY = 0.005

_GROUPS = {
    "traces": _TRACE_POSITIONS,
    "majors": list(range(_N_MAJORS)),
    "all": list(range(len(VOCAB))),
}


def _build_tasks(
    positions, arm, latent_cache, xlog_fit, xlog_val, frac_fit, frac_val,
    pop_fit, pop_val, sf, sv,
):
    """Tuning tasks `(x_fit, y_fit, x_val, y_val)` for one arm over `positions`.

    `arm` is "1a" (input log-features, target column deleted) or "1b" (cached
    masked-token latent).  Rows are the `sf`/`sv` subsample further filtered to
    where the target element is populated, matching `run_imputation_diagnostic`.
    """
    tasks = []
    for e in positions:
        if arm == "1a":
            rf = sf[pop_fit[sf, e]]
            rv = sv[pop_val[sv, e]]
            if rf.size == 0 or rv.size == 0:
                continue
            tasks.append((
                np.delete(xlog_fit, e, axis=1)[rf], np.log10(frac_fit[rf, e]),
                np.delete(xlog_val, e, axis=1)[rv], np.log10(frac_val[rv, e]),
            ))
        else:
            rf_mask = pop_fit[sf, e]
            rv_mask = pop_val[sv, e]
            if rf_mask.sum() == 0 or rv_mask.sum() == 0:
                continue
            h_fit, h_val, _ = latent_cache[e]
            tasks.append((
                h_fit[sf][rf_mask], np.log10(frac_fit[sf][rf_mask, e]),
                h_val[sv][rv_mask], np.log10(frac_val[sv][rv_mask, e]),
            ))
    return tasks


def _final_table(
    positions, best_1a, best_1b, latent_cache,
    xlog_fit, xlog_val, xlog_test, frac_fit, frac_val, frac_test,
    pop_fit, pop_val, pop_test, seed,
):
    """Per-element log-space R2 on the full test split, shaped like the gate's
    `per_element` so `_build_gate` can read it directly."""
    out = {}
    for token_pos in positions:
        rows_fit = pop_fit[:, token_pos]
        rows_val = pop_val[:, token_pos]
        rows_test = pop_test[:, token_pos]
        if rows_fit.sum() == 0 or rows_test.sum() == 0:
            continue
        sym = VOCAB[token_pos][0]
        y_fit = np.log10(frac_fit[rows_fit, token_pos])
        y_val = np.log10(frac_val[rows_val, token_pos])
        y_test = np.log10(frac_test[rows_test, token_pos])

        pred_1a = _fit_predict_xgb(
            best_1a,
            np.delete(xlog_fit, token_pos, axis=1)[rows_fit], y_fit,
            np.delete(xlog_val, token_pos, axis=1)[rows_val], y_val,
            np.delete(xlog_test, token_pos, axis=1)[rows_test],
            seed=seed,
        )
        h_fit, h_val, h_test = latent_cache[token_pos]
        pred_1b = _fit_predict_xgb(
            best_1b, h_fit[rows_fit], y_fit, h_val[rows_val], y_val,
            h_test[rows_test], seed=seed,
        )
        out[sym] = {
            "token_pos": token_pos,
            "run_1a": {"r2": _regression_metrics(y_test, pred_1a)["r2"]},
            "run_1b": {"r2": _regression_metrics(y_test, pred_1b)["r2"]},
        }
    return out


def _delta_rows(base_table, alt_table):
    """Per-element (r2_1a, r2_1b) for base vs alt with deltas, in token order."""
    rows = []
    for sym in sorted(base_table, key=lambda s: base_table[s]["token_pos"]):
        b, a = base_table[sym], alt_table[sym]
        b1a, b1b = b["run_1a"]["r2"], b["run_1b"]["r2"]
        a1a, a1b = a["run_1a"]["r2"], a["run_1b"]["r2"]
        rows.append({
            "sym": sym,
            "base_1a": b1a, "alt_1a": a1a, "d_1a": a1a - b1a,
            "base_1b": b1b, "alt_1b": a1b, "d_1b": a1b - b1b,
        })
    return rows


def _print_table(title, base_label, alt_label, rows):
    print(f"\n{title}")
    print(f"  {'el':<4s} {base_label+' 1a':>11s} {alt_label+' 1a':>11s} {'d1a':>7s}"
          f" {base_label+' 1b':>11s} {alt_label+' 1b':>11s} {'d1b':>7s}")
    for r in rows:
        flag = "  *" if max(abs(r["d_1a"]), abs(r["d_1b"])) > MATERIALITY else ""
        print(f"  {r['sym']:<4s} {r['base_1a']:>11.3f} {r['alt_1a']:>11.3f}"
              f" {r['d_1a']:>+7.3f} {r['base_1b']:>11.3f} {r['alt_1b']:>11.3f}"
              f" {r['d_1b']:>+7.3f}{flag}")
    max_d1a = max(abs(r["d_1a"]) for r in rows)
    max_d1b = max(abs(r["d_1b"]) for r in rows)
    print(f"  max |delta|: 1a={max_d1a:.3f} 1b={max_d1b:.3f}"
          f"  (materiality {MATERIALITY})")
    return max_d1a, max_d1b


def run_audit(
    checkpoint_path, *, n_trials=40, study_subsample=None, batch_size=512,
    device=None,
):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    tag = checkpoint_tag(checkpoint_path)
    output_dir = RESULTS_ROOT / tag
    output_dir.mkdir(parents=True, exist_ok=True)
    device = device or pick_device()
    print(f"device:     {device}")
    print(f"checkpoint: {checkpoint_path}")

    encoder, ckpt = load_encoder_from_checkpoint(checkpoint_path)
    encoder.to(device)
    seed = int(ckpt.config.get("seed", 0))

    dataset = create_dataset(ckpt.config)
    splits = reserve_test(dataset, ckpt.config)
    trainval_idx = np.asarray(splits["trainval"], dtype=np.int64)
    test_idx = np.asarray(splits["test"], dtype=np.int64)
    fit_idx, val_idx = train_test_split(
        trainval_idx, test_size=0.2, random_state=seed)
    fit_idx = np.asarray(fit_idx, dtype=np.int64)
    val_idx = np.asarray(val_idx, dtype=np.int64)
    print(f"dataset n={len(dataset)} fit={len(fit_idx)} val={len(val_idx)} test={len(test_idx)}")

    frac_fit, pop_fit = _collect_frac(dataset, fit_idx, batch_size=batch_size)
    frac_val, pop_val = _collect_frac(dataset, val_idx, batch_size=batch_size)
    frac_test, pop_test = _collect_frac(dataset, test_idx, batch_size=batch_size)
    xlog_fit = _log_feature_matrix(frac_fit, pop_fit)
    xlog_val = _log_feature_matrix(frac_val, pop_val)
    xlog_test = _log_feature_matrix(frac_test, pop_test)

    print(f"encoding {len(VOCAB)} masked-token latents (once) ...")
    latent_cache = {}
    for e in range(len(VOCAB)):
        h_fit = _encode_with_masked_token(
            encoder, dataset, fit_idx, e, device=device, batch_size=batch_size).astype(np.float32)
        h_val = _encode_with_masked_token(
            encoder, dataset, val_idx, e, device=device, batch_size=batch_size).astype(np.float32)
        h_test = _encode_with_masked_token(
            encoder, dataset, test_idx, e, device=device, batch_size=batch_size).astype(np.float32)
        latent_cache[e] = (h_fit, h_val, h_test)
        print(f"  {VOCAB[e][0]:<3s} cached")

    rng = np.random.default_rng(seed)
    if study_subsample is None:
        sf = np.arange(len(fit_idx))
        sv = np.arange(len(val_idx))
    else:
        sf = rng.choice(len(fit_idx), size=min(study_subsample, len(fit_idx)), replace=False)
        sv = rng.choice(len(val_idx), size=min(study_subsample, len(val_idx)), replace=False)
    print(f"tuning rows: fit={len(sf)} val={len(sv)}"
          f"{' (all)' if study_subsample is None else f' (subsample={study_subsample})'}")

    best = {}
    for group, positions in _GROUPS.items():
        for arm in ("1a", "1b"):
            tasks = _build_tasks(
                positions, arm, latent_cache, xlog_fit, xlog_val,
                frac_fit, frac_val, pop_fit, pop_val, sf, sv)
            print(f"tuning {arm} / {group} ({len(tasks)} tasks, {n_trials} trials) ...")
            best[(group, arm)] = _tune_shared(tasks, n_trials=n_trials, seed=seed)
            print(f"  best {arm}/{group}: {best[(group, arm)]}")

    def final(positions, group):
        return _final_table(
            positions, best[(group, "1a")], best[(group, "1b")], latent_cache,
            xlog_fit, xlog_val, xlog_test, frac_fit, frac_val, frac_test,
            pop_fit, pop_val, pop_test, seed)

    print("\nfinal fits on full splits ...")
    traces_base = final(_TRACE_POSITIONS, "traces")
    traces_all = final(_TRACE_POSITIONS, "all")
    majors_base = final(list(range(_N_MAJORS)), "traces")
    majors_self = final(list(range(_N_MAJORS)), "majors")

    gate_base = _build_gate(traces_base)
    gate_all = _build_gate(traces_all)

    rows_a = _delta_rows(traces_base, traces_all)
    rows_b = _delta_rows(majors_base, majors_self)

    print("\n" + "=" * 72)
    print(f"AUDIT: trace-gate tuning objective -- {tag}")
    print("=" * 72)
    ma_1a, ma_1b = _print_table(
        "(a) verdict stability -- traces, best_traces vs best_all",
        "trc", "all", rows_a)
    print(f"  verdict  best_traces: {gate_base['verdict']}"
          f"  (med_1a {gate_base['median_r2_1a']:.3f} med_1b {gate_base['median_r2_1b']:.3f})")
    print(f"  verdict  best_all:    {gate_all['verdict']}"
          f"  (med_1a {gate_all['median_r2_1a']:.3f} med_1b {gate_all['median_r2_1b']:.3f})")
    verdict_stable = gate_base["verdict"] == gate_all["verdict"]
    trace_stable = max(ma_1a, ma_1b) <= MATERIALITY and verdict_stable

    mb_1a, mb_1b = _print_table(
        "(b) majors handicap -- majors, best_traces vs best_majors",
        "trc", "maj", rows_b)
    majors_ok = max(mb_1a, mb_1b) <= MATERIALITY

    print("\nverdict:")
    print(f"  (a) trace verdict {'STABLE' if trace_stable else 'MOVED'}"
          f" -- verdict_unchanged={verdict_stable}, max trace |delta|={max(ma_1a, ma_1b):.3f}")
    print(f"  (b) majors {'NOT handicapped' if majors_ok else 'HANDICAPPED'}"
          f" -- max major |delta|={max(mb_1a, mb_1b):.3f}")
    assumption_holds = trace_stable and majors_ok
    print(f"  => traces-only tuning objective {'VALIDATED' if assumption_holds else 'QUESTIONED'}")

    payload = {
        "checkpoint": str(checkpoint_path),
        "tag": tag,
        "n_trials": n_trials,
        "study_subsample": study_subsample,
        "materiality": MATERIALITY,
        "best_params": {f"{g}/{a}": best[(g, a)] for (g, a) in best},
        "a_verdict_stability": {
            "rows": rows_a,
            "verdict_best_traces": gate_base["verdict"],
            "verdict_best_all": gate_all["verdict"],
            "median_best_traces": {"r2_1a": gate_base["median_r2_1a"], "r2_1b": gate_base["median_r2_1b"]},
            "median_best_all": {"r2_1a": gate_all["median_r2_1a"], "r2_1b": gate_all["median_r2_1b"]},
            "max_abs_delta": {"r2_1a": ma_1a, "r2_1b": ma_1b},
            "stable": trace_stable,
        },
        "b_majors_handicap": {
            "rows": rows_b,
            "max_abs_delta": {"r2_1a": mb_1a, "r2_1b": mb_1b},
            "not_handicapped": majors_ok,
        },
        "assumption_validated": assumption_holds,
    }
    out_path = output_dir / "audit_tuning_objective.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nwrote {out_path}")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--n-trials", type=int, default=40)
    parser.add_argument(
        "--study-subsample", type=int, default=None,
        help="cap per-split rows during Optuna tuning; default tunes on all rows "
             "(production parity). final fits always use all rows.")
    args = parser.parse_args()
    run_audit(
        checkpoint_path=args.checkpoint.resolve(),
        n_trials=args.n_trials,
        study_subsample=args.study_subsample,
    )


if __name__ == "__main__":
    main()
