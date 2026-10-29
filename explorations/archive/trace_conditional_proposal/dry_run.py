"""Sampler-level dry run of the trace-conditional (P, H2O) proposal (#48).

Runs the candidate proposal side by side with the current broad sampler on
the committed corpus rosters -- no MAGEMin -- and measures tier occupancy,
range narrowing, proxy-vs-class contradictions, setting-posterior entropy,
the 24-draw envelope contraction, a stress-case gallery, and the box atlas.

    conda run -n mineral python explorations/trace_conditional_proposal/dry_run.py
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cnmineral.data.condition_proposal import (
    _MAFICITY,
    P_HI,
    P_LO,
    SETTINGS,
    TRACE_TIER_BOXES,
    WALLACE_PLANK_2013,
    # archived record: ProposalKnobs and bulk_proposal's knobs= were removed
    # on 2026-10-29; the locked values now live in condition_proposal.py
    ProposalKnobs,
    bulk_proposal,
    draw_conditions,
    w_max,
)
from cnmineral.eval.supergroup import qap_to_supergroup

REPO = Path(__file__).resolve().parents[2]
TRACE_COLS = ["th_ppm", "nb_ppm", "yb_ppm", "v_ppm", "y_ppm", "tio2"]
SETTING_COLORS = {"arc": "#E8883A", "ridge": "#2E5BFF", "intraplate": "#2A9D8F"}


def load_bulks(roster):
    frames = []
    names = {"train": ["train"], "valtest": ["valtest"], "both": ["train", "valtest"]}[roster]
    for name in names:
        df = pd.read_csv(REPO / "data" / "corpus" / f"{name}_bulks.csv")
        df["roster"] = name
        frames.append(df)
    bulks = pd.concat(frames, ignore_index=True)
    bulks["qap_name"] = bulks.qap_name.where(bulks.qap_name.notna(), None)
    gard = pd.read_parquet(REPO / "data" / "bulks" / "gard2019.parquet",
                           columns=TRACE_COLS).loc[bulks.sample_id]
    for c in TRACE_COLS:
        bulks[c] = gard[c].to_numpy()
    return bulks


def span(x):
    lo, hi = np.percentile(x, [5, 95])
    return hi - lo


def measure_bulk(row, knobs, n_draws, seed):
    traces = {c: row[c] for c in TRACE_COLS}
    bp = bulk_proposal(row["qap_name"], traces, knobs)
    sid = int(row["sample_id"])
    p_n, w_n, _ = bp.draw(n_draws, np.random.default_rng([seed, sid, 0]))
    p_o, w_o, _ = draw_conditions(n_draws, np.random.default_rng([seed, sid, 1]))
    p_n24, w_n24, _ = bp.draw(24, np.random.default_rng([seed, sid, 2]))
    p_o24, w_o24, _ = draw_conditions(24, np.random.default_rng([seed, sid, 3]))

    sw = np.asarray(bp.setting_weights)
    if sw.sum() > 0:
        pr = sw / sw.sum()
        ent = float(-(pr[pr > 0] * np.log(pr[pr > 0])).sum())
    else:
        ent = np.nan
    key = row["qap_name"] if row["qap_name"] in WALLACE_PLANK_2013 else "default"
    c_lo, c_hi = WALLACE_PLANK_2013[key]
    return {
        "sample_id": sid,
        "roster": row["roster"],
        "qap_name": row["qap_name"],
        "supergroup": qap_to_supergroup(row["qap_name"]),
        "tier": bp.tier,
        "rung": bp.rung if bp.rung is not None else "none",
        "w_arc": sw[0], "w_ridge": sw[1], "w_intraplate": sw[2],
        "entropy": ent,
        "p_width_new": span(p_n), "p_width_old": span(p_o),
        "w_width_new": span(w_n), "w_width_old": span(w_o),
        "area_new": span(p_n) * span(w_n), "area_old": span(p_o) * span(w_o),
        "area24_new": span(p_n24) * span(w_n24),
        "area24_old": span(p_o24) * span(w_o24),
        "class_h2o_overlap": float(np.mean((w_n >= c_lo) & (w_n <= c_hi))),
    }


def fig_occupancy(df, out):
    rung_order = ["th_nb_yb", "ti_v", "nb_y", "none"]
    colors = {"th_nb_yb": "#6C5CE7", "ti_v": "#12B5A5", "nb_y": "#2A9D8F", "none": "#C8CDD6"}
    groups = ["all"] + sorted(df.supergroup.unique())
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, g in enumerate(groups):
        sub = df if g == "all" else df[df.supergroup == g]
        y0 = 0.0
        for rung in rung_order:
            frac = (sub.rung == rung).mean()
            ax.bar(i, frac, bottom=y0, color=colors[rung],
                   label=rung if i == 0 else None)
            y0 += frac
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("fraction of bulks")
    ax.legend(fontsize=8)
    ax.set_title("discriminant rung occupancy (none = class fallback)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_narrowing(df, out):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, col, label in [
            (axes[0], ("p_width_new", "p_width_old"), "P 5-95% width ratio"),
            (axes[1], ("w_width_new", "w_width_old"), "H2O 5-95% width ratio"),
            (axes[2], ("area_new", "area_old"), "joint area ratio")]:
        r = df[col[0]] / df[col[1]]
        ax.hist(r, bins=60, color="#2E5BFF")
        ax.axvline(r.median(), color="k", lw=1)
        ax.set_title(f"{label}  (median {r.median():.2f})", fontsize=9)
        ax.set_xlabel("new / old")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_entropy(df, out):
    tr = df[df.tier == "trace"]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(tr.entropy.dropna(), bins=50, color="#6C5CE7")
    ax.axvline(np.log(3), color="k", lw=1, ls="--", label="uniform (log 3)")
    ax.set_xlabel("setting-posterior entropy (nats)")
    ax.set_ylabel("trace-tier bulks")
    ax.legend(fontsize=8)
    frac_amb = float((tr.entropy > 0.5).mean())
    ax.set_title(f"posterior entropy; {frac_amb:.0%} of trace-tier bulks > 0.5 nats", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_envelope(df, out):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    r = df.area24_new / df.area24_old
    ax.hist(r, bins=60, color="#12B5A5")
    ax.axvline(r.median(), color="k", lw=1)
    ax.set_xlabel("24-draw envelope area, new / old")
    ax.set_ylabel("bulks")
    ax.set_title(f"envelope contraction at 24 draws (median {r.median():.2f})", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_gallery(bulks, knobs, seed, out):
    # the stress set predates the split swap, so most of it is off-roster;
    # its traces come straight from gard2019 instead
    stress = pd.read_csv(REPO / "explorations" / "equilibrium_multimodality"
                         / "results" / "selected_bulks.csv")
    gard = pd.read_parquet(REPO / "data" / "bulks" / "gard2019.parquet",
                           columns=TRACE_COLS).loc[stress.sample_id]
    picks = stress[["sample_id", "qap_name"]].copy()
    for c in TRACE_COLS:
        picks[c] = gard[c].to_numpy()
    picks["why"] = "stress"
    rest = bulks[~bulks.sample_id.isin(stress.sample_id)]
    exemplars = (rest.groupby(rest.qap_name.map(qap_to_supergroup), group_keys=False)
                 .apply(lambda g: g.nsmallest(2, "sample_id")))
    exemplars = exemplars.copy()
    exemplars["why"] = "exemplar"
    panels = pd.concat([picks, exemplars], ignore_index=True)

    ncol = 6
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 2.2 * nrow))
    pg = np.linspace(P_LO, P_HI, 200)
    for ax, (_, row) in zip(axes.ravel(), panels.iterrows()):
        traces = {c: row[c] for c in TRACE_COLS}
        bp = bulk_proposal(row["qap_name"], traces, knobs)
        sid = int(row["sample_id"])
        p_o, w_o, _ = draw_conditions(256, np.random.default_rng([seed, sid, 4]))
        p_n, w_n, _ = bp.draw(256, np.random.default_rng([seed, sid, 5]))
        dom = SETTINGS[int(np.argmax(bp.setting_weights))] if bp.tier == "trace" else None
        color = SETTING_COLORS.get(dom, "#8A93A3")
        ax.scatter(p_o, w_o, s=2, c="#C8CDD6", alpha=0.5)
        ax.scatter(p_n, w_n, s=2, c=color, alpha=0.6)
        ax.plot(pg, w_max(pg), color="k", lw=0.5)
        ax.set_xlim(0, P_HI + 0.3)
        ax.set_ylim(0, 8.5)
        ax.set_xticks([])
        ax.set_yticks([])
        tag = f"{sid} {row['qap_name']}"[:28]
        ax.set_title(f"{tag}\n{row['why']} | {bp.tier}:{bp.rung or 'none'}", fontsize=6)
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    fig.suptitle("old broad cloud (gray) vs trace-conditional cloud (dominant-setting color)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return len(panels)


def fig_atlas(bulks, knobs, out):
    box_share = defaultdict(float)
    fb_share = defaultdict(float)
    floor_share = 0.0
    n = len(bulks)
    for _, row in bulks.iterrows():
        bp = bulk_proposal(row["qap_name"], {c: row[c] for c in TRACE_COLS}, knobs)
        floor_share += bp.floor / n
        # bulk_proposal appends the kept setting boxes in SETTINGS order,
        # then the single fallback box -- recover identity from that order
        bucket = _MAFICITY.get(qap_to_supergroup(row["qap_name"]), "intermediate")
        kept = [s for s, x in zip(SETTINGS, bp.setting_weights) if x > 1e-6]
        for i, (box, wt) in enumerate(zip(bp.boxes, bp.weights)):
            if i < len(kept):
                box_share[(bucket, kept[i])] += wt / n
            else:
                fb_share[(box.p_lo, box.p_hi, box.w_lo, box.w_hi,
                          row["qap_name"])] += wt / n

    fig, (ax, axb) = plt.subplots(
        1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [3, 2]})
    pg = np.linspace(P_LO, P_HI, 300)
    ax.plot(pg, w_max(pg), color="k", lw=1)
    ax.add_patch(plt.Rectangle((P_LO, 0), P_HI - P_LO, 8, fill=False,
                               ls="--", ec="#8A93A3", lw=1.2))
    # mafic-arc and intermediate-arc are the same rectangle: draw unique
    # geometry here, carry per-key shares in the bar panel
    drawn = set()
    for (bucket, setting) in box_share:
        rect = TRACE_TIER_BOXES[(bucket, setting)]
        if rect in drawn:
            continue
        drawn.add(rect)
        p_lo, p_hi, w_lo, w_hi = rect
        c = SETTING_COLORS[setting]
        ax.add_patch(plt.Rectangle((p_lo, w_lo), p_hi - p_lo, w_hi - w_lo,
                                   fill=True, fc=c, alpha=0.18, ec=c, lw=1.5))
    fb_sorted = sorted(fb_share.items(), key=lambda kv: -kv[1])
    for (p_lo, p_hi, w_lo, w_hi, qap), share in fb_sorted[:8]:
        ax.add_patch(plt.Rectangle((p_lo, w_lo), p_hi - p_lo, w_hi - w_lo,
                                   fill=False, ec="#8A93A3", lw=1.0))
    ax.set_xlabel("P (kbar)")
    ax.set_ylabel("H2O (wt%)")
    ax.set_title("boxes actually sampled (colored = trace tier, gray = top class-fallback,\n"
                 "dashed = broad floor, black = solubility bound)", fontsize=9)
    ax.set_xlim(0, P_HI + 0.3)
    ax.set_ylim(0, 8.7)

    bars = ([(f"{b}-{s}", share, SETTING_COLORS[s])
             for (b, s), share in box_share.items()]
            + [(f"{qap} (fb)", share, "#8A93A3")
               for (p1, p2, w1, w2, qap), share in fb_sorted[:8]]
            + [(f"{len(fb_sorted) - 8} smaller fb boxes",
                sum(s for _, s in fb_sorted[8:]), "#C8CDD6"),
               ("broad floor", floor_share, "#5A6372")])
    bars.sort(key=lambda t: t[1])
    axb.barh(range(len(bars)), [b[1] for b in bars], color=[b[2] for b in bars])
    axb.set_yticks(range(len(bars)))
    axb.set_yticklabels([b[0] for b in bars], fontsize=7)
    for i, (_, share, _) in enumerate(bars):
        axb.text(share + 0.002, i, f"{share:.1%}", va="center", fontsize=7)
    axb.set_xlabel("share of total corpus draws")
    axb.set_title("per-box draw share (sums to 1 with all fallback boxes)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    atlas = {f"{b}|{s}": round(v, 5) for (b, s), v in box_share.items()}
    atlas["floor"] = round(floor_share, 5)
    atlas["fallback_total"] = round(sum(fb_share.values()), 5)
    return atlas


def sensitivity(bulks, seed, out_csv):
    variants = {
        "default": ProposalKnobs(),
        "felsic_out": ProposalKnobs(felsic_in_trace_tier=False),
        "sigma_half": ProposalKnobs(sigma=0.075),
        "sigma_double": ProposalKnobs(sigma=0.30),
        "eps_004": ProposalKnobs(eps=0.04),
        "eps_016": ProposalKnobs(eps=0.16),
    }
    rows = []
    for name, knobs in variants.items():
        ratios, tiers = [], []
        for _, row in bulks.iterrows():
            traces = {c: row[c] for c in TRACE_COLS}
            bp = bulk_proposal(row["qap_name"], traces, knobs)
            sid = int(row["sample_id"])
            p_n, w_n, _ = bp.draw(128, np.random.default_rng([seed, sid, 6]))
            p_o, w_o, _ = draw_conditions(128, np.random.default_rng([seed, sid, 7]))
            ratios.append((span(p_n) * span(w_n)) / (span(p_o) * span(w_o)))
            tiers.append(bp.tier)
        tiers = np.asarray(tiers)
        rows.append({"variant": name,
                     "trace_tier_frac": float((tiers == "trace").mean()),
                     "median_area_shrink": float(np.median(ratios))})
        print(f"  {name}: trace {rows[-1]['trace_tier_frac']:.3f}, "
              f"median shrink {rows[-1]['median_area_shrink']:.3f}")
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roster", choices=["train", "valtest", "both"], default="both")
    ap.add_argument("--n-draws", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="subsample bulks for smoke runs")
    args = ap.parse_args()

    out = Path(__file__).resolve().parent / "results"
    out.mkdir(exist_ok=True)
    knobs = ProposalKnobs()

    bulks = load_bulks(args.roster)
    if args.limit:
        bulks = bulks.sample(args.limit, random_state=args.seed).reset_index(drop=True)
    print(f"{len(bulks)} bulks ({args.roster})")

    records = [measure_bulk(row, knobs, args.n_draws, args.seed)
               for _, row in bulks.iterrows()]
    df = pd.DataFrame(records)
    df.to_csv(out / "bulk_metrics.csv", index=False)

    occ = (df.groupby(["supergroup", "rung"]).size().unstack(fill_value=0))
    occ = occ.div(occ.sum(axis=1), axis=0)
    occ.loc["all"] = df.rung.value_counts(normalize=True)
    occ.to_csv(out / "tier_occupancy.csv")
    print("occupancy (all):", dict(df.rung.value_counts(normalize=True).round(3)))

    shrink = df.area_new / df.area_old
    stats = {
        "n_bulks": len(df),
        "trace_tier_frac": float((df.tier == "trace").mean()),
        "median_area_shrink": float(shrink.median()),
        "p10_area_shrink": float(shrink.quantile(0.10)),
        "p90_area_shrink": float(shrink.quantile(0.90)),
        "frac_not_shrinking": float((shrink > 0.9).mean()),
        "median_h2o_width_shrink": float((df.w_width_new / df.w_width_old).median()),
        "median_p_width_shrink": float((df.p_width_new / df.p_width_old).median()),
        "median_envelope24_shrink": float((df.area24_new / df.area24_old).median()),
        "n_barely_overlap": int(((df.tier == "trace")
                                 & (df.class_h2o_overlap < 0.1)).sum()),
        "median_entropy_trace_tier": float(df[df.tier == "trace"].entropy.median()),
    }
    (out / "narrowing_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))

    flags = df[(df.tier == "trace") & (df.class_h2o_overlap < 0.1)]
    flags.sort_values("class_h2o_overlap").to_csv(out / "overlap_flags.csv", index=False)

    fig_occupancy(df, out / "occupancy.png")
    fig_narrowing(df, out / "narrowing.png")
    fig_entropy(df, out / "setting_entropy.png")
    fig_envelope(df, out / "envelope_contraction.png")
    n_panels = fig_gallery(bulks, knobs, args.seed, out / "gallery.png")
    print(f"gallery: {n_panels} panels")
    atlas = fig_atlas(bulks, knobs, out / "box_atlas.png")
    (out / "box_atlas.json").write_text(json.dumps(atlas, indent=2))

    print("knob sensitivity:")
    sensitivity(bulks, args.seed, out / "knob_sensitivity.csv")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
