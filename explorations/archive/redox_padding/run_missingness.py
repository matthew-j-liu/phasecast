"""Is a padded redox slot a neutral blank, or a hint about the analysis?

  python explorations/redox_padding/run_missingness.py
  python explorations/redox_padding/run_missingness.py --arms inputs

Run from the repo root. The padded state is described as marginalizing over
an unknown redox ratio, which presumes the blank carries no information of
its own. It might not: the rows that report a measured FeO + Fe2O3 pair are
overwhelmingly older analyses (97% of pre-1970 bulks, 10% of post-2010),
because ferrous titration went out of routine use. If the rest of the
composition predicts whether the pair was reported, then a blank slot is
partly a statement about the source, and "no information" overstates it.

Two arms on the SSL pretrain corpus, the population the encoder actually
trained on, split on its own pinned test roster:

- inputs: the 22 element values plus their presence indicators (absence is
  visible to the encoder too, and older analyses report fewer traces). This
  is the ceiling -- what any model could extract.
- latent: the frozen pooled latent with the redox slot padded, which is what
  the encoder actually carries into every downstream head.

Both predict the same target and never see the redox slot itself. Writes
`results/missingness.json`.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_WARNINGS", "off")

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from xgboost import XGBClassifier

from cnmineral.data.tokens import REDOX_TOKEN_ID, REDOX_TOKEN_INDEX, VOCAB
from cnmineral.eval.latent import (
    PaddedRedoxView,
    extract_latents,
    load_encoder_from_checkpoint,
)
from cnmineral.training import create_dataset, pick_device, reserve_test

REPO = Path(__file__).resolve().parents[2]
RESULTS = Path(__file__).resolve().parent / "results"

ENCODER = "checkpoints/pretrain_allrock_redox_seed0/epoch_best.pt"

# Matches the ceiling-probe protocol used elsewhere in explorations/: a
# large forest with early stopping on a held-out slice rather than a tuned
# depth, since the question is what is extractable, not what a tuned model
# would score.
XGB_KWARGS = dict(
    n_estimators=2000,
    early_stopping_rounds=50,
    tree_method="hist",
    objective="binary:logistic",
    eval_metric="auc",
    max_depth=6,
    learning_rate=0.05,
)


def corpus_tensors(dataset) -> tuple[np.ndarray, np.ndarray]:
    """Stack the corpus `(src, frac)` through the public item interface."""
    loader = DataLoader(dataset, batch_size=8192, shuffle=False, num_workers=0)
    src, frac = [], []
    for batch in loader:
        src.append(batch["src"])
        frac.append(batch["frac"])
    return torch.cat(src).numpy(), torch.cat(frac).numpy()


def input_features(src: np.ndarray, frac: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Element values and presence flags, redox slot excluded."""
    values = frac[:, : len(VOCAB)]
    present = (src[:, : len(VOCAB)] > 0).astype(np.float32)
    names = [f"{sym}_value" for sym, _z in VOCAB] + [f"{sym}_present" for sym, _z in VOCAB]
    return np.concatenate([values, present], axis=1), names


def fit_and_score(x, y, fit_idx, hold_idx, test_idx, seed: int = 0) -> tuple[dict, XGBClassifier]:
    model = XGBClassifier(random_state=seed, **XGB_KWARGS)
    model.fit(
        x[fit_idx], y[fit_idx],
        eval_set=[(x[hold_idx], y[hold_idx])],
        verbose=False,
    )
    p = model.predict_proba(x[test_idx])[:, 1]
    return {
        "n_fit": int(len(fit_idx)),
        "n_test": int(len(test_idx)),
        "base_rate": round(float(y[test_idx].mean()), 4),
        "auroc": round(float(roc_auc_score(y[test_idx], p)), 4),
        "pr_auc": round(float(average_precision_score(y[test_idx], p)), 4),
        "best_iteration": int(model.best_iteration),
    }, model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", default=ENCODER)
    parser.add_argument(
        "--arms", default="inputs,latent",
        help="comma-separated subset of inputs,latent",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="subsample the corpus to N rows before splitting (smoke runs)",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    encoder, ckpt = load_encoder_from_checkpoint(REPO / args.encoder)
    config = ckpt.config
    print(f"building the SSL corpus from {args.encoder} (epoch {ckpt.epoch})...")
    dataset = create_dataset(config)
    splits = reserve_test(dataset, config)
    print(f"corpus n={len(dataset)} trainval={len(splits['trainval'])} "
          f"test={len(splits['test'])}")

    src, frac = corpus_tensors(dataset)
    y = (src[:, REDOX_TOKEN_INDEX] == REDOX_TOKEN_ID).astype(np.int8)
    print(f"redox populated: {y.mean():.1%} of rows")

    rng = np.random.default_rng(args.seed)
    trainval = np.asarray(splits["trainval"])
    test_idx = np.asarray(splits["test"])
    if args.limit:
        trainval = rng.choice(trainval, min(args.limit, len(trainval)), replace=False)
        test_idx = rng.choice(test_idx, min(args.limit // 4, len(test_idx)), replace=False)
    shuffled = rng.permutation(trainval)
    cut = int(0.9 * len(shuffled))
    fit_idx, hold_idx = shuffled[:cut], shuffled[cut:]

    out = {
        "encoder": args.encoder,
        "corpus_rows": int(len(dataset)),
        "populated_fraction": round(float(y.mean()), 4),
        "arms": {},
    }

    if "inputs" in arms:
        print("\ninputs arm: element values + presence flags")
        x, names = input_features(src, frac)
        scores, model = fit_and_score(x, y, fit_idx, hold_idx, test_idx, args.seed)
        order = np.argsort(model.feature_importances_)[::-1][:10]
        scores["top_features"] = [
            (names[i], round(float(model.feature_importances_[i]), 4)) for i in order
        ]
        out["arms"]["inputs"] = scores
        print(f"  AUROC {scores['auroc']}  PR-AUC {scores['pr_auc']}  "
              f"(base rate {scores['base_rate']})")
        print(f"  top: {', '.join(n for n, _ in scores['top_features'][:5])}")

    if "latent" in arms:
        print("\nlatent arm: frozen pooled latent, redox slot padded")
        device = pick_device()
        latents = extract_latents(
            encoder, PaddedRedoxView(dataset), device=device,
        ).numpy()
        scores, _ = fit_and_score(latents, y, fit_idx, hold_idx, test_idx, args.seed)
        out["arms"]["latent"] = scores
        print(f"  AUROC {scores['auroc']}  PR-AUC {scores['pr_auc']}  "
              f"(base rate {scores['base_rate']})")

    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / "missingness.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
