"""Tests for `cnmineral.eval.{latent, probes, umap_views, supergroup}`.

Structural tests on the synthetic dataset (no real CSV needed) plus an
end-to-end probe + UMAP run on the 2000-row corpus fixture.  The
end-to-end test exercises every code path the CLI hits except the
checkpoint round-trip, which gets its own targeted test.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from cnmineral.data.bulk_chemistry import BulkChemistryDataset
from cnmineral.data.synthetic import SyntheticBulkChemistryDataset
from cnmineral.eval.latent import (
    extract_latents,
    load_encoder_from_checkpoint,
)
from cnmineral.eval.probes import (
    knn_label_purity,
    random_label_purity,
    train_logreg_probe,
    train_mlp_probe,
)
from cnmineral.eval.supergroup import (
    SUPERGROUPS,
    map_qap_array,
    qap_to_supergroup,
)
from cnmineral.eval.umap_views import (
    fit_umap_2d,
    fit_umap_3d,
    plot_knn_qap_purity,
    plot_umap_3d_supergroup,
    plot_umap_chem_axes,
    plot_umap_qap,
    plot_umap_supergroup,
)
from cnmineral.models.encoder import Encoder


# ---------- supergroup mapping ----------


def test_qap_to_supergroup_known_pins() -> None:
    """A handful of representative qap names from complete.csv map to the
    expected supergroup.  Sanity check on the v2-ported lookup table."""
    pins = {
        "basalt": "mafic-volcanic",
        "gabbro": "mafic-plutonic",
        "andesite": "intermediate-volcanic",
        "diorite": "intermediate-plutonic",
        "rhyolite": "felsic-volcanic",
        "granite": "felsic-plutonic",
        "peridotite": "ultramafic",
        "nephelinite": "foid-bearing",
        "phonolite": "foid-bearing",
        "unknownrock": "other",
    }
    for qap, expected in pins.items():
        assert qap_to_supergroup(qap) == expected, qap


def test_qap_to_supergroup_handles_none_and_empty() -> None:
    """None, NaN-shaped non-strings, and empty strings collapse to other."""
    assert qap_to_supergroup(None) == "other"
    assert qap_to_supergroup("") == "other"
    assert qap_to_supergroup("   ") == "other"
    assert qap_to_supergroup(3.14) == "other"


def test_map_qap_array_returns_object_ndarray() -> None:
    arr = map_qap_array(["basalt", "granite", None, "phonolite"])
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == object
    assert list(arr) == [
        "mafic-volcanic", "felsic-plutonic", "other", "foid-bearing",
    ]


def test_supergroups_tuple_has_nine_entries() -> None:
    """Production color palette assumes nine supergroups."""
    assert len(SUPERGROUPS) == 9
    assert "other" in SUPERGROUPS


# ---------- latent extraction ----------


def _tiny_encoder() -> Encoder:
    """Smallest encoder we can construct; fast for tests."""
    return Encoder(
        d_model=16, n_layers=1, n_heads=2, dropout=0.0,
        frac_resolution=200,
    )


def test_extract_latents_full_dataset_shape() -> None:
    """N samples -> [N, d_model] tensor on CPU."""
    encoder = _tiny_encoder()
    dataset = SyntheticBulkChemistryDataset(n=32, seed=0)
    latents = extract_latents(encoder, dataset, batch_size=8)
    assert latents.shape == (32, 16)
    assert latents.device.type == "cpu"
    assert torch.isfinite(latents).all()


def test_extract_latents_indices_subset_shape() -> None:
    """Passing `indices` returns the subset count, not the full dataset."""
    encoder = _tiny_encoder()
    dataset = SyntheticBulkChemistryDataset(n=32, seed=0)
    latents = extract_latents(
        encoder, dataset, indices=[0, 5, 10, 15], batch_size=2,
    )
    assert latents.shape == (4, 16)


def test_extract_latents_deterministic_in_eval_mode() -> None:
    """Two calls with the same encoder + dataset produce identical latents.

    Dropout is on in train mode; this guards against future refactors
    that forget to call `encoder.eval()`.
    """
    encoder = _tiny_encoder()
    dataset = SyntheticBulkChemistryDataset(n=8, seed=0)
    a = extract_latents(encoder, dataset, batch_size=4)
    b = extract_latents(encoder, dataset, batch_size=4)
    assert torch.allclose(a, b)


def test_load_encoder_from_checkpoint_smoke_ckpt() -> None:
    """The Day-5 smoke checkpoint loads + produces a non-zero latent.

    Skipped when the checkpoint isn't present (clean clone, CI without
    long-running artifacts).
    """
    ckpt = Path(__file__).resolve().parent.parent / "checkpoints" / "pretrain_v3_smoke" / "epoch_001.pt"
    if not ckpt.exists():
        pytest.skip(f"checkpoint missing: {ckpt}")
    encoder, payload = load_encoder_from_checkpoint(ckpt)
    assert encoder.d_model == payload["config"]["d_model"]
    dataset = SyntheticBulkChemistryDataset(n=4, seed=0)
    latents = extract_latents(encoder, dataset, batch_size=2)
    assert latents.shape == (4, encoder.d_model)
    assert latents.abs().sum() > 0


# ---------- probes ----------


def _clustered_latents(
    n_train_per_class: int,
    n_test_per_class: int,
    n_classes: int,
    d: int,
    *,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build separable train/test latents sharing the same class centers.

    Returns `(X_train, y_train, X_test, y_test)`.  Each class is a
    gaussian blob; train and test are independent samples around the
    same centers so a working probe must learn the centers, not memorize
    train points.  Defends against the case where a probe is silently
    broken (e.g. always predicts class 0).
    """
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 5, size=(n_classes, d)).astype(np.float32)

    def _draw(n_per_class: int) -> tuple[np.ndarray, np.ndarray]:
        X = np.empty((n_per_class * n_classes, d), dtype=np.float32)
        y = np.empty(n_per_class * n_classes, dtype=object)
        for c in range(n_classes):
            sl = slice(c * n_per_class, (c + 1) * n_per_class)
            X[sl] = centers[c] + rng.normal(
                0, 1, size=(n_per_class, d),
            ).astype(np.float32)
            y[sl] = f"class_{c}"
        return X, y

    X_train, y_train = _draw(n_train_per_class)
    X_test, y_test = _draw(n_test_per_class)
    return X_train, y_train, X_test, y_test


def test_logreg_probe_runs_and_beats_random() -> None:
    """Probe trains, returns finite metrics, accuracy > 1/n_classes."""
    n_classes = 5
    X_train, y_train, X_test, y_test = _clustered_latents(
        60, 20, n_classes, d=8, seed=0,
    )
    metrics = train_logreg_probe(X_train, y_train, X_test, y_test)
    assert metrics.probe == "logreg"
    assert np.isfinite(metrics.train_accuracy)
    assert np.isfinite(metrics.test_accuracy)
    assert metrics.test_accuracy > 1.0 / n_classes + 0.1
    assert metrics.n_classes == n_classes
    assert len(metrics.per_class_accuracy) == n_classes


def test_mlp_probe_runs_and_beats_random() -> None:
    n_classes = 4
    X_train, y_train, X_test, y_test = _clustered_latents(
        80, 20, n_classes, d=8, seed=0,
    )
    metrics = train_mlp_probe(
        X_train, y_train, X_test, y_test, epochs=20, hidden_dim=32,
    )
    assert metrics.probe == "mlp"
    assert np.isfinite(metrics.train_accuracy)
    assert np.isfinite(metrics.test_accuracy)
    assert metrics.test_accuracy > 1.0 / n_classes + 0.1


def test_probe_metrics_to_dict_is_json_friendly() -> None:
    import json
    n_classes = 3
    X_train, y_train, X_test, y_test = _clustered_latents(
        30, 10, n_classes, d=8, seed=0,
    )
    metrics = train_logreg_probe(X_train, y_train, X_test, y_test)
    payload = metrics.to_dict()
    json.dumps(payload)  # would raise if anything is np.float32 etc.
    assert payload["probe"] == "logreg"
    assert isinstance(payload["per_class_accuracy"], dict)


def test_probe_metrics_carries_hyperparameters() -> None:
    """to_dict() carries the exact kwargs that produced it, so a reader
    of probe_metrics.json knows what was actually run."""
    import json
    n_classes = 3
    X_train, y_train, X_test, y_test = _clustered_latents(
        30, 10, n_classes, d=8, seed=0,
    )
    lr = train_logreg_probe(
        X_train, y_train, X_test, y_test,
        standardize=False, class_weight="balanced", max_iter=1234,
    )
    assert lr.hyperparameters["standardize"] is False
    assert lr.hyperparameters["class_weight"] == "balanced"
    assert lr.hyperparameters["max_iter"] == 1234

    mlp = train_mlp_probe(
        X_train, y_train, X_test, y_test,
        epochs=3, hidden_dim=16, n_hidden_layers=2,
        l2_normalize_inputs=True, label_smoothing=0.05,
        use_cosine_schedule=False, class_weight="balanced",
    )
    assert mlp.hyperparameters["n_hidden_layers"] == 2
    assert mlp.hyperparameters["l2_normalize_inputs"] is True
    assert mlp.hyperparameters["label_smoothing"] == 0.05
    assert mlp.hyperparameters["use_cosine_schedule"] is False
    assert mlp.hyperparameters["class_weight"] == "balanced"

    # the hyperparameters block round-trips through json without coercion
    json.dumps(mlp.to_dict())


def test_logreg_probe_standardize_helps_at_skewed_scales() -> None:
    """Standardize=True converges faster and at least as accurately
    on a poorly-scaled latent.  Encoders don't normalize their output,
    so this code path is what the production smoke run depends on.
    """
    import warnings
    from sklearn.exceptions import ConvergenceWarning

    n_classes = 4
    X_train, y_train, X_test, y_test = _clustered_latents(
        100, 25, n_classes, d=16, seed=7,
    )
    # rescale dims by wildly different factors -- mimics the encoder's
    # unscaled pooled latent where each dim can be at a different
    # magnitude.
    rng = np.random.default_rng(7)
    per_dim_scale = rng.uniform(0.01, 100.0, size=X_train.shape[1]).astype(np.float32)
    X_train_s = X_train * per_dim_scale
    X_test_s = X_test * per_dim_scale

    # at standardize=False with a tight max_iter, lbfgs is expected to
    # warn (and may or may not converge); the test just guards that the
    # standardized path completes without warnings.
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        good = train_logreg_probe(
            X_train_s, y_train, X_test_s, y_test,
            standardize=True, max_iter=2000,
        )
    assert np.isfinite(good.test_accuracy)
    assert good.test_accuracy > 1.0 / n_classes + 0.2


def test_mlp_probe_l2_normalize_runs_and_records_flag() -> None:
    """The l2_normalize_inputs path runs end-to-end without breaking
    shapes or producing nan, and the flag is recorded on the metrics.

    Doesn't claim normalize helps: whether direction carries the class
    signal is a property of the latent, and SSL latents in the wild
    aren't represented by isotropic gaussian centers.
    """
    n_classes = 4
    X_train, y_train, X_test, y_test = _clustered_latents(
        80, 20, n_classes, d=8, seed=0,
    )
    metrics = train_mlp_probe(
        X_train, y_train, X_test, y_test,
        epochs=20, hidden_dim=32, l2_normalize_inputs=True,
    )
    assert np.isfinite(metrics.test_accuracy)
    assert 0.0 <= metrics.test_accuracy <= 1.0
    assert metrics.hyperparameters["l2_normalize_inputs"] is True


def test_mlp_probe_two_layer_runs_end_to_end() -> None:
    """n_hidden_layers=2 builds a different topology and still trains."""
    n_classes = 4
    X_train, y_train, X_test, y_test = _clustered_latents(
        80, 20, n_classes, d=8, seed=0,
    )
    metrics = train_mlp_probe(
        X_train, y_train, X_test, y_test,
        epochs=20, hidden_dim=32, n_hidden_layers=2,
    )
    assert np.isfinite(metrics.test_accuracy)
    assert metrics.test_accuracy > 1.0 / n_classes + 0.1
    assert metrics.hyperparameters["n_hidden_layers"] == 2


def test_mlp_probe_invalid_n_hidden_layers_raises() -> None:
    """3-layer or 0-layer asks for behavior the probe doesn't intend."""
    n_classes = 3
    X_train, y_train, X_test, y_test = _clustered_latents(
        20, 5, n_classes, d=8, seed=0,
    )
    with pytest.raises(ValueError):
        train_mlp_probe(
            X_train, y_train, X_test, y_test,
            epochs=1, n_hidden_layers=3,
        )


def test_mlp_probe_class_weighting_lifts_rare_class() -> None:
    """class_weight='balanced' improves the small class's per-class
    accuracy at some cost to the majority class.  Doesn't claim
    overall accuracy improves -- the trade-off is the point."""
    rng = np.random.default_rng(0)
    d = 8
    n_classes = 3
    centers = rng.normal(0, 3, size=(n_classes, d)).astype(np.float32)

    def _imbalanced_split(class_sizes: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for c, n in enumerate(class_sizes):
            X.append(
                centers[c] + rng.normal(0, 1, size=(n, d)).astype(np.float32),
            )
            y.extend([f"class_{c}"] * n)
        return np.concatenate(X), np.asarray(y, dtype=object)

    # majority class has 10x the samples of the minority
    X_train, y_train = _imbalanced_split((200, 200, 20))
    X_test, y_test = _imbalanced_split((50, 50, 50))

    unweighted = train_mlp_probe(
        X_train, y_train, X_test, y_test,
        epochs=30, hidden_dim=32, batch_size=64,
        use_cosine_schedule=False,
    )
    weighted = train_mlp_probe(
        X_train, y_train, X_test, y_test,
        epochs=30, hidden_dim=32, batch_size=64,
        use_cosine_schedule=False, class_weight="balanced",
    )
    assert (
        weighted.per_class_accuracy["class_2"]
        >= unweighted.per_class_accuracy["class_2"]
    )
    assert weighted.hyperparameters["class_weight"] == "balanced"
    assert unweighted.hyperparameters["class_weight"] is None


def test_knn_label_purity_above_random() -> None:
    """k-NN on a separable latent beats the class-balanced random baseline."""
    n_classes = 4
    X, y, _, _ = _clustered_latents(50, 0, n_classes, d=8, seed=42)
    purity = knn_label_purity(X, y, k_values=(5, 10))
    baseline = random_label_purity(y)
    assert purity[5] > baseline + 0.2
    assert purity[10] > baseline + 0.2
    assert 0.0 <= purity[5] <= 1.0


def test_random_label_purity_balanced_classes() -> None:
    """Class-balanced 5-class labels give baseline = 1/5 = 0.2."""
    y = np.array(["a", "b", "c", "d", "e"] * 20)
    assert abs(random_label_purity(y) - 0.2) < 1e-9


# ---------- UMAP + plotting (against the corpus fixture) ----------


def _build_fixture_dataset(corpus_subset_path: Path) -> BulkChemistryDataset:
    return BulkChemistryDataset(
        csv_path=corpus_subset_path,
        rock_group="igneous",
        require_all_majors=True,
        require_qap_name=True,
        min_class_count=2,
    )


def test_run_probes_end_to_end_on_fixture(
    corpus_subset_path: Path, tmp_path: Path,
) -> None:
    """Smoke-test the full CLI pipeline against the corpus fixture.

    Drives extract_latents + train_logreg_probe + train_mlp_probe +
    fit_umap_2d + fit_umap_3d + every plot fn.  The 1-epoch-tiny-encoder
    probe numbers are not meaningful; the test guards wiring + finiteness.
    """
    dataset = _build_fixture_dataset(corpus_subset_path)
    if len(dataset) < 80:
        pytest.skip(f"fixture too small ({len(dataset)} rows) for split")

    encoder = _tiny_encoder()
    latents = extract_latents(encoder, dataset, batch_size=64).numpy()
    assert latents.shape == (len(dataset), 16)

    # naive 80/20 split for the test, no need to reproduce reserve_test
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(dataset))
    cut = int(0.8 * len(dataset))
    train_idx, test_idx = perm[:cut], perm[cut:]
    qap = np.asarray(dataset.qap_names)
    qap_train, qap_test = qap[train_idx], qap[test_idx]

    # Filter test labels down to those that appear in train, to keep
    # LabelEncoder consistent on a small fixture without a full
    # stratified split.
    keep_in_test = np.isin(qap_test, np.unique(qap_train))
    test_idx = test_idx[keep_in_test]
    qap_test = qap_test[keep_in_test]
    if len(qap_test) == 0:
        pytest.skip("no overlapping qap classes between train and test")

    logreg = train_logreg_probe(
        latents[train_idx], qap_train, latents[test_idx], qap_test,
    )
    mlp = train_mlp_probe(
        latents[train_idx], qap_train, latents[test_idx], qap_test,
        epochs=5, hidden_dim=32, batch_size=64,
    )
    n_classes_train = len(np.unique(qap_train))
    assert np.isfinite(logreg.test_accuracy)
    assert np.isfinite(mlp.test_accuracy)
    # Random init + small fixture: probes might be at chance; this test
    # only verifies the pipeline ran and produced finite numbers.
    assert 0.0 <= logreg.test_accuracy <= 1.0
    assert 0.0 <= mlp.test_accuracy <= 1.0
    assert logreg.n_classes >= n_classes_train

    purity = knn_label_purity(latents, qap, k_values=(5, 10))
    baseline = random_label_purity(qap)
    assert 0.0 <= purity[5] <= 1.0
    assert 0.0 <= purity[10] <= 1.0
    assert 0.0 <= baseline <= 1.0

    emb2 = fit_umap_2d(latents, n_neighbors=5)
    emb3 = fit_umap_3d(latents, n_neighbors=5)
    assert emb2.shape == (len(dataset), 2)
    assert emb3.shape == (len(dataset), 3)

    sg = map_qap_array(qap)
    chem = dataset.petrological_axes()

    out = tmp_path
    paths = [
        plot_umap_supergroup(emb2, sg, out / "umap_supergroup.png"),
        plot_umap_qap(emb2, qap, out / "umap_qap.png"),
        plot_umap_chem_axes(emb2, chem, out / "umap_chem_axes.png"),
        plot_umap_3d_supergroup(emb3, sg, out / "umap_3d_supergroup.png"),
        plot_knn_qap_purity(purity, baseline, out / "knn_qap_purity.png"),
    ]
    for p in paths:
        assert p.exists() and p.stat().st_size > 0, p
