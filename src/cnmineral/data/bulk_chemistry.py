"""
Reads `data/bulks/gard2019.csv` (latin-1 encoded, 1.02M rows, 192 columns),
filters to igneous rocks with all 9 major oxides reported, converts
oxide wt% to cation wt%, and emits per-sample `{src, frac, sample_id}`
dicts in the shape the training loop consumes.

The 22-token vocabulary lives in `cnmineral.data.tokens`: 9 majors
(Si Al Ca Mg Fe K Na Ti Cr) + 13 traces (Sr Zr Ba Y Rb Ni Nb V La Ce
Nd Yb Th).  Each token carries `(src=atomic_number, frac in [0, 1])`
for populated slots and `(src=0, frac=0)` for padded slots.

How to handle Fe: 
  - Under `require_all_majors=True` (the default): `feo` and `fe2o3`
    must both be reported.  Fe cation wt% = feo * 0.7773 + fe2o3 *
    0.6994.  The fallback chain does not fire.
  - Under `require_all_majors=False`: feo or fe2o3 alone is sufficient
    (the missing one is treated as 0).  If both are NaN, fall back to
    `feo_tot` first, then `fe2o3_tot`.  If no iron source exists, the
    Fe slot is padded.

Default filter (`require_all_majors=True`) yields ~167k igneous bulks
with clean majors.  Setting `require_all_majors=False` exercises the
Fe fallback chain and yields ~493k rows.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from cnmineral.data.tokens import MAJORS, TRACES, N_TOKENS


OXIDE_TO_CATION_FACTOR: dict[str, float] = {
    "sio2": 0.4675,
    "al2o3": 0.5293,
    "cao": 0.7147,
    "mgo": 0.6030,
    "feo": 0.7773,
    "fe2o3": 0.6994,
    "k2o": 0.8302,
    "na2o": 0.7419,
    "tio2": 0.5995,
    "cr2o3": 0.6842,
}

# token-position -> CSV column. Order matches `MAJORS + TRACES` in
# tokens.py. Fe is computed by `_resolve_iron`, not pulled from a
# single column, so the table never reads it.
MAJOR_TOKEN_TO_COL: dict[str, str] = {
    "Si": "sio2",
    "Al": "al2o3",
    "Ca": "cao",
    "Mg": "mgo",
    "K": "k2o",
    "Na": "na2o",
    "Ti": "tio2",
    "Cr": "cr2o3",
}

# Iron source columns read alongside the majors so `_resolve_iron`
# can compute the Fe cation slot.  None of these is a token-position
# itself, so they live outside MAJOR_TOKEN_TO_COL.
IRON_SOURCE_COLS: tuple[str, ...] = ("feo", "fe2o3", "feo_tot", "fe2o3_tot")

TRACE_TOKEN_TO_COL: dict[str, str] = {
    sym: f"{sym.lower()}_ppm" for sym, _z in TRACES
}

# CSV columns read on construction.  Single source of truth across the
# raw-load path and the per-column zero/negative filters.
_ALL_USED_COLS: tuple[str, ...] = (
    "sample_id", "qap_name", "rock_group",
    *MAJOR_TOKEN_TO_COL.values(),
    *IRON_SOURCE_COLS,
    *TRACE_TOKEN_TO_COL.values(),
)

# Per-column dtypes used when parsing the CSV.  float32 (~7 sig figs)
# is plenty for geochemistry; the two string columns are low-cardinality
# (~70 and 3 unique values) so category gives constant-size codes
# instead of one Python string per row.
_NUMERIC_COLS: tuple[str, ...] = (
    *MAJOR_TOKEN_TO_COL.values(),
    *IRON_SOURCE_COLS,
    *TRACE_TOKEN_TO_COL.values(),
)
TIGHT_DTYPES: dict[str, str] = {
    "sample_id": "int32",
    "qap_name": "category",
    "rock_group": "category",
    **{c: "float32" for c in _NUMERIC_COLS},
}


def _load_raw_dataframe(csv_path: Path) -> pd.DataFrame:
    """Read the 28-column subset, preferring a parquet sidecar.

    First call on a CSV builds `csv_path.with_suffix('.parquet')` as
    a side effect; subsequent calls hit the parquet (~10x faster read,
    ~25x smaller on disk, ~10x lower transient RSS than CSV parsing).
    Filters and tensor construction happen downstream, so the cache is
    parameter-independent: one parquet serves every constructor knob.
    """
    parquet_path = csv_path.with_suffix(".parquet")
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        if set(_ALL_USED_COLS).issubset(df.columns):
            return df[list(_ALL_USED_COLS)]
        # parquet predates a column-list change; rebuild from CSV.

    df = pd.read_csv(
        csv_path,
        encoding="latin-1",
        usecols=list(_ALL_USED_COLS),
        dtype=TIGHT_DTYPES,
        low_memory=False,
    )
    df.to_parquet(parquet_path, index=False)
    return df


def _resolve_iron(df: pd.DataFrame) -> pd.Series:
    """Compute Fe cation wt% column with fallback to `feo_tot` / `fe2o3_tot`.

    Both the primary and fallback branches are computed for every row
    (less wasteful than it seems: ~5 ms total on 740k rows, because
    vectorized numpy on contiguous float32 columns is memory-bandwidth-
    bound regardless of which products get discarded).  Any "skip the
    unused branch" rewrite breaks SIMD packing and is measurably slower
    at our N.  Returns Fe cation wt%, NaN where no Fe source is available.
    """
    # primary path: feo + fe2o3 (treat individual NaN as 0 once at least
    # one is present, so partial measurements still contribute).
    feo_or_fe2o3 = df["feo"].notna() | df["fe2o3"].notna()
    fe_cation = (
        df["feo"].fillna(0.0) * OXIDE_TO_CATION_FACTOR["feo"]
        + df["fe2o3"].fillna(0.0) * OXIDE_TO_CATION_FACTOR["fe2o3"]
    )

    # fallback: feo_tot treated as FeO; then fe2o3_tot treated as Fe2O3.
    need_fallback = ~feo_or_fe2o3
    feotot_fill = need_fallback & df["feo_tot"].notna()
    fe2o3tot_fill = need_fallback & ~df["feo_tot"].notna() & df["fe2o3_tot"].notna()

    fe_cation = fe_cation.where(
        feo_or_fe2o3,
        other=np.where(
            feotot_fill,
            df["feo_tot"].fillna(0.0) * OXIDE_TO_CATION_FACTOR["feo"],
            np.where(
                fe2o3tot_fill,
                df["fe2o3_tot"].fillna(0.0) * OXIDE_TO_CATION_FACTOR["fe2o3"],
                np.nan,
            ),
        ),
    )
    return fe_cation


class BulkChemistryDataset(Dataset[dict[str, torch.Tensor | int]]):
    """
    SSL pretrain corpus over `data/bulks/gard2019.csv`.

    Parameters
    ----------
    csv_path
        Path to the source CSV.  Defaults to `data/bulks/gard2019.csv`.
    rock_group
        Value of the `rock_group` column to keep.  Defaults to
        `"igneous"`.  Pass `None` to skip the filter entirely.
    drop_zero_as_nan
        If True (default), an exact 0 in any oxide or trace column is
        treated as missing rather than as a measured zero.  Geochemists
        sometimes report 0 for below-detection-limit; the v1 pipeline's
        convention is to drop those slots.
    require_all_majors
        If True (default), drop rows where any of the 9 majors (Si Al
        Ca Mg Fe K Na Ti Cr -- with Cr ignored when `include_cr2o3` is
        False) is missing.  If False, keep rows with partial majors and
        pad the missing slots.
    include_cr2o3
        If True, treat Cr as a required major and populate the slot
        when present.  Cr2O3 coverage is ~2% on igneous; default False
        leaves the Cr slot padded for every row.
    require_qap_name
        If True (default), drop rows where `qap_name` is null.  Every
        surviving row carries a usable stratification label, which the
        downstream split logic relies on.  Set False to keep null-qap
        rows; the `qap_names` attribute then contains None entries.
    min_class_count
        Drop rows whose `qap_name` class has fewer than this many
        samples in the filtered corpus.  Default 25 matches the
        CLAUDE.md mandate for production splits and guarantees
        sklearn's `train_test_split(stratify=...)` accepts every
        class.  Only applied when `require_qap_name=True`.

    Attributes (populated at construction)
    --------------------------------------
    sample_ids : np.ndarray[int64]
        Per-row sample identifier from the source CSV.  Stable across
        constructions for split-index reproducibility.
    qap_names : np.ndarray
        Per-row IUGS QAPF rock name.  When `require_qap_name=True`
        every entry is a `str`; otherwise None entries mark unlabeled
        rows.  Carried so downstream split logic can stratify by rock
        class without rebuilding the dataset.
    """

    def __init__(
        self,
        csv_path: str | Path = "data/bulks/gard2019.csv",
        rock_group: str | None = "igneous",
        drop_zero_as_nan: bool = True,
        require_all_majors: bool = True,
        include_cr2o3: bool = False,
        require_qap_name: bool = True,
        min_class_count: int = 25,
    ) -> None:

        # read the 28 relevant columns from the csv file
        # 8 major oxides + cr2o3
        # 4 cols for Fe species
        # 13 traces
        # 3 identifiers + filter (sample_id, qap_name, and rock_group)
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"bulk corpus CSV not found at {csv_path}")

        df = _load_raw_dataframe(csv_path)

        if rock_group is not None:
            df = df[df["rock_group"] == rock_group]

        # zero-as-NaN runs before negative-drop so a row that has an
        # exact-0 in a major doesn't get kicked out by the negative
        # filter (and vice versa for trace columns).
        oxide_cols = [*MAJOR_TOKEN_TO_COL.values(), *IRON_SOURCE_COLS]
        trace_cols = list(TRACE_TOKEN_TO_COL.values())

        if drop_zero_as_nan:
            for c in oxide_cols + trace_cols:
                df.loc[df[c] == 0, c] = np.nan

        # negatives: drop the row if any oxide is negative (corrupt sample);
        # NaN-out individual traces if the trace alone is negative (one
        # bad trace measurement shouldn't discard an otherwise-good bulk).
        oxide_neg_mask = (df[oxide_cols] < 0).any(axis=1)
        df = df[~oxide_neg_mask]
        for c in trace_cols:
            df.loc[df[c] < 0, c] = np.nan

        # required-major filter: in strict mode, require feo AND fe2o3
        # both non-null (the v1 convention).  In loose mode, fall back
        # to feo_tot / fe2o3_tot when both primary Fe oxides are missing.
        if require_all_majors:
            required = ["sio2", "al2o3", "cao", "mgo", "feo", "fe2o3",
                        "k2o", "na2o", "tio2"]
            if include_cr2o3:
                required.append("cr2o3")
            df = df[df[required].notna().all(axis=1)]

        # qap_name filter: drop nulls and rare classes so the downstream
        # split can stratify on the full surviving label set.  The
        # filter runs before _resolve_iron / tensor build so we don't
        # waste vectorized work on rows that are about to be dropped.
        if require_qap_name:
            df = df[df["qap_name"].notna()]
            if min_class_count > 1:
                counts = df["qap_name"].value_counts()
                keep_classes = counts.index[counts >= min_class_count]
                df = df[df["qap_name"].isin(keep_classes)]

        # resolve Fe to a single cation-wt% column.  Under strict
        # require_all_majors=True every row already has both feo and
        # fe2o3, so _resolve_iron returns feo*0.7773 + fe2o3*0.6994.
        # Under loose mode the fallback chain fires.
        df = df.assign(_fe_cation_wt_pct=_resolve_iron(df))

        # build the per-sample tensors.  Each row gets its own (src, frac)
        # pair where src=0 marks padded slots.  Tensors are stacked into
        # two big [N, 22] arrays at construction time so __getitem__ is a
        # cheap slice.
        n = len(df)
        src = np.zeros((n, N_TOKENS), dtype=np.int64)
        frac = np.zeros((n, N_TOKENS), dtype=np.float32)

        for tok_idx, (sym, z) in enumerate(MAJORS):
            if sym == "Cr" and not include_cr2o3:
                continue
            if sym == "Fe":
                values = df["_fe_cation_wt_pct"].to_numpy(dtype=np.float32)
            else:
                csv_col = MAJOR_TOKEN_TO_COL[sym]
                raw_oxide = df[csv_col].to_numpy(dtype=np.float32)
                values = raw_oxide * OXIDE_TO_CATION_FACTOR[csv_col]
            populated = np.isfinite(values) & (values > 0)
            src[populated, tok_idx] = z
            frac[populated, tok_idx] = values[populated] / 100.0

        for trace_offset, (sym, z) in enumerate(TRACES):
            tok_idx = len(MAJORS) + trace_offset
            csv_col = TRACE_TOKEN_TO_COL[sym]
            ppm = df[csv_col].to_numpy(dtype=np.float32)
            populated = np.isfinite(ppm) & (ppm > 0)
            src[populated, tok_idx] = z
            frac[populated, tok_idx] = ppm[populated] / 1e6

        self._src = torch.from_numpy(src)
        self._frac = torch.from_numpy(frac)
        self.sample_ids: np.ndarray = df["sample_id"].to_numpy(dtype=np.int64)
        # qap_name: object dtype with None for missing (np.nan would also
        # work but stratification consumers expect None or str)
        self.qap_names: np.ndarray = np.array(
            [q if isinstance(q, str) else None for q in df["qap_name"]],
            dtype=object,
        )

    def __len__(self) -> int:
        return int(self._src.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int]:
        return {
            "src": self._src[idx].clone(),
            "frac": self._frac[idx].clone(),
            "sample_id": int(self.sample_ids[idx]),
        }

    def petrological_axes(
        self, indices: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """SiO2 wt%, Mg#, and Na2O+K2O wt% per row.

        Inverts the encoder-input cation conversion: SiO2 wt% comes
        back as `frac[Si] * 100 / OXIDE_TO_CATION_FACTOR['sio2']`.
        Token positions are `MAJORS + TRACES`: Si=0, Al=1, Ca=2,
        Mg=3, Fe=4, K=5, Na=6.  Padded slots map to NaN so the caller
        can filter at viz time.

        Parameters
        ----------
        indices
            Row indices to score.  Defaults to every row in the
            dataset.

        Returns
        -------
        dict[str, np.ndarray]
            Keys `"SiO2 (wt%)"`, `"Mg#"`, `"Na2O+K2O (wt%)"`.  Each
            value is a 1D array aligned with `indices`.
        """
        if indices is None:
            indices = np.arange(len(self))
        indices = np.asarray(indices, dtype=np.int64)
        src_sub = self._src.numpy()[indices]
        frac_sub = self._frac.numpy()[indices]

        def oxide_wt_pct(token_idx: int, oxide_key: str) -> np.ndarray:
            cation_wt = frac_sub[:, token_idx] * 100.0
            factor = OXIDE_TO_CATION_FACTOR[oxide_key]
            oxide = cation_wt / factor
            oxide[src_sub[:, token_idx] == 0] = np.nan
            return oxide

        sio2 = oxide_wt_pct(0, "sio2")
        mgo = oxide_wt_pct(3, "mgo")
        # `frac[Fe]` is already cation wt% (Fe is resolved at
        # construction, not via OXIDE_TO_CATION_FACTOR).  Molar mass
        # of Fe = 55.845 g/mol; molar mass of MgO = 40.30 g/mol.
        feo = frac_sub[:, 4] * 100.0
        feo[src_sub[:, 4] == 0] = np.nan
        na2o = oxide_wt_pct(6, "na2o")
        k2o = oxide_wt_pct(5, "k2o")

        mol_mgo = mgo / 40.30
        mol_feo = feo / 55.845
        denom = mol_mgo + mol_feo
        with np.errstate(invalid="ignore", divide="ignore"):
            mg_num = np.where(denom > 0, mol_mgo / denom, np.nan)

        return {
            "SiO2 (wt%)": sio2,
            "Mg#": mg_num,
            "Na2O+K2O (wt%)": na2o + k2o,
        }
