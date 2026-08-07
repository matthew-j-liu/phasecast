"""Tripwire test for the `sediment_transfer` exploration.

Locks the two pure pieces that decide whether the Andrews benchmark means what
the README says: the oxide inversion that has to round-trip through the shipped
tokenizer, and the mineral mapping that has to land inside the shipped
coarsening. The pickle itself lives outside the repo, so the loader's
end-to-end path runs from the CLI, not here.
"""

import numpy as np
import pandas as pd
import pytest

from cnmineral.data.bulk_chemistry import OXIDE_TO_CATION_FACTOR, tokenize_bulk_frame
from cnmineral.data.tokens import N_TOKENS, VOCAB
from cnmineral.eval.real_rock import TRUTH_TO_GROUP
from explorations.sediment_transfer.load_andrews import (
    ELEMENT_TO_OXIDE, MINERAL_TO_PHASE, _FACTOR_KEY,
)


def _oxide_frame(element_wt_frac: dict[str, np.ndarray]) -> pd.DataFrame:
    """The exploration's conversion, isolated from the pickle."""
    frame = pd.DataFrame({"sample_id": np.arange(len(next(iter(
        element_wt_frac.values()))))})
    for element, column in ELEMENT_TO_OXIDE.items():
        factor = OXIDE_TO_CATION_FACTOR[_FACTOR_KEY.get(column, column)]
        frame[column] = element_wt_frac[element] * 100.0 / factor
    return frame


def test_oxide_inversion_round_trips_through_the_tokenizer() -> None:
    """Every populated token must equal the source elemental weight fraction.

    This is what lets the README claim the tokenization introduces no
    assumption. It breaks if `OXIDE_TO_CATION_FACTOR` changes or if the Fe
    fallback stops reading `feo_tot` as FeO.
    """
    rng = np.random.default_rng(0)
    values = {element: rng.uniform(0.005, 0.3, size=16)
              for element in ELEMENT_TO_OXIDE}

    src, frac = tokenize_bulk_frame(_oxide_frame(values))

    token_of = {symbol: i for i, (symbol, _z) in enumerate(VOCAB)}
    for element, expected in values.items():
        j = token_of[element]
        assert np.allclose(frac[:, j], expected, atol=1e-7)
        assert (src[:, j] > 0).all()


def test_only_the_reported_elements_populate() -> None:
    """Andrews reports nine elements, seven of which we can carry. The other
    fifteen slots must pad, which is the load-bearing OOD fact in the README."""
    rng = np.random.default_rng(1)
    values = {element: rng.uniform(0.005, 0.3, size=8)
              for element in ELEMENT_TO_OXIDE}

    src, _frac = tokenize_bulk_frame(_oxide_frame(values))

    assert src.shape[1] == N_TOKENS
    assert (src > 0).sum(1).tolist() == [len(ELEMENT_TO_OXIDE)] * 8


@pytest.mark.parametrize("phase", sorted(set(MINERAL_TO_PHASE.values())))
def test_every_mapped_mineral_lands_in_the_shipped_coarsening(phase: str) -> None:
    """The exploration hands `_truth_arrays` its own `magemin_phase` strings;
    any that `TRUTH_TO_GROUP` does not know would be silently dropped and the
    truth would renormalize over the wrong groups."""
    assert phase in TRUTH_TO_GROUP


def test_olivine_and_biotite_have_no_source_mineral() -> None:
    """Two of the seven groups are absent from Andrews by construction, which
    is why the magnitude branch is scored on five groups and presence on
    seven. If a mapping ever supplies them, the README's framing is stale."""
    groups = {TRUTH_TO_GROUP[p] for p in MINERAL_TO_PHASE.values()}
    assert "ol" not in groups
    assert "bi" not in groups
