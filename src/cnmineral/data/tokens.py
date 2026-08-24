"""
Token vocabulary and masking convention.

The encoder consumes a fixed-length sequence of 22 element tokens per
sample.  Each token is described by two integer/float values:

- `src`: the element's atomic number (e.g. 14 for Si).  Special values:
    - `0` marks an *absent* slot (padding).
    - `MASK_TOKEN_ID` (= 119, one past the periodic table) marks a
      SSL-masked slot whose original value was hidden for pretraining.
- `frac`: the element's normalized abundance in `[0, 1]`.  Majors use
  `wt% / 100`; traces use `ppm / 1e6`.  For SSL-masked tokens,
  `frac = 0` (the value to predict).

padding = `(src=0, frac=0)`
populated = `(src in [1..118], frac > 0)`
masked = `(src=MASK_TOKEN_ID, frac=0)` # distinguishable from padding!

So suppose we have a rock sample where K2O says 0 wt% in `complete.csv` (1279 rows
have these explicit 0's). We would treat this as a NaN entry, i.e.,
drop K from the rock's tokens

Corpora built with `include_redox=True` carry an optional 23rd token
(`REDOX_TOKEN_ID`, Fe3+/total-Fe ratio in `frac`); everything above
applies to it unchanged.
"""

from dataclasses import dataclass
import torch

# Fix order vocabulary: token i in `src`/`frac` corresponds to entry i in this list.
MAJORS: list[tuple[str, int]] = [
    ("Si", 14), ("Al", 13), ("Ca", 20), ("Mg", 12), ("Fe", 26),
    ("K", 19),  ("Na", 11), ("Ti", 22), ("Cr", 24),
]

TRACES: list[tuple[str, int]] = [
    ("Sr", 38), ("Zr", 40), ("Ba", 56), ("Y", 39),  ("Rb", 37),
    ("Ni", 28), ("Nb", 41), ("V", 23),  ("La", 57), ("Ce", 58),
    ("Nd", 60), ("Yb", 70), ("Th", 90),
]

VOCAB: list[tuple[str, int]] = MAJORS + TRACES
N_TOKENS: int = len(VOCAB) # 22

# Pre-computed lookup: position index -> atomic number.
ATOMIC_NUMBERS: torch.Tensor = torch.tensor(
    [z for element_symbol, z in VOCAB], dtype=torch.long,
)

# position:    0    1    2    3    4    5    6    7    8    9   10  ...  21
# element:    Si   Al   Ca   Mg   Fe    K   Na   Ti   Cr   Sr   Zr  ...  Th
# src (Z):    14   13   20   12   26   19   11   22   24   38   40  ...  90

# If all rocks in a batch have all 22 elements reported (no padding), then every row
# in the src tensor would just look identical


# Reserved SSL mask token.  One past the periodic table (Z=118 is the
# heaviest known element); the encoder's element embedding table
# reserves one extra row for this id and treats it as a learnable
# "[MASK]" embedding analogous to BERT's mask token.
MASK_TOKEN_ID: int = 119

# Optional 23rd token: Fe redox state.  `frac` carries the Fe3+/total-Fe
# cation ratio, populated only where the measured FeO+Fe2O3 pair exists
# and padded (src=0, frac=0) elsewhere -- the padded state is a distinct
# learned "no speciation reported" state, NOT an implicit average over
# the unknown ratio (measured 2026-08-06: it imitates a low fixed
# ratio and barely tracks the rock).  `src` uses
# the next sentinel past MASK; the
# encoder reserves a learnable embedding row for it when built with
# `include_redox_token=True`.  A populated redox token is a full citizen
# of SSL masking (`mask_for_ssl` sees `src > 0`).
REDOX_TOKEN_ID: int = 120
REDOX_TOKEN_INDEX: int = 22
N_TOKENS_REDOX: int = N_TOKENS + 1


@dataclass(frozen=True, slots=True)
class MaskedBatch:
    """
    Output of `mask_for_ssl`

    B = batch size = how many rocks the model processes in parallel in one forward pass
    T = N_TOKENS = 22 = how many elements per rock

    Attributes
    ----------
    src_corrupted, frac_corrupted
        Inputs to the encoder.  Masked positions have
        `src=MASK_TOKEN_ID, frac=0`.  Padding positions still have
        `src=0, frac=0`
    mask_positions
        Boolean tensor of shape `[B, T]`.  `True` where a
        token was originally populated AND was selected for masking.
        The SSL loss reads its targets at exactly these positions.
    target_frac
        The original `frac` values before masking, used as the
        reconstruction target at `mask_positions`.  Same shape as
        `frac_corrupted`.
    """

    src_corrupted: torch.Tensor
    frac_corrupted: torch.Tensor
    mask_positions: torch.Tensor
    target_frac: torch.Tensor


def mask_for_ssl(
    src: torch.Tensor,
    frac: torch.Tensor,
    mask_prob: float = 0.25,
    generator: torch.Generator | None = None,
) -> MaskedBatch:
    """
    Apply SSL masking to a batch.

    For each sample, identify populated tokens (those with `src > 0`
    AND `src != MASK_TOKEN_ID` -- though real input data never carries
    MASK_TOKEN_ID) and select a fraction `mask_prob` of them uniformly
    at random for masking.  Masked positions get
    `src = MASK_TOKEN_ID, frac = 0`: BERT-style, the mask is a
    distinct sentinel the encoder can attend to.

    Parameters
    ----------
    src
        Token atomic numbers, shape `[B, N_TOKENS]`, dtype long.
        `src[b, i] == 0` indicates the i-th token is absent for sample b.
        Real input data must not contain `MASK_TOKEN_ID`.
    frac
        Token normalized values, shape `[B, N_TOKENS]`, dtype float.
        For populated tokens, `frac > 0` (mass fractions, never exactly
        zero in real data).  For absent tokens, `frac == 0`.
    mask_prob
        Per-populated-token masking probability.  v2 used 0.25; v3 default
        matches.
    generator 
        Optional torch RNG for reproducibility 

    Returns
    -------
    MaskedBatch
        See `MaskedBatch` docstring for field semantics.

    Notes
    -----
    Per-sample mask count is stochastic; with `mask_prob=0.25` and 9
    populated majors a sample averages ~2 masked positions.  Loss
    averaging in `cnmineral.losses.pretrain_loss.masked_element_log_mse`
    handles the variability.
    """

    # define a boolean mask for which entries will be masked
    # ----------------------------------------------------------------------------------------------------------
    populated = src > 0 # [B, T] boolean Tensor 
    mask_draws = torch.rand(size=src.shape, generator=generator, device=src.device) # [B, T] random numbers 0-1 

    # define the positions we mask: entry must be True in populated and less than mask_prob in mask_draws 
    # note that `and` is Python boolean logic; & is elementwise (what you want for vectorized ops)
    # A = tensor([True, False, True])
    # B = tensor([True, True, False]) 
    # `A and B` gives an error, while
    # `A & B` gives tensor([True, False, False])
    # also note that mask_draws[populated] would return a 1D tensor 
    # would only be helpful for like counting masked tokens 
    mask_positions = populated & (mask_draws < mask_prob) 

    # use the mask to create tensors that are like src and frac,
    # but with MASK_TOKEN_ID and 0 in the entries where the mask is True 
    # ----------------------------------------------------------------------------------------------------------

    src_corrupted = src.clone() # new memory allocation (but remains connected to the computation graph)
    frac_corrupted = frac.clone()

    src_corrupted[mask_positions] = MASK_TOKEN_ID
    frac_corrupted[mask_positions] = 0.0 

    return MaskedBatch(
        src_corrupted=src_corrupted,
        frac_corrupted=frac_corrupted,
        mask_positions=mask_positions,
        target_frac=frac)

def assert_padding_convention(src: torch.Tensor, frac: torch.Tensor) -> None:
    """
    Check if a batch is well-formed 

    Asserts that:

    - `src == 0` iff `frac == 0` (no half-populated tokens), AND
    - `MASK_TOKEN_ID` does not appear anywhere in `src`

    Do NOT call this function on `MaskedBatch` outputs
    """

    src_zero = src == 0
    frac_zero = frac == 0
    if not torch.equal(src_zero, frac_zero):
        bad = (src_zero ^ frac_zero).sum().item()
        raise ValueError(
            f"src/frac padding convention violated at {bad} positions: "
            f"src==0 must coincide with frac==0 in the v3 input format"
        )
    if (src == MASK_TOKEN_ID).any():
        n_mask = (src == MASK_TOKEN_ID).sum().item()
        raise ValueError(
            f"MASK_TOKEN_ID ({MASK_TOKEN_ID}) appears in input data at "
            f"{n_mask} positions.  This sentinel is reserved for SSL "
            f"corruption only; real input must not contain it."
        )
