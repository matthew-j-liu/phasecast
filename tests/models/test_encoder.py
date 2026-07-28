"""Unit tests pinning the load-bearing v3 encoder invariants.

These tests exist specifically because v2 didn't have them: a regression
that masks gradient flow at masked-token positions is silent and only
detectable by inspecting downstream metrics (the v2 SSL pretraining
appeared to train cleanly for 8 epochs while the encoder weights never
updated).  Each test below fails on a specific failure mode that would
otherwise reach production unnoticed.

Run with:
    pytest tests/models/test_encoder.py -v
"""

import pytest
import torch

from cnmineral.data.tokens import (
    MASK_TOKEN_ID,
    N_TOKENS,
    REDOX_TOKEN_ID,
    ATOMIC_NUMBERS,
    assert_padding_convention,
    mask_for_ssl,
)
from cnmineral.models.encoder import (
    ELEMENT_EMBEDDING_ROWS,
    MAT2VEC_DIM,
    Encoder,
    load_mat2vec_matrix,
)
from cnmineral.models.pretrain_head import PretrainHead


# ---------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------

@pytest.fixture(scope="module")
def encoder() -> Encoder:
    """One encoder shared across tests in this module."""
    torch.manual_seed(0)
    return Encoder(d_model=64, n_layers=2, n_heads=4, dropout=0.0)


@pytest.fixture(scope="module")
def head() -> PretrainHead:
    """One pretrain head matched to the test encoder."""
    return PretrainHead(d_model=64)


def _make_batch(
    batch_size: int = 4,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Synthesize a batch of (src, frac) inputs that obeys the v3 convention.

    Populates the 9 majors deterministically with random ``frac``
    values in ``(0.01, 0.5]``; leaves the 13 trace slots padded
    (``src=0, frac=0``).  Useful for the smaller deterministic tests.
    """
    g = torch.Generator().manual_seed(seed)
    src = torch.zeros(batch_size, N_TOKENS, dtype=torch.long)
    frac = torch.zeros(batch_size, N_TOKENS, dtype=torch.float32)
    # Fill the first 9 token slots with the 9 majors.
    src[:, :9] = ATOMIC_NUMBERS[:9]
    frac[:, :9] = 0.01 + 0.49 * torch.rand(
        (batch_size, 9), generator=g,
    )
    return src, frac


# ---------------------------------------------------------------------
# Padding convention
# ---------------------------------------------------------------------

class TestPaddingConvention:
    """v3 invariant: src==0 iff frac==0 at the dataset boundary."""

    def test_valid_batch_passes(self) -> None:
        src, frac = _make_batch()
        assert_padding_convention(src, frac)  # should not raise

    def test_src_populated_but_frac_zero_fails(self) -> None:
        """A populated src with frac==0 is the v2 bug shape -- catch it."""
        src, frac = _make_batch()
        # Simulate the buggy v2 mask: keep src valid, zero frac.
        frac[0, 0] = 0.0
        with pytest.raises(ValueError, match="padding convention"):
            assert_padding_convention(src, frac)

    def test_src_zero_but_frac_populated_fails(self) -> None:
        src, frac = _make_batch()
        src[0, 0] = 0
        # frac[0, 0] is still > 0
        with pytest.raises(ValueError, match="padding convention"):
            assert_padding_convention(src, frac)


# ---------------------------------------------------------------------
# mat2vec loader direct tests
# ---------------------------------------------------------------------

class TestMat2vecLoading:
    """Direct tests for the mat2vec embedding matrix loader.

    The encoder fixture catches shape-mismatch errors in load_mat2vec_matrix
    but would happily accept a matrix where Si's data landed in Mg's row or
    a matrix that's all zeros.  These tests pin the value-level contract so
    such silent value bugs are caught directly rather than surfacing
    downstream as "the model trains but doesn't learn anything."
    """

    @pytest.fixture(scope="class")
    def matrix(self) -> torch.Tensor:
        """One matrix shared across tests in this class."""
        return load_mat2vec_matrix()

    def test_shape(self, matrix: torch.Tensor) -> None:
        assert matrix.shape == (ELEMENT_EMBEDDING_ROWS, MAT2VEC_DIM)

    def test_dtype_is_float32(self, matrix: torch.Tensor) -> None:
        """Must be float32 so nn.Embedding.from_pretrained doesn't upcast
        the embedding table to float64 (2x memory + dtype clashes with
        the float32 Linear layers downstream)."""
        assert matrix.dtype == torch.float32, (
            f"expected float32, got {matrix.dtype}; numpy defaults to "
            f"float64 -- cast at load time via dtype=torch.float32"
        )

    def test_padding_row_is_zero(self, matrix: torch.Tensor) -> None:
        """Row 0 = padding embedding; must stay at zeros so the encoder's
        padding handling matches the dataset convention (src=0 -> absent)."""
        assert (matrix[0] == 0).all()

    def test_mask_row_is_zero_before_encoder_rebinds(
        self, matrix: torch.Tensor,
    ) -> None:
        """Row MASK_TOKEN_ID is zero at load time; Encoder.__init__ later
        re-initializes it as N(0, 0.02).  The loader is not responsible
        for that re-init; it just must not leave junk there."""
        assert (matrix[MASK_TOKEN_ID] == 0).all()

    def test_real_elements_have_nonzero_embeddings(
        self, matrix: torch.Tensor,
    ) -> None:
        """Spot-check that rocks-relevant elements actually loaded.
        Si (Z=14), Fe (Z=26), Th (Z=90) -- our heaviest v3 trace."""
        for z in (14, 26, 90):
            assert matrix[z].abs().sum() > 0, (
                f"row {z} is all zeros; mat2vec for that element didn't "
                f"load (check the loader's symbol -> row-index mapping)"
            )

    def test_known_element_distinct_from_padding(
        self, matrix: torch.Tensor,
    ) -> None:
        """Trivial sanity check: Si's embedding shouldn't equal the padding
        embedding.  Catches "I wrote into the wrong row" bugs."""
        assert not torch.equal(matrix[14], matrix[0])


# ---------------------------------------------------------------------
# Embedding-table invariants the Encoder constructor must establish
# ---------------------------------------------------------------------

class TestEmbeddingTable:
    """The loader hands over zeros at rows 0 and MASK_TOKEN_ID; the
    Encoder is responsible for re-initializing the MASK row (so the
    mask token is learnable and distinct from padding) while keeping
    the padding row pinned at zero.  If the re-init silently broke,
    SSL would still train -- just with a degenerate mask embedding."""

    def test_mask_row_reinitialized_nonzero(self, encoder: Encoder) -> None:
        mask_row = encoder.element_embedding.weight[MASK_TOKEN_ID]
        assert mask_row.abs().sum() > 0, (
            "MASK row is all zeros after Encoder.__init__; the N(0, 0.02) "
            "re-init didn't happen and the mask token embeds like padding"
        )

    def test_padding_row_zero_after_init(self, encoder: Encoder) -> None:
        assert (encoder.element_embedding.weight[0] == 0).all()

    def test_padding_row_stays_zero_after_optimizer_step(self) -> None:
        """padding_idx=0 keeps the padding row gradient-free, and
        AdamW's decoupled weight decay multiplies a zero row by a
        scalar -- so row 0 must stay exactly zero through a real
        train step.  Guards against losing padding_idx in a refactor."""
        torch.manual_seed(0)
        encoder = Encoder(d_model=64, n_layers=1, n_heads=4, dropout=0.0)
        head = PretrainHead(d_model=64)
        optimizer = torch.optim.AdamW(
            list(encoder.parameters()) + list(head.parameters()),
            lr=1e-2, weight_decay=0.01,
        )
        src, frac = _make_batch(batch_size=8)
        masked = mask_for_ssl(src, frac, mask_prob=0.5)
        predicted = head(encoder(masked.src_corrupted, masked.frac_corrupted))
        m = masked.mask_positions
        loss = ((predicted - masked.target_frac) ** 2 * m).sum() / m.sum().clamp(min=1)
        loss.backward()
        optimizer.step()
        assert (encoder.element_embedding.weight[0] == 0).all(), (
            "padding row moved during training; padding_idx=0 contract broken"
        )


class TestRedoxTokenWidth:
    """`include_redox_token=True` widens the embedding table by one
    learnable row for REDOX_TOKEN_ID; the default reproduces the
    22-token table so pre-redox checkpoints load strictly."""

    def test_default_table_is_120_rows(self) -> None:
        encoder = Encoder(d_model=64, n_layers=1, n_heads=4)
        assert encoder.element_embedding.weight.shape[0] == ELEMENT_EMBEDDING_ROWS

    def test_redox_table_has_one_extra_row(self) -> None:
        encoder = Encoder(
            d_model=64, n_layers=1, n_heads=4, include_redox_token=True,
        )
        assert (
            encoder.element_embedding.weight.shape[0]
            == ELEMENT_EMBEDDING_ROWS + 1
        )

    def test_redox_row_reinitialized_nonzero(self) -> None:
        encoder = Encoder(
            d_model=64, n_layers=1, n_heads=4, include_redox_token=True,
        )
        assert encoder.element_embedding.weight[REDOX_TOKEN_ID].abs().sum() > 0

    def test_redox_forward_pools_23_tokens(self) -> None:
        encoder = Encoder(
            d_model=64, n_layers=1, n_heads=4, include_redox_token=True,
        )
        src, frac = _make_batch(batch_size=4)
        redox_src = torch.full((4, 1), REDOX_TOKEN_ID, dtype=src.dtype)
        redox_frac = torch.full((4, 1), 0.2, dtype=frac.dtype)
        out = encoder(
            src=torch.cat([src, redox_src], dim=1),
            frac=torch.cat([frac, redox_frac], dim=1),
        )
        assert out.shape == (4, 64)
        assert torch.isfinite(out).all()

    def test_width_mismatch_fails_loudly(self) -> None:
        """A 22-token state dict must not load into a redox-width model:
        strict load_state_dict is the guard against silently scoring a
        checkpoint through the wrong-width table."""
        old = Encoder(d_model=64, n_layers=1, n_heads=4)
        new = Encoder(
            d_model=64, n_layers=1, n_heads=4, include_redox_token=True,
        )
        with pytest.raises(RuntimeError):
            new.load_state_dict(old.state_dict())


def test_odd_d_model_raises() -> None:
    """The two FractionalEncoder halves concatenate to d_model, so an
    odd value must fail loudly at construction."""
    with pytest.raises(ValueError, match="even"):
        Encoder(d_model=33, n_layers=1, n_heads=1)


def test_shape_dims_required() -> None:
    """d_model / n_layers / n_heads carry no defaults: the config is the
    single source of truth for the architecture.  Omitting one must fail
    loudly rather than silently build a phantom model from a stale default.
    """
    with pytest.raises(TypeError):
        Encoder(d_model=64, n_heads=4)  # missing n_layers


# ---------------------------------------------------------------------
# Encoder shape + padding-mask semantics
# ---------------------------------------------------------------------

class TestEncoderForward:
    """The encoder's job: produce a pooled latent shaped [B, d_model]."""

    def test_output_shape(self, encoder: Encoder) -> None:
        src, frac = _make_batch(batch_size=4)
        h = encoder(src, frac)
        assert h.shape == (4, encoder.d_model)

    def test_all_padding_does_not_nan(self, encoder: Encoder) -> None:
        """An all-zero (all padding) row should not divide-by-zero."""
        src = torch.zeros(2, N_TOKENS, dtype=torch.long)
        frac = torch.zeros(2, N_TOKENS, dtype=torch.float32)
        h = encoder(src, frac)
        assert torch.isfinite(h).all(), "all-padding input produced non-finite output"

    def test_padding_invariance(self, encoder: Encoder) -> None:
        """Reordering the populated tokens vs padding slots within the
        sequence must NOT change the pooled latent (masked-mean pool is
        order-invariant over present-vs-absent assignment of slots).

        This is a regression catch for any future change that, say,
        weighted positions or used positional encodings.
        """
        src_a, frac_a = _make_batch(batch_size=1)
        # Shuffle the 9 populated majors into a different set of slot
        # indices; pad the rest.
        perm = torch.randperm(N_TOKENS)
        src_b = src_a[:, perm]
        frac_b = frac_a[:, perm]
        h_a = encoder(src_a, frac_a)
        h_b = encoder(src_b, frac_b)
        # The masked-mean pool is order-invariant, but the transformer
        # mixes tokens, so the per-token outputs differ; the pooled
        # latent should be close-but-not-identical.  Loose tolerance.
        torch.testing.assert_close(h_a, h_b, rtol=1e-3, atol=1e-3)


# ---------------------------------------------------------------------
# The load-bearing test: masked-token gradient flow
# ---------------------------------------------------------------------

class TestSSLGradientFlow:
    """The v2 bug regression test.

    The v2 SSL pretrain ran for 8 epochs while encoder weights stayed at
    their random-init values, byte-for-byte.  The cause was a padding
    convention where masked positions were treated as padding by the
    encoder, so the SSL reconstruction loss had zero gradient through
    the encoder.

    This test verifies that with the v3 masking convention and the
    pooled-latent decoder, encoder parameters receive non-zero
    gradient from the SSL loss.
    """

    def test_encoder_receives_gradient(
        self, encoder: Encoder, head: PretrainHead,
    ) -> None:
        torch.manual_seed(42)
        src, frac = _make_batch(batch_size=8, seed=42)

        masked = mask_for_ssl(src, frac, mask_prob=0.25)

        h = encoder(masked.src_corrupted, masked.frac_corrupted)
        predicted = head(h)
        m = masked.mask_positions
        loss = ((predicted - masked.target_frac) ** 2 * m).sum() / m.sum().clamp(min=1)

        assert loss.requires_grad, "loss must require grad to test gradient flow"
        loss.backward()

        # Every encoder parameter should have a non-zero gradient.
        zero_grad_params: list[str] = []
        for name, param in encoder.named_parameters():
            if param.grad is None:
                zero_grad_params.append(f"{name} (grad=None)")
                continue
            if param.grad.abs().sum().item() == 0.0:
                zero_grad_params.append(f"{name} (grad.norm=0)")
        assert not zero_grad_params, (
            "SSL loss must flow gradient to every encoder parameter; "
            f"these had no gradient: {zero_grad_params}.  This is the "
            f"v2 padding-mask bug shape -- check Encoder.forward's "
            f"padding mask derivation."
        )

    def test_loss_is_finite(
        self, encoder: Encoder, head: PretrainHead,
    ) -> None:
        torch.manual_seed(43)
        src, frac = _make_batch(batch_size=8, seed=43)
        masked = mask_for_ssl(src, frac, mask_prob=0.25)
        h = encoder(masked.src_corrupted, masked.frac_corrupted)
        predicted = head(h)
        m = masked.mask_positions
        loss = ((predicted - masked.target_frac) ** 2 * m).sum() / m.sum().clamp(min=1)
        assert torch.isfinite(loss).item(), f"loss not finite: {loss.item()}"


# ---------------------------------------------------------------------
# Masking convention smoke checks
# ---------------------------------------------------------------------

class TestMaskingConvention:
    """v3 masking convention C: masked src is MASK_TOKEN_ID; masked frac is 0."""

    def test_masked_src_is_mask_token_id(self) -> None:
        torch.manual_seed(0)
        src, frac = _make_batch(batch_size=16, seed=1)
        masked = mask_for_ssl(src, frac, mask_prob=0.5)
        m = masked.mask_positions
        assert (masked.src_corrupted[m] == MASK_TOKEN_ID).all(), (
            f"masked positions must have src=MASK_TOKEN_ID ({MASK_TOKEN_ID}); "
            f"convention C BERT-style mask token, NOT src=0 (padding)"
        )
        assert (masked.frac_corrupted[m] == 0).all(), (
            "masked positions must have frac=0 (the value the SSL loss predicts)"
        )

    def test_mask_token_distinguishable_from_padding(self) -> None:
        """MASK_TOKEN_ID positions must NOT collide with padding positions."""
        torch.manual_seed(0)
        src, frac = _make_batch(batch_size=8, seed=99)
        masked = mask_for_ssl(src, frac, mask_prob=0.5)
        m = masked.mask_positions
        # Where mask_positions is True, src is MASK_TOKEN_ID, not 0.
        assert (masked.src_corrupted[m] != 0).all(), (
            "masked positions must NOT have src=0 -- that would make them "
            "indistinguishable from padding (the v3 convention A failure mode)"
        )
        # Where mask_positions is False, src is either 0 (padding) or in
        # [1, 118] (populated) -- it is never MASK_TOKEN_ID outside the
        # masked positions.
        not_m = ~m
        assert (masked.src_corrupted[not_m] != MASK_TOKEN_ID).all(), (
            "MASK_TOKEN_ID must appear ONLY at masked positions"
        )

    def test_masking_preserves_target(self) -> None:
        """The target_frac should equal the *original* frac, not the corrupted one."""
        torch.manual_seed(0)
        src, frac = _make_batch(batch_size=4, seed=2)
        masked = mask_for_ssl(src, frac, mask_prob=0.25)
        torch.testing.assert_close(masked.target_frac, frac)

    def test_only_populated_positions_get_masked(self) -> None:
        """An originally-absent token (src==0) must never appear in mask_positions."""
        src, frac = _make_batch(batch_size=4)
        masked = mask_for_ssl(src, frac, mask_prob=0.9)
        # masked.mask_positions must be a strict subset of (src > 0).
        assert (
            masked.mask_positions & (src == 0)
        ).sum().item() == 0, (
            "mask_for_ssl masked an absent token; only populated "
            "(src>0) tokens may be selected"
        )
