import pytest
import torch

from cnmineral.data.phase_vocab import N_PHASES
from cnmineral.models.mode_head import ModeHead


def test_output_shapes():
    head = ModeHead(d_model=16)
    logits, alpha = head(torch.randn(4, 16))
    assert logits.shape == (4, N_PHASES)
    assert alpha.shape == (4, N_PHASES)


def test_alpha_strictly_positive():
    torch.manual_seed(0)
    head = ModeHead(d_model=16, alpha_eps=1e-6)
    _, alpha = head(torch.randn(64, 16) * 10)
    assert torch.all(alpha >= 1e-6)


def test_default_hidden_is_half_d_model():
    head = ModeHead(d_model=32)
    assert head.trunk[0].out_features == 16
    assert ModeHead(d_model=32, hidden=8).trunk[0].out_features == 8


def test_layers_stack_the_trunk():
    deep = ModeHead(d_model=32, layers=3)
    assert sum(isinstance(m, torch.nn.Linear) for m in deep.trunk) == 3
    logits, alpha = deep(torch.randn(4, 32))
    assert logits.shape == (4, N_PHASES) and torch.all(alpha > 0)
    shallow = sum(p.numel() for p in ModeHead(d_model=32).parameters())
    assert sum(p.numel() for p in deep.parameters()) > shallow
    with pytest.raises(ValueError):
        ModeHead(d_model=32, layers=0)


def test_gradients_flow_to_all_parameters():
    torch.manual_seed(0)
    head = ModeHead(d_model=16)
    logits, alpha = head(torch.randn(8, 16))
    (logits.sum() + alpha.sum()).backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters())
