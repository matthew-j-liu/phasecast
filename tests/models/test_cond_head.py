import torch

from cnmineral.data.phase_vocab import N_PHASES
from cnmineral.models.cond_head import N_AXES, CondHead


def _forward(head, batch=4, d_model=32):
    h = torch.randn(batch, d_model)
    presence = (torch.rand(batch, N_PHASES) > 0.5).float()
    return head(h, presence), h, presence


def test_output_shapes():
    head = CondHead(d_model=32, n_components=3)
    (log_w, mu, sigma), _, _ = _forward(head)
    assert log_w.shape == mu.shape == sigma.shape == (4, N_AXES, 3)


def test_log_weights_normalize():
    head = CondHead(d_model=32, n_components=4)
    (log_w, _, _), _, _ = _forward(head)
    assert torch.allclose(
        torch.logsumexp(log_w, dim=-1), torch.zeros(4, N_AXES), atol=1e-5
    )


def test_sigma_respects_floor():
    head = CondHead(d_model=32, sigma_eps=1e-3)
    (_, _, sigma), _, _ = _forward(head)
    assert (sigma >= 1e-3).all()


def test_presence_changes_output():
    torch.manual_seed(0)
    head = CondHead(d_model=32)
    h = torch.randn(1, 32)
    base = torch.zeros(1, N_PHASES)
    flipped = base.clone()
    flipped[0, 0] = 1.0
    _, mu_a, _ = head(h, base)
    _, mu_b, _ = head(h, flipped)
    assert not torch.allclose(mu_a, mu_b)


def test_init_output_bias_seeds_mixture_at_target_scale():
    head = CondHead(d_model=8, n_components=3)
    mean = torch.tensor([900.0, 6.0, 4.0])
    std = torch.tensor([150.0, 3.0, 2.0])
    head.init_output_bias(mean, std)
    _, mu, sigma = head(torch.zeros(1, 8), torch.zeros(1, N_PHASES))
    assert torch.allclose(mu.mean(-1)[0], mean, rtol=0.2)
    assert torch.allclose(sigma[0, :, 1], std, rtol=0.2)


def test_gradients_flow_to_all_parameters():
    head = CondHead(d_model=16)
    (log_w, mu, sigma), _, _ = _forward(head, d_model=16)
    (log_w.sum() + mu.sum() + sigma.sum()).backward()
    assert all(p.grad is not None for p in head.parameters())
