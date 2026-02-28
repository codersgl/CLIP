import pytest
import torch
import torch.nn.functional as F
from clip import CLIPLoss


@pytest.fixture
def features():
    batch = 4
    dim = 8
    image = F.normalize(torch.randn(batch, dim))
    text = F.normalize(torch.randn(batch, dim))
    return image, text


def test_loss_output_shape(features):
    loss_fn = CLIPLoss()
    loss = loss_fn(*features)
    assert loss.ndim == 0  # 0 维张量


def test_loss_symmetric(features):
    loss_fn = CLIPLoss()
    loss1 = loss_fn(*features)
    loss2 = loss_fn(features[1], features[0])
    assert torch.allclose(loss1, loss2, atol=1e-6)


def test_gradient_flow(features):
    image, text = features
    image.requires_grad_(True)
    text.requires_grad_(True)
    loss_fn = CLIPLoss()
    loss = loss_fn(image, text)
    loss.backward()
    assert image.grad is not None
    assert text.grad is not None
    assert not torch.isnan(image.grad).any()
    assert not torch.isnan(text.grad).any()


def test_temperature_scale():
    loss_low = CLIPLoss(temperature=0.07)
    loss_high = CLIPLoss(temperature=0.5)
    scale_low = loss_low.logit_scale.exp().item()
    scale_high = loss_high.logit_scale.exp().item()
    assert scale_low > scale_high
    assert abs(scale_low - 1 / 0.07) < 1e-5
    assert abs(scale_high - 1 / 0.5) < 1e-5


def test_batch_one():
    image = F.normalize(torch.randn(1, 8))
    text = F.normalize(torch.randn(1, 8))
    loss_fn = CLIPLoss()
    loss = loss_fn(image, text)
    assert loss.ndim == 0
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)
