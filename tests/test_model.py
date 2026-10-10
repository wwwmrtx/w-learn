import pytest
import torch
from torch import nn

from mineformer import MineFormerV1


@pytest.fixture(autouse=True)
def limit_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def small_model(**kwargs):
    return MineFormerV1(
        channels=(16, 32, 48, 64), depths=(1, 1, 1, 1), heads=(4, 4),
        stem_channels=16, decoder_channels=16, **kwargs,
    )


@pytest.mark.parametrize("size", [(64, 96), (65, 97), (32, 32)])
def test_output_and_feature_shapes(size):
    model = small_model().eval()
    x = torch.randn(1, 3, *size)
    with torch.no_grad():
        features = model.forward_features(x)
        logits = model(x)
    assert logits.shape == (1, 1, *size)
    assert torch.isfinite(logits).all()
    for feature, channels, scale in zip(features, (16, 32, 48, 64), (4, 8, 16, 32)):
        assert feature.shape == (1, channels, *((value + scale - 1) // scale for value in size))


def test_loss_backpropagates_through_all_components():
    torch.manual_seed(42)
    model = small_model(in_channels=1)
    logits = model(torch.randn(2, 1, 64, 64))
    target = torch.randint(0, 2, logits.shape).float()
    loss = nn.BCEWithLogitsLoss()(logits, target)
    loss.backward()
    assert torch.isfinite(loss)
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


def test_requested_architecture():
    model = small_model()
    for stage in (model.stage1, model.stage2):
        assert any(isinstance(module, nn.Conv2d) for module in stage.modules())
        assert not any(isinstance(module, nn.MultiheadAttention) for module in stage.modules())
    for stage in (model.stage3, model.stage4):
        assert any(isinstance(module, nn.MultiheadAttention) for module in stage.modules())
    assert len(model.decoder.projections) == 4
    assert model.segmentation_head.out_channels == 1
    assert not any("boundary" in name for name, _ in model.named_modules())


def test_default_model():
    model = MineFormerV1().eval()
    with torch.no_grad():
        assert model(torch.randn(1, 3, 64, 64)).shape == (1, 1, 64, 64)


@pytest.mark.parametrize("kwargs", [
    {"channels": (16, 32)}, {"depths": (1, 1, 1, 0)},
    {"heads": (3, 3)}, {"decoder_channels": 0},
    {"mlp_ratio": 0}, {"dropout": 1},
])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        MineFormerV1(**kwargs)


@pytest.mark.parametrize("shape", [(1, 3, 16, 64), (1, 1, 64, 64), (3, 64, 64)])
def test_invalid_input(shape):
    with pytest.raises(ValueError):
        small_model()(torch.randn(*shape))
