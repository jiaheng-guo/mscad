import pytest
import torch

from mscad.model import CrossScaleBridge, MSCADModel


def small_model(**kwargs):
    options = dict(patch_sizes=(4, 8), d_model=8, n_heads=2,
                   n_layers_per_scale=1, bridge_depth=2, dropout=0.0)
    options.update(kwargs)
    return MSCADModel(**options)


def test_forward_errors_and_gradients_reach_all_branches_and_bridges():
    torch.manual_seed(1)
    model = small_model()
    errors, scores = model(torch.randn(3, 16, 1))
    assert errors.shape == (3, 2)
    assert scores.shape == (3,)
    torch.testing.assert_close(scores, errors.mean(dim=1))
    assert torch.isfinite(errors).all() and (errors >= 0).all()
    scores.mean().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


def test_cross_scale_updates_use_one_unchanged_snapshot():
    torch.manual_seed(2)
    bridge = CrossScaleBridge(8, 2, dropout=0.0).eval()
    tokens = [torch.randn(2, n, 8) for n in (7, 3, 1)]
    before = [token.clone() for token in tokens]
    actual = bridge(tokens)
    for i, query in enumerate(before):
        context = torch.cat([x for j, x in enumerate(before) if i != j], dim=1)
        attention, _ = bridge.cross_attn(query, context, context, need_weights=False)
        residual = bridge.norm1(query + attention)
        expected = bridge.norm2(residual + bridge.ff(residual))
        torch.testing.assert_close(actual[i], expected)
        torch.testing.assert_close(tokens[i], before[i])
    # Reordering scales must only reorder outputs: no privileged scale or
    # accidental in-place sequential exchange may change the computation.
    permuted = bridge([tokens[2], tokens[0], tokens[1]])
    for expected, result in zip([actual[2], actual[0], actual[1]], permuted):
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-5)


def test_headline_architecture_has_two_bridges_and_preserves_embedding_details():
    model = MSCADModel()
    assert len(model.cross_scale_bridge) == 2
    assert sum(p.numel() for p in model.parameters()) == 6_756_692
    for branch in model.branches:
        assert branch.pos_enc.pe.shape == (1, 512, 256)
        assert isinstance(branch.patch_embed.norm, torch.nn.LayerNorm)


@pytest.mark.parametrize('shape', [(2, 16), (2, 16, 2), (0, 16, 1), (2, 2, 1)])
def test_model_rejects_invalid_input_shape(shape):
    with pytest.raises(ValueError):
        small_model()(torch.zeros(shape))


def test_model_rejects_position_overflow():
    with pytest.raises(ValueError, match='512'):
        small_model(patch_sizes=(2,))(torch.zeros(1, 514, 1))


def test_one_scale_uses_uniform_identity_fusion():
    model = small_model(patch_sizes=(4,))
    errors, scores = model(torch.zeros(2, 16, 1))
    assert len(model.cross_scale_bridge) == 0
    torch.testing.assert_close(scores, errors[:, 0])
