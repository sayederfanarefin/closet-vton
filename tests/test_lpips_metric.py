import sys
import types

import pytest
import torch

from evaluation.lpips_metric import LpipsScorer


class _FakeModel:
    def __init__(self) -> None:
        self.inputs = None

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, pred, gt):
        self.inputs = (pred, gt)
        return (pred - gt).abs().mean().reshape(1, 1, 1, 1)


def test_lpips_scorer_normalizes_images_to_minus_one_one(monkeypatch) -> None:
    model = _FakeModel()
    fake_module = types.SimpleNamespace(LPIPS=lambda net: model)
    monkeypatch.setitem(sys.modules, "lpips", fake_module)

    scorer = LpipsScorer(net="alex", device="cpu")
    value = scorer.score(torch.ones(3, 16, 16), torch.zeros(3, 16, 16))

    assert value == pytest.approx(2.0)
    pred_input, gt_input = model.inputs
    assert torch.all(pred_input == 1.0)
    assert torch.all(gt_input == -1.0)


def test_lpips_scorer_rejects_mismatched_shapes(monkeypatch) -> None:
    fake_module = types.SimpleNamespace(LPIPS=lambda net: _FakeModel())
    monkeypatch.setitem(sys.modules, "lpips", fake_module)
    scorer = LpipsScorer(device="cpu")

    with pytest.raises(ValueError, match="equal image shapes"):
        scorer.score(torch.zeros(3, 16, 16), torch.zeros(3, 8, 8))
