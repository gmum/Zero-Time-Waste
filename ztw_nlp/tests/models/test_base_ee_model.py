import pytest
import torch
from typing import List
from src.models.ee.base import would_early_exit


@pytest.mark.parametrize("layer_logits,threshold,expected",
                         [
                             ([torch.tensor([0.9, 0.1])], 1., True),
                             ([torch.tensor([0.9, 0.1])], 0.5, False),
                             ([torch.tensor([0.9, 0.1])], 0., False)

                         ])
def test_would_early_exit_entropy(layer_logits: List[torch.Tensor], threshold: float,
                                  expected: bool) -> None:
    would_exit = would_early_exit(layer_logits=layer_logits, ee_threshold=threshold,
                                  ee_criterion="entropy")

    assert would_exit == expected


@pytest.mark.parametrize("layer_logits,threshold,expected",
                         [
                             ([torch.tensor([0.9, 0.1])], 1., False),
                             ([torch.tensor([0.8, 0.05])], 0.5, True),
                             ([torch.tensor([1.2, 0.01])], 0.7, True)

                         ])
def test_would_early_exit_max_confidence(layer_logits: List[torch.Tensor], threshold: float,
                                         expected: bool) -> None:
    would_exit = would_early_exit(layer_logits=layer_logits, ee_threshold=threshold,
                                  ee_criterion="max_confidence")

    assert would_exit == expected


@pytest.mark.parametrize("layer_logits,threshold,expected",
                         [
                             ([torch.tensor([0.9, 0.1]), torch.tensor([0.9, 0.1])], 1, True),
                             ([torch.tensor([0.9, 0.1]), torch.tensor([0.9, 0.1])], 2, False),
                             ([torch.tensor([0.8, 0.05]), torch.tensor([0.8, 0.95]),
                               torch.tensor([0.8, 0.05])], 1, False),
                             ([torch.tensor([0.8, 0.05]), torch.tensor([0.8, 0.95]),
                               torch.tensor([0.8, 1.05])], 2, False),
                             ([torch.tensor([0.8, 1.05]), torch.tensor([0.8, 0.95]),
                               torch.tensor([0.8, 1.05])], 2, True)
                         ])
def test_would_early_exit_patience(layer_logits: List[torch.Tensor], threshold: int,
                                   expected: bool) -> None:
    would_exit = would_early_exit(layer_logits=layer_logits, ee_threshold=threshold,
                                  ee_criterion="patience")

    assert would_exit == expected
