import random

import numpy as np
import pytest
import torch

from clip.utils import set_seed


def test_set_seed_reproducibility():
    """Test that set_seed ensures reproducibility across random, numpy, and torch."""
    seed = 42

    # Run 1
    set_seed(seed)
    py_rand_1 = random.random()
    np_rand_1 = np.random.rand()
    torch_rand_1 = torch.rand(1).item()

    # Run 2
    set_seed(seed)
    py_rand_2 = random.random()
    np_rand_2 = np.random.rand()
    torch_rand_2 = torch.rand(1).item()

    assert py_rand_1 == py_rand_2, "Python random seed not set correctly"
    assert np_rand_1 == np_rand_2, "Numpy random seed not set correctly"
    assert torch_rand_1 == torch_rand_2, "Torch random seed not set correctly"


def test_set_seed_difference():
    """Test that different seeds produce different results."""
    # Run 1
    set_seed(42)
    val_1 = torch.rand(1).item()

    # Run 2
    set_seed(123)
    val_2 = torch.rand(1).item()

    assert val_1 != val_2, "Different seeds should produce different results"


def test_set_seed_cuda_calls():
    """Test that cuda seed functions are called (even if cuda not available, the function calls shouldn't crash)."""
    # The utils.py calls torch.cuda.manual_seed and torch.cuda.manual_seed_all directly.
    # We want to ensure these don't raise errors even on CPU environments (torch handles this gracefully usually,
    # but let's verify execution flow).
    try:
        set_seed(42)
    except Exception as e:
        pytest.fail(f"set_seed raised an exception: {e}")
