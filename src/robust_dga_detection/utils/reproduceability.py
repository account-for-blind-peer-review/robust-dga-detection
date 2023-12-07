import random
import torch
import torch.utils.data
import numpy as np
import os


def setup_deterministic_environment() -> None:
    """Create a deterministic environment.

    This function sets up an environment that's as deterministic as possible to ensure reproducibility of results
    Based on https://pytorch.org/docs/stable/notes/randomness.html
    """
    torch.manual_seed(406069)
    random.seed(406069)
    np.random.seed(406069)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    assert str(os.getenv("CUBLAS_WORKSPACE_CONFIG", "NONE")) == ":4096:8", (
        "Please set the environment variable 'CUBLAS_WORKSPACE_CONFIG' to ':4096:8' to enable a more deterministic "
        "CUDA environment"
    )

    assert str(os.getenv("PYTHONHASHSEED", "NONE")) == "0", (
        "Please set the environment variable 'PYTHONHASHSEED' to '0' to deactivate the randomization of the hash() "
        "function."
    )

    assert (
        hash("abc") == -4594863902769663758
    ), "Python hash of 'abc' does not match the expected deterministic value"
