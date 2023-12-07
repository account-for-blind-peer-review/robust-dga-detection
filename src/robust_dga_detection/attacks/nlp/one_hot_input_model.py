import torch
import torch.nn as nn

from robust_dga_detection.models import BinaryDgaDetectorWithCharacterEmbedding


class OneHotModelInputWrapper(nn.Module):
    """A wrapper around a torch model that accepts both index-encoded vectors and one-hot encoded vectors."""

    def __init__(self, model: BinaryDgaDetectorWithCharacterEmbedding):
        """Create a OneHotModelInputWrapper.

        :param model: the model to wrap
        """
        super().__init__()
        self.model = model

    def forward(self, input_domains: torch.Tensor) -> torch.Tensor:
        """Overridden."""
        if input_domains.ndim == 3:
            one_hot_embedded_domains = torch.matmul(
                input_domains, self.model.embedding.weight
            )
            return self.model.net(one_hot_embedded_domains)
        else:
            return self.model(input_domains)
