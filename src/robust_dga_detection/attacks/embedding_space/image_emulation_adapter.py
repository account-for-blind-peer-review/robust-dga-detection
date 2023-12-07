from collections.abc import Callable

import torch
import torch.nn as nn

from robust_dga_detection.models import BinaryDgaDetectorWithCharacterEmbedding


class ImageEmulationAdapter(nn.Module):
    """A translation layer that makes a DGA-detection network "appear" as an Image-Classification network.

    Many adversarial attacks are implemented such that they expect and Input of Shape
        [BATCH_SIZE, CHANNEL, WIDTH, HEIGHT] with all "pixels" having a value in [0, 1].

    To use these attacks with DGA classifiers, we implemented this Adapter.
    """

    model: BinaryDgaDetectorWithCharacterEmbedding
    embedding_max: float
    embedding_min: float

    def __init__(self, model: BinaryDgaDetectorWithCharacterEmbedding):
        """Create an Image-Emulation adapter.

        :param model: the model to adapt
        """
        super().__init__()
        self.model = model
        self.embedding_max = model.embedding.weight.max().item()
        self.embedding_min = model.embedding.weight.min().item()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward an "emulated" image batch through the network.

        Input Shape: [BATCH, 1, SEQ_LEN, EMBEDDING_DIM]
        Output Shape: [BATCH, 2]
        """
        assert input.ndim == 4 and input.shape[1] == 1
        model_output = self.model.net(torch.squeeze(self.__unscale(input), dim=1))
        result = torch.cat((-model_output, model_output), dim=1)
        return result

    def apply_attack(
        self,
        attack: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        embedded_domains: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Apply a generic adversarial attack to the network.

        :param attack: The attack gets inputs (EMULATED_IMAGE_INPUTS, LABELS) and is expected to return
            adversarial examples.
        :param embedded_domains: The original embedded domain vectors.
        :param labels: The actual labels of the inputs.
        :return: Adversarial embedded domain vectors.
        """
        self.eval()
        attack_input = torch.unsqueeze(self.__scale(embedded_domains), dim=1)
        scaled_attack_results = attack(attack_input.detach(), labels.detach())
        attack_results = torch.squeeze(self.__unscale(scaled_attack_results), dim=1)
        return attack_results

    def __scale(self, input: torch.Tensor) -> torch.Tensor:
        return (input - self.embedding_min) / (self.embedding_max - self.embedding_min)

    def __unscale(self, input: torch.Tensor) -> torch.Tensor:
        return input * (self.embedding_max - self.embedding_min) + self.embedding_min
