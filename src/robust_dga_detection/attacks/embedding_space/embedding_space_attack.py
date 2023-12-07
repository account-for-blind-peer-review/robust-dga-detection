import abc

import torch

from robust_dga_detection.models import BinaryDgaDetectorWithCharacterEmbedding


class EmbeddingSpaceAttack(abc.ABC):
    """An abstract base class for Embedding Space (Continuous) Attacks."""

    target_model_config: BinaryDgaDetectorWithCharacterEmbedding

    def __init__(self, target_model_config: BinaryDgaDetectorWithCharacterEmbedding):
        """Create an embedding-space attack.

        :param target_model_config the model to attack
        """
        self.target_model_config = target_model_config

    @abc.abstractmethod
    def forward(self, inputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Execute the adversarial attack against the provided inputs.

        :param inputs: The (embedded) inputs to attack. Expected input Shape: [BATCH_SIZE, SEQ_LEN, EMBEDDING_DIM].
        :param labels: The correct labels of the input. The attack will try to change that label.
        :return: The adversarial embedding vectors.
        """
        pass

    def __call__(self, inputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Execute the adversarial attack against the provided inputs.

        :param inputs: The (embedded) inputs to attack. Expected input Shape: [BATCH_SIZE, SEQ_LEN, EMBEDDING_DIM].
        :param labels: The correct labels of the input. The attack will try to change that label.
        :return: The adversarial embedding vectors.
        """
        return self.forward(inputs, labels)


class Identity(EmbeddingSpaceAttack):
    """An attack that does nothing."""

    def forward(self, inputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Overriden."""
        return inputs
