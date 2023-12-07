from collections.abc import Callable

import torch
import torchattacks.attack

from robust_dga_detection.attacks.embedding_space.embedding_space_attack import (
    EmbeddingSpaceAttack,
)
from robust_dga_detection.attacks.embedding_space.image_emulation_adapter import (
    ImageEmulationAdapter,
)
from robust_dga_detection.models import BinaryDgaDetectorWithCharacterEmbedding

TorchattackConstructor = Callable[[ImageEmulationAdapter], torchattacks.attack.Attack]


class TorchAttackAdapter(EmbeddingSpaceAttack):
    """Adapts a TorchAttack attack to an EmbeddingSpaceAttack."""

    torchattack_constructor: TorchattackConstructor
    model_for_attack: ImageEmulationAdapter
    torchattack: torchattacks.attack.Attack

    def __init__(
        self,
        target_model_config: BinaryDgaDetectorWithCharacterEmbedding,
        torchattack_constructor: TorchattackConstructor,
    ):
        """Create a TorchAttack adapter.

        :param target_model_config the model to adapt
        :param torchattack_constructor a function that constructs a TorchAttack attack from a translated model
        """
        super().__init__(target_model_config)

        self.torchattack_constructor = torchattack_constructor
        self.model_for_attack = ImageEmulationAdapter(target_model_config.model)

        self.torchattack = self.torchattack_constructor(self.model_for_attack)

    def forward(self, inputs: torch.Tensor, labels: torch.Tensor):
        """Overriden."""
        self.model_for_attack.eval()
        return self.model_for_attack.apply_attack(self.torchattack, inputs, labels)
