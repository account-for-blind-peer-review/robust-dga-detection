from typing import Any

import foolbox
import torch
from foolbox import Attack, PyTorchModel

from robust_dga_detection.attacks.embedding_space.embedding_space_attack import (
    EmbeddingSpaceAttack,
)
from robust_dga_detection.attacks.embedding_space.image_emulation_adapter import (
    ImageEmulationAdapter,
)
from robust_dga_detection.models import BinaryDgaDetectorWithCharacterEmbedding


class FoolboxAdapter(EmbeddingSpaceAttack):
    """Adapts a Foolbox attack to an EmbeddingSpaceAttack."""

    foolbox_attack: Attack
    model_for_attack: ImageEmulationAdapter
    foolbox_model: PyTorchModel
    attack_kwargs: Any

    def __init__(
        self,
        target_model_config: BinaryDgaDetectorWithCharacterEmbedding,
        foolbox_attack: Attack,
        device: Any,
        **atack_kwargs: Any,
    ):
        """Create a FoolBox adapter.

        :param target_model_config the model to attack
        :param foolbox_attack the attack to apply
        :param device the pytorch device to use during the attack
        """
        super().__init__(target_model_config)

        self.foolbox_attack = foolbox_attack
        self.model_for_attack = ImageEmulationAdapter(target_model_config.model)
        self.foolbox_model = PyTorchModel(
            self.model_for_attack,
            bounds=(0, 1),
            device=device,
        )
        self.attack_kwargs = atack_kwargs

    def __attck_fun(self, inputs: torch.Tensor, labels: torch.Tensor):
        criterion = foolbox.Misclassification(labels)
        _, adv_examples, _ = self.foolbox_attack(
            self.foolbox_model,
            inputs,
            criterion,
            **self.attack_kwargs,
        )
        return adv_examples

    def forward(self, inputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Overriden."""
        self.model_for_attack.eval()
        return self.model_for_attack.apply_attack(self.__attck_fun, inputs, labels)
