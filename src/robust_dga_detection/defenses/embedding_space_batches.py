import enum
import random

import foolbox as fb
import numpy as np
import torch
import torchattacks
from foolbox import PyTorchModel

from robust_dga_detection.attacks.embedding_space import (
    ImageEmulationAdapter,
    BinaryAutoAttack,
)
from robust_dga_detection.models import BinaryDgaDetectorWithCharacterEmbedding


class EmbeddingSpaceAttackCatalogue(enum.Enum):
    """A collection of the names of all implemented embedding space attacks."""

    PGD_L2 = 0
    PGD_LInf = 1
    AT_L2 = 2
    AT_Linf = 3
    CW_L2 = 4


class EmbeddingSpaceBatchAttack:
    """A batch attack that incorporates all 5 baseline embedding space attack.

    Each batch is split in n slice where n is the number of enabled attacks. Each attack is used to fill one slice.
    The primary attack hyperparameters are sampled from a uniform distribution.
    """

    model: BinaryDgaDetectorWithCharacterEmbedding
    attacks: set[EmbeddingSpaceAttackCatalogue]
    attack_list: list[EmbeddingSpaceAttackCatalogue]

    def __init__(
        self,
        model: BinaryDgaDetectorWithCharacterEmbedding,
        attacks: set[EmbeddingSpaceAttackCatalogue],
    ):
        """Create an EmbeddingSpaceBatch attack against the provided model with the given attacks.

        :param model: the model to attack
        :param attacks: the set of attacks to use during the attack
        """
        self.model = model
        self.attacks = attacks
        self.attack_list = list(attacks)

    def attack(self, input_x: torch.Tensor, input_y: torch.Tensor) -> torch.Tensor:
        """Perform the attack."""
        # Sample Attack Hyperparameters
        eps_linf = np.random.uniform(0.01, 1.0)
        eps_l2 = np.random.uniform(0.01, np.sqrt(input_x.size(1) * input_x.size(2)))

        restrict_cw_confidence = random.choice([True, False])
        if restrict_cw_confidence:
            cw_confidence = np.random.uniform(0.01, 100)
        else:
            cw_confidence = 0

        # Construct attack targets
        image_model = ImageEmulationAdapter(self.model).eval()
        foolbox_model = PyTorchModel(
            image_model, bounds=(0, 1), device=input_x.device
        )

        indices = torch.arange(0, input_x.size(0))
        embedding_space_attack_splits = torch.tensor_split(
            indices, len(self.attack_list)
        )
        adversarial_samples = input_x.clone()

        # Perform chosen attacks
        for i, attack_name in enumerate(self.attack_list):
            attack_ind = embedding_space_attack_splits[i]
            if attack_name == EmbeddingSpaceAttackCatalogue.PGD_L2:
                pgd_l2 = torchattacks.PGDL2(
                    model=image_model, eps=eps_l2, steps=50, random_start=True
                )
                adversarial_samples[attack_ind] = image_model.apply_attack(
                    pgd_l2, adversarial_samples[attack_ind], input_y[attack_ind].long()
                )
            elif attack_name == EmbeddingSpaceAttackCatalogue.PGD_LInf:
                pgd_linf = torchattacks.PGD(
                    model=image_model, eps=eps_linf, steps=50, random_start=True
                )
                adversarial_samples[attack_ind] = image_model.apply_attack(
                    pgd_linf,
                    adversarial_samples[attack_ind],
                    input_y[attack_ind].long(),
                )
            elif attack_name == EmbeddingSpaceAttackCatalogue.AT_L2:
                at_l2 = BinaryAutoAttack(model=image_model, eps=eps_l2, norm="L2")
                adversarial_samples[attack_ind] = image_model.apply_attack(
                    at_l2, adversarial_samples[attack_ind], input_y[attack_ind].long()
                )
            elif attack_name == EmbeddingSpaceAttackCatalogue.AT_Linf:
                at_linf = BinaryAutoAttack(model=image_model, eps=eps_linf, norm="Linf")
                adversarial_samples[attack_ind] = image_model.apply_attack(
                    at_linf, adversarial_samples[attack_ind], input_y[attack_ind].long()
                )
            elif attack_name == EmbeddingSpaceAttackCatalogue.CW_L2:
                cw = fb.attacks.L2CarliniWagnerAttack(
                    steps=50, confidence=cw_confidence
                )

                def cw_attack_fun(inputs, labels):
                    criterion = fb.Misclassification(labels)
                    _, adv_examples, _ = cw(
                        foolbox_model,
                        inputs,
                        criterion,
                        epsilons=128,
                    )
                    return adv_examples

                adversarial_samples[attack_ind] = image_model.apply_attack(
                    cw_attack_fun,
                    adversarial_samples[attack_ind],
                    input_y[attack_ind].long(),
                )
            else:
                raise ValueError(f"Unknown attack {attack_name}")

        return adversarial_samples
