import enum
import random

import foolbox as fb
import numpy as np
import torch
import torchattacks
from foolbox import PyTorchModel

from robust_dga_detection.attacks.discretization import (
    E2lDDiscretizerWithLengthCutoff,
    E2lDDiscretizerWithLengthBruteForce,
    RoundingNorm,
)
from robust_dga_detection.attacks.nlp import HotFlip, MaskDGA, OneHotModelInputWrapper
from robust_dga_detection.utils import domains

from robust_dga_detection.attacks.embedding_space import (
    BinaryAutoAttack,
    ImageEmulationAdapter,
)
from robust_dga_detection.models import BinaryDgaDetectorWithCharacterEmbedding
from robust_dga_detection.defenses.embedding_space_batches import (
    EmbeddingSpaceAttackCatalogue,
)


class EmbeddingSpaceDiscretizationCatalogue(enum.Enum):
    """A collection of the names of all implemented embedding space discretization approaches."""

    BF_L2 = 0
    BF_LInf = 1
    BF_Cos = 2
    CO_L2 = 3
    CO_Linf = 4
    CO_Cos = 5


class NLPAttackCatalogue(enum.Enum):
    """A collection of the names of all implemented nlp attacks."""

    HotFlip = 0
    MaskDGA = 1


class DiscreteDomainbatchAttack:
    """A batch attack that incorporates both embedding space + disc attacks as well as native nlp attacks.

    Each batch is split in n slice where n is the number of enabled attacks. Each attack is used to fill one slice.
    The primary attack hyperparameters are sampled from a uniform distribution.
    """

    model: BinaryDgaDetectorWithCharacterEmbedding
    continuous_attacks: list[EmbeddingSpaceAttackCatalogue]
    discretization_schemes: list[EmbeddingSpaceDiscretizationCatalogue]
    discrete_attacks: list[NLPAttackCatalogue]

    def __init__(
        self,
        model: BinaryDgaDetectorWithCharacterEmbedding,
        continuous_attacks: set[EmbeddingSpaceAttackCatalogue],
        discretization_schemes: set[EmbeddingSpaceDiscretizationCatalogue],
        discrete_attacks: set[NLPAttackCatalogue],
    ):
        """Create a DiscreteDomainBatch attack.

        :param model: the model to attack
        :param continuous_attacks: the set of embedding space attacks to use
        :param discrete_attacks: the set of discrete attacks to use
        :param discretization_schemes: the set of discretization schemes to use
        """
        self.model = model
        self.continuous_attacks = list(continuous_attacks)
        self.discretization_schemes = list(discretization_schemes)
        self.discrete_attacks = list(discrete_attacks)

    def attack(self, input_x: torch.Tensor, input_y: torch.Tensor) -> torch.Tensor:
        """Perform the attack."""
        # Sample Attack Hyperparameters
        eps_linf = np.random.uniform(0.01, 1.0)

        restrict_cw_confidence = random.choice([True, False])
        if restrict_cw_confidence:
            cw_confidence = np.random.uniform(0.01, 100)
        else:
            cw_confidence = 0

        hotflip_n_flips = np.random.randint(1, 10)

        # Construct attack targets
        image_model = ImageEmulationAdapter(self.model).eval()
        foolbox_model = PyTorchModel(
            image_model, bounds=(0, 1), device=input_x.device
        )
        onehot_model = OneHotModelInputWrapper(self.model)

        indices = torch.arange(0, input_x.size(0))
        attack_indices = torch.tensor_split(
            indices, len(self.continuous_attacks) + len(self.discrete_attacks)
        )
        adversarial_samples = self.model.embedding(input_x)

        eps_l2 = np.random.uniform(
            0.01, np.sqrt(adversarial_samples.size(1) * adversarial_samples.size(2))
        )

        # Perform chosen discrete attacks
        for i, attack_name in enumerate(self.continuous_attacks):
            attack_ind = attack_indices[i]
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

        # Perform Discretization on chosen attacks
        disc_splits_pre = [
            torch.tensor_split(it, len(self.discretization_schemes))
            for it in attack_indices[: len(self.continuous_attacks)]
        ]
        disc_splits = [
            torch.cat([splits_pre[j] for splits_pre in disc_splits_pre])
            for j in range(len(self.discretization_schemes))
        ]

        discretized_domains = torch.zeros_like(input_x)
        disc_map = {
            EmbeddingSpaceDiscretizationCatalogue.BF_L2: E2lDDiscretizerWithLengthBruteForce(
                self.model, norm=RoundingNorm.L_2, minimum_output_length=7
            ),
            EmbeddingSpaceDiscretizationCatalogue.BF_LInf: E2lDDiscretizerWithLengthBruteForce(
                self.model, norm=RoundingNorm.L_INF, minimum_output_length=7
            ),
            EmbeddingSpaceDiscretizationCatalogue.BF_Cos: E2lDDiscretizerWithLengthBruteForce(
                self.model, norm=RoundingNorm.COS, minimum_output_length=7
            ),
            EmbeddingSpaceDiscretizationCatalogue.CO_L2: E2lDDiscretizerWithLengthCutoff(
                self.model, norm=RoundingNorm.L_2, minimum_output_length=7
            ),
            EmbeddingSpaceDiscretizationCatalogue.CO_Linf: E2lDDiscretizerWithLengthCutoff(
                self.model, norm=RoundingNorm.L_INF, minimum_output_length=7
            ),
            EmbeddingSpaceDiscretizationCatalogue.CO_Cos: E2lDDiscretizerWithLengthCutoff(
                self.model, norm=RoundingNorm.COS, minimum_output_length=7
            ),
        }

        for i, disc in enumerate(self.discretization_schemes):
            disc_ind = disc_splits[i]
            discretized_domains[disc_ind] = disc_map[disc](
                input_x[disc_ind], adversarial_samples[disc_ind]
            ).int()

        # Employ HotFlip and MaskDGA
        for i, attack in enumerate(self.discrete_attacks):
            if attack == NLPAttackCatalogue.HotFlip:
                hotflip_ind = attack_indices[len(self.continuous_attacks) + i]
                hotflip_attack = HotFlip(onehot_model, 10, hotflip_n_flips)
                discretized_domains[hotflip_ind] = hotflip_attack.forward(
                    None, input_x[hotflip_ind], input_y[hotflip_ind]
                )
            elif attack == NLPAttackCatalogue.MaskDGA:
                maskdga_ind = attack_indices[len(self.continuous_attacks) + i]
                maskdga_attack = MaskDGA(onehot_model)
                discretized_domains[maskdga_ind] = maskdga_attack.forward(
                    domains.decode_domains(input_x[maskdga_ind]),
                    input_x[maskdga_ind],
                    input_y[maskdga_ind],
                )

        self.model.train()
        return discretized_domains
