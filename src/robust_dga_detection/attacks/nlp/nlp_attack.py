import abc

import torch
import torch.nn as nn


class NLPAttack(abc.ABC):
    """An abstract base class for discrete Character-Space Attacks."""

    target_model: nn.Module

    def __init__(self, target_model: nn.Module):
        """Create an NLPAttack instance.

        :param target_model: the model to attack
        """
        self.target_model = target_model

    @abc.abstractmethod
    def forward(
        self, domains: list[str], encoded_domains: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Execute the adversarial attack against the provided inputs.

        :param domains: The domains (strings) to attack.
        :param encoded_domains: The (encoded) domains. Expected input Shape: [BATCH_SIZE, SEQ_LEN]
        :param labels: The correct labels of the input. The attack will try to change that label.
        :return: The adversarial encoded domains.
        """
        pass

    def __call__(
        self, domains: list[str], encoded_domains: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Execute the adversarial attack against the provided inputs.

        :param domains: The domains (strings) to attack.
        :param encoded_domains: The (encoded) domains. Expected input Shape: [BATCH_SIZE, SEQ_LEN]
        :param labels: The correct labels of the input. The attack will try to change that label.
        :return: The adversarial encoded domains.
        """
        return self.forward(domains, encoded_domains, labels)


class Identity(NLPAttack):
    """An attack that does nothing."""

    def forward(
        self, domains: list[str], encoded_domains: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Overriden."""
        return encoded_domains
