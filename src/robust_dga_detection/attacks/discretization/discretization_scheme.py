import abc

import torch


class DiscretizationScheme(abc.ABC):
    """An interface for a discretizer.

    A discretizer is a function that maps an arbitrary embedding-space vector to a valid domain encoding.
    """

    @abc.abstractmethod
    def __call__(
        self,
        original_encoded_domains: torch.Tensor,
        adversarial_embedding_vector: torch.Tensor,
    ) -> torch.Tensor:
        """Map an adversarial embedding vector to a valid encoded domain.

        :param original_encoded_domains: A tensor of shape (N, SEQ_LEN) containing the encoded
            original domains that have been perturbed to obtain the adversarial_embedding_vector
        :param adversarial_embedding_vector: A tensor of shape (N, SEQ_LEN, EMBEDDING_DIM) containing
            adversarial embedding vectors for N domains.
        :return: A tensor of shape (N, SEQ_LEN) containing the encoded discrete domains.
        """
        pass
