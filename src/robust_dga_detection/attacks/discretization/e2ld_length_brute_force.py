import torch

from robust_dga_detection.attacks.discretization.controllable_e2ld_embedding_rounder import (
    ControllableE2LDEmbeddingRounder,
)
from robust_dga_detection.attacks.discretization.discretization_scheme import (
    DiscretizationScheme,
)
from robust_dga_detection.attacks.discretization.embedding_subset import RoundingNorm
from robust_dga_detection.models.binary_dga_detector_with_character_embedding import (
    BinaryDgaDetectorWithCharacterEmbedding,
)


class E2lDDiscretizerWithLengthBruteForce(DiscretizationScheme):
    """A Discretization algorithm that rounds each character to the nearest valid embedding vector.

    The optimal length of the output is determined by brute-force.
    """

    minimum_output_length: int
    model: BinaryDgaDetectorWithCharacterEmbedding
    norm: RoundingNorm

    controllable_rounder: ControllableE2LDEmbeddingRounder

    def __init__(
        self,
        model: BinaryDgaDetectorWithCharacterEmbedding,
        norm: RoundingNorm,
        minimum_output_length: int = 7,
    ):
        """Create a Length Brute-Force Discretization instance.

        :param model: the model to attack
        :param norm: the norm for measuring closeness between embeddings
        :param minimum_output_length: the minimum length of the generated e2LDs
        """
        self.model = model
        self.minimum_output_length = minimum_output_length
        self.norm = norm
        self.controllable_rounder = ControllableE2LDEmbeddingRounder(model)

    def __call__(self, _: torch.Tensor, embedding_vector: torch.Tensor) -> torch.Tensor:
        """Overriden."""
        seq_len = embedding_vector.shape[1]

        lp_targets_at_different_lengths = torch.stack(
            [
                self.controllable_rounder.get_domains_at_length(
                    embedding_vector, len, self.norm
                )
                for len in range(self.minimum_output_length, seq_len)
            ]
        ).to(embedding_vector.device)

        results_at_lengths = torch.stack(
            [
                self.model(lp_targets_at_different_lengths[i])
                for i in range(lp_targets_at_different_lengths.shape[0])
            ]
        )

        optimal_lengths_idx = torch.argmin(results_at_lengths, dim=0)
        output = torch.zeros_like(lp_targets_at_different_lengths[0])

        for i in range(output.shape[0]):
            output[i, :] = lp_targets_at_different_lengths[optimal_lengths_idx[i], i, :]

        return output
