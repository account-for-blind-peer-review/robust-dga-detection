import torch

from robust_dga_detection.attacks.discretization.controllable_e2ld_embedding_rounder import (
    ControllableE2LDEmbeddingRounder,
)
from robust_dga_detection.attacks.discretization.discretization_scheme import (
    DiscretizationScheme,
)
from robust_dga_detection.attacks.discretization.embedding_subset import (
    RoundingNorm,
    EmbeddingSubset,
)
from robust_dga_detection.models.binary_dga_detector_with_character_embedding import (
    BinaryDgaDetectorWithCharacterEmbedding,
)
from robust_dga_detection.utils import domains


class E2lDDiscretizerWithLengthCutoff(DiscretizationScheme):
    """A Discretization algorithm that rounds each character to the nearest valid embedding vector.

    After reaching the first padding character, all further entries are discarded.
    """

    minimum_output_length: int
    model: BinaryDgaDetectorWithCharacterEmbedding
    norm: RoundingNorm
    embedding_subset: EmbeddingSubset
    controllable_rounder: ControllableE2LDEmbeddingRounder

    def __init__(
        self,
        model: BinaryDgaDetectorWithCharacterEmbedding,
        norm: RoundingNorm,
        minimum_output_length: int = 7,
    ):
        """Create a Length Cutoff Discretization instance.

        :param model: the model to attack
        :param norm: the norm for measuring closeness between embeddings
        :param minimum_output_length: the minimum length of the generated e2LDs
        """
        self.model = model
        self.norm = norm
        self.minimum_output_length = minimum_output_length

        self.embedding_subset = EmbeddingSubset.get_embedding_for_characters(
            model.embedding.weight, set(domains.char2ix.keys()) - {"_", "."}
        )

        self.controllable_rounder = ControllableE2LDEmbeddingRounder(model)

    def __call__(self, _: torch.Tensor, embedding_vector: torch.Tensor) -> torch.Tensor:
        """Overriden."""
        seq_len = embedding_vector.shape[1]
        lp_targets = self.embedding_subset.round_embedded_vector_to_encoded_vector(
            embedding_vector, self.norm
        )

        # Calculate position of first padding character
        length_mask = (lp_targets == 0) * (torch.arange(seq_len, 0, -1) - 1).to(
            embedding_vector.device
        )
        lengths, _ = torch.min(
            torch.where(length_mask > 0, length_mask, seq_len), dim=1
        )

        lengths = torch.clamp(lengths, min=self.minimum_output_length)

        # Calculate best output according to lengths
        output = self.controllable_rounder.get_domains_at_lengths(
            embedding_vector, lengths, self.norm
        )

        return output
