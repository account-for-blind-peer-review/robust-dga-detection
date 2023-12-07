import torch

from robust_dga_detection.attacks.discretization import discretization_utils
from robust_dga_detection.attacks.discretization.embedding_subset import (
    EmbeddingSubset,
    RoundingNorm,
)
from robust_dga_detection.models import BinaryDgaDetectorWithCharacterEmbedding
from robust_dga_detection.utils import domains


class ControllableE2LDEmbeddingRounder:
    """A 'controllable' embedding rounder that ensures that returned domains are valid e2LDs."""

    model: BinaryDgaDetectorWithCharacterEmbedding

    embedding_with_hyphen: EmbeddingSubset
    embedding_only_unrestricted: EmbeddingSubset
    embedding_only_hyphen: EmbeddingSubset

    def __init__(self, model: BinaryDgaDetectorWithCharacterEmbedding):
        """Construct a controllable e2LD embedding rounder.

        :param model: the underlying model used for accessing the embedding.
        """
        self.model = model
        self.embedding_with_hyphen = EmbeddingSubset.get_embedding_for_characters(
            model.embedding.weight, set(domains.char2ix.keys()) - {"_", ".", "~"}
        )
        self.embedding_only_unrestricted = EmbeddingSubset.get_embedding_for_characters(
            model.embedding.weight, domains.unrestricted_characters
        )
        self.embedding_only_hyphen = EmbeddingSubset.get_embedding_for_characters(
            model.embedding.weight, set("-")
        )

    def get_domains_at_lengths(
        self, embedded_vector: torch.Tensor, lengths: torch.Tensor, norm: RoundingNorm
    ) -> torch.Tensor:
        """Round the embedding vectors to valid encoded domains of the provided lengths using the specified norm.

        The returned domains are valid E2LDs
            (i.e, only consist of allowed characters and do not start or end with a hyphen,
            and do not have a hyphen in both the third and fourth position.)

        :param embedded_vector: A vector of domain embeddings of shape (N, SEQ_LEN, EMBEDDING_DIM).
        :param lengths: A vector of shape (N) containing the lengths to round to.
        :param norm: The norm used to decide on the closest valid embedding.
        :return: A vector of domain encodings of shape (N, SEQ_LEN)
        """
        n_domains = embedded_vector.shape[0]
        lp_target_with_hyphen = (
            self.embedding_with_hyphen.round_embedded_vector_to_encoded_vector(
                embedded_vector, norm
            )
        )
        lp_targets_unrestricted = (
            self.embedding_only_unrestricted.round_embedded_vector_to_encoded_vector(
                embedded_vector, norm
            )
        )

        output = discretization_utils.replace_with_zeros_until_len_individual(
            lp_target_with_hyphen, lengths
        )

        # Last character may not be a hyphen
        output[:, -1] = lp_targets_unrestricted[:, -1]

        hyphen_ix = domains.char2ix["-"]
        # First character may not be a hyphen
        for i in range(n_domains):
            domain_length = lengths[i].item()
            output[i, -domain_length] = lp_targets_unrestricted[i, -domain_length]

            if domain_length >= 4:
                # Not both the third and fourth character may be hyphens (Adhere to IDNA2008)
                would_like_hyphen_in_third_position = (
                    lp_target_with_hyphen[i, -domain_length + 2] == hyphen_ix
                )
                would_like_hyphen_in_fourth_position = (
                    lp_target_with_hyphen[i, -domain_length + 3] == hyphen_ix
                )

                if (
                    would_like_hyphen_in_third_position
                    and would_like_hyphen_in_fourth_position
                ):
                    # Cant have hyphen in third and fourth position. Need to intervene.
                    embed_third_fourth = torch.unsqueeze(
                        embedded_vector[i, -domain_length + 2 : -domain_length + 4],
                        dim=0,
                    )
                    dist_third_fourth = self.embedding_only_hyphen.get_distance_matrix_to_embedding_vectors(
                        embed_third_fourth, norm
                    )[0, 0]

                    if dist_third_fourth[0] < dist_third_fourth[1]:
                        # Character 3 closer to '-' than character 4 --> Setting character 4 to non-hyphen
                        output[i, -domain_length + 3] = lp_targets_unrestricted[
                            i, -domain_length + 3
                        ]
                    else:
                        # Character 4 closer to '-' than character 3 --> Setting character 3 to non-hyphen
                        output[i, -domain_length + 2] = lp_targets_unrestricted[
                            i, -domain_length + 2
                        ]

        return output

    def get_domains_at_length(
        self, embedded_vector: torch.Tensor, length: int, norm: RoundingNorm
    ) -> torch.Tensor:
        """Round the embedding vectors to valid encoded domains of the provided length using the specified norm.

        The returned domains are valid E2LDs
            (i.e, only consist of allowed characters and do not start or end with a hyphen,
            and do not have a hyphen in both the third and fourth position.)

        :param embedded_vector: A vector of domain embeddings of shape (N, SEQ_LEN, EMBEDDING_DIM).
        :param length: The length of the output domains.
        :param norm: The norm used to decide on the closest valid embedding.
        :return: A vector of domain encodings of shape (N, SEQ_LEN)
        """
        lp_target_with_hyphen = (
            self.embedding_with_hyphen.round_embedded_vector_to_encoded_vector(
                embedded_vector, norm
            )
        )
        lp_targets_unrestricted = (
            self.embedding_only_unrestricted.round_embedded_vector_to_encoded_vector(
                embedded_vector, norm
            )
        )

        output = discretization_utils.replace_with_zeros_until_len(
            lp_target_with_hyphen, length
        )

        # Last character may not be a hyphen
        output[:, -1] = lp_targets_unrestricted[:, -1]

        # First character may not be a hyphen
        output[:, -length] = lp_targets_unrestricted[:, -length]

        if length >= 4:
            # Not both the third and fourth character may be hyphens (Adhere to IDNA2008)
            hyphen_ix = domains.char2ix["-"]
            would_like_hyphen_in_third_position = (
                lp_target_with_hyphen[:, -length + 2] == hyphen_ix
            )
            would_like_hyphen_in_fourth_position = (
                lp_target_with_hyphen[:, -length + 3] == hyphen_ix
            )

            would_like_hyphen_in_third_and_fourth = (
                would_like_hyphen_in_third_position
                & would_like_hyphen_in_fourth_position
            )

            embed_third_fourth = embedded_vector[:, -length + 2 : -length + 4]

            dist_third_fourth = torch.squeeze(
                self.embedding_only_hyphen.get_distance_matrix_to_embedding_vectors(
                    embed_third_fourth, norm
                ),
                dim=1,
            )

            distance_third_lower = dist_third_fourth[:, 0] < dist_third_fourth[:, 1]
            pick_third = distance_third_lower & would_like_hyphen_in_third_and_fourth
            pick_fourth = ~distance_third_lower & would_like_hyphen_in_third_and_fourth

            output[pick_third, -length + 3] = lp_targets_unrestricted[
                pick_third, -length + 3
            ]
            output[pick_fourth, -length + 2] = lp_targets_unrestricted[
                pick_fourth, -length + 2
            ]

        return output
