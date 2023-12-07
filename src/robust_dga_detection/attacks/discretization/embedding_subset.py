from __future__ import annotations

from enum import Enum

import torch
import torch.nn.functional as F

from robust_dga_detection.utils import domains


class RoundingNorm(Enum):
    """A norm for calculating the distance between two vectors."""

    L_2 = 0
    L_INF = 1
    COS = 2


class EmbeddingSubset:
    """An EmbeddingSubset is a utility class for calculating distances to (partial) embeddings.

    The EmbeddingSubset is the backbone of all implemented discretization attacks. It can calcualte
    distances from arbitrary embedding-space vectors to valid embeddings using a varierty of different norms.
    """

    reduced_embedding_index_map: torch.Tensor
    reduced_embedding: torch.Tensor

    def __init__(
        self, reduced_embedding_index_map: torch.Tensor, reduced_embedding: torch.Tensor
    ):
        """Create an EmbeddingSubset.

        :param reduced_embedding_index_map: A torch index tensor representing the selected subset indices
        :param reduced_embedding: A torch tensor with the concrete embedding subset
        """
        self.reduced_embedding = reduced_embedding
        self.reduced_embedding_index_map = reduced_embedding_index_map

        assert (
            self.reduced_embedding_index_map.ndim == 1
        ), "Expected reduced_embedding_index_map to have exactly one dimension"

        assert (
            self.reduced_embedding.ndim == 2
            and self.reduced_embedding.shape[0]
            == self.reduced_embedding_index_map.shape[0]
        ), "Expected reduced_embedding to have the shape [NUM_EMBEDDINGS, EMBEDDING_DIM]"

    @staticmethod
    def get_embedding_for_characters(
        embedding_weights: torch.Tensor, chars: set[str]
    ) -> EmbeddingSubset:
        """From the provided full embedding, extract the embedding of the specified characters.

        :param embedding_weights: The full embedding weight matrix of a torch embedding.
        :param chars: The set of characters to include in the embedding-subset.
        :return: The created embedding subset.
        """
        reduced_embedding_index_map = torch.as_tensor(
            sorted([domains.char2ix[char] for char in chars])
        ).to(embedding_weights.device)

        reduced_embedding = embedding_weights[reduced_embedding_index_map]

        return EmbeddingSubset(reduced_embedding_index_map, reduced_embedding)

    def get_distance_matrix_to_embedding_vectors(
        self, embedded_vector: torch.Tensor, norm: RoundingNorm
    ) -> torch.Tensor:
        """Compute the distance matrix for every character in the embedding vector to the embedding subset.

        :param embedded_vector: Tensor of shape (N, SEQ_LEN, EMBEDDING_DIM)
        :param norm: The norm to use for calculating distances
        :return: A Tensor of shape (N, EMBEDDING_DIM, SEQ_LEN)
            where entry (i, j, k) is the distance of character k of domain i to the embedding vector j
        """
        if norm == RoundingNorm.L_2:
            distance_mat = torch.cdist(self.reduced_embedding, embedded_vector, p=2)
        elif norm == RoundingNorm.L_INF:
            distance_mat = torch.cdist(
                self.reduced_embedding, embedded_vector, p=float("inf")
            )
        elif norm == RoundingNorm.COS:
            normalized_embedding_mat = F.normalize(self.reduced_embedding, p=2, dim=1)
            normalized_embedded_vec = F.normalize(embedded_vector, p=2, dim=2)
            distance_mat = 1 - normalized_embedding_mat.matmul(
                normalized_embedded_vec.transpose_(1, 2)
            )
        else:
            raise ValueError(f"Unknown norm {norm}")
        return distance_mat

    def round_embedded_vector_to_encoded_vector(
        self, embedded_vector: torch.Tensor, norm: RoundingNorm
    ) -> torch.Tensor:
        """Compute the closest embedding character for every character vector in the embedding vector.

        :param embedded_vector: Tensor of shape (N, SEQ_LEN, EMBEDDING_DIM)
        :param norm: The norm to use for calculating distances
        :return: A Tensor of shape (N, SEQ_LEN) where entry (i, j) is the index of the embedding
            that is closest to character j of domain i w.r.t. the specified norm
        """
        distance_mat = self.get_distance_matrix_to_embedding_vectors(
            embedded_vector, norm
        )
        return self.reduced_embedding_index_map[torch.argmin(distance_mat, dim=1)]
