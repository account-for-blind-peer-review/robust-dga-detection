import torch
import torch.nn as nn

from robust_dga_detection.utils import domains


class BinaryDgaDetectorWithCharacterEmbedding(nn.Module):
    """A BinaryDgaDetectorWithCharacterEmbedding is a binary DGA-detection network with a character-level embedding.

    Every model has to provide:
    1. A character-level embedding that uses the EXACT SAME character mapping as described in utils/domains.py
    2. A neural network "net" that receives the embedded domain(s) and outputs logits corresponding
        to the propability of the input domain being malicious.
    3. (Optional) a designated input length "sequence_length"
    """

    embedding: nn.Embedding
    net: nn.Module
    sequence_length: int | None

    def __init__(
        self, embedding: nn.Embedding, net: nn.Module, sequence_length: int | None
    ):
        """Create a Binary DGA Detector with Character Embedding.

        :param embedding: A Character-Level embedding that uses the EXACT SAME character mapping
            as described in utils/domains.py
        :param net: A neural network "net" that receives the embedded domain(s) and outputs logits corresponding
             to the propability of the input domain being malicious.
        :param sequence_length: (Optional) a designated input length
        """
        super().__init__()
        self.embedding = embedding

        assert self.embedding.num_embeddings == len(domains.char2ix), (
            f"The provided embedding has {self.embedding.num_embeddings} embeddings. "
            f"An embedding layer with {len(domains.char2ix)} embeddings was expected."
        )

        self.net = net
        self.sequence_length = sequence_length

    def forward(self, batch_x: torch.Tensor) -> torch.Tensor:
        """Overwridden.

        :param batch_x: A tensor of shape [BATCH_SIZE, SEQUENCE_LENGHT] containing the encoded domain names
        """
        assert batch_x.ndim == 2, (
            f"The network input has {batch_x.ndim} dimensions, "
            f"expected 2 ([BATCH_SIZE, SEQUENCE_LENGTH])"
        )

        assert (
            self.sequence_length is None or batch_x.shape[1] == self.sequence_length
        ), f"Expected input of shape [BATCH_SIZE, {self.sequence_length}]. Received [BATCH_SIZE, {batch_x.shape[1]}]"

        encoded_x = self.embedding(batch_x.long())
        classifier_output = self.net(encoded_x)

        return classifier_output.flatten()
