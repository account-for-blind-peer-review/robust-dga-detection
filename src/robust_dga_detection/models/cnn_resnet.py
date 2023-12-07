import contextlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from robust_dga_detection.models.binary_dga_detector_with_character_embedding import (
    BinaryDgaDetectorWithCharacterEmbedding,
)


class CNNResNetWithEmbedding(BinaryDgaDetectorWithCharacterEmbedding):
    """A PyTorch implementation of the 'B-ResNet' model introduced by Drichel et al.

    The Implementation is based on the paper:
    Arthur Drichel, Ulrike Meyer, Samuel Schüppen, and Dominik Teubert. 2020.
    Analyzing the real-world applicability of DGA classifiers.
    In Proceedings of the 15th International Conference on Availability, Reliability and Security (ARES '20).
    Association for Computing Machinery, New York, NY, USA, Article 15, 1–11. DOI:https://doi.org/10.1145/3407023.3407030
    """

    embedding_dim: int
    vocab_size: int

    def __init__(
        self,
        embedding_dim: int = 128,
        vocab_size: int = 40,
        out_channels: int = 128,
        seq_len: int = 63,
    ):
        """Construct a new B-ResNet model.

        :param embedding_dim: the number of dimensions of the embedding (Use 128 to match reference paper)
        :param vocab_size: the number of characters in the embedding (Use 40 to match reference paper)
        :param out_channels: the number of output channels of the convolutional blocks (Use 128 to match reference)
        :param seq_len: the length of the domains (Use 63 for e2LDs)
        """
        super().__init__(
            embedding=nn.Embedding(
                num_embeddings=vocab_size, embedding_dim=embedding_dim
            ),
            net=CNNResNet(embedding_dim, out_channels, seq_len),
            sequence_length=seq_len,
        )


class ResidualITsec(nn.Module):
    """Residual convolutional block with two convolutional layer and a skip connection.

    The skip connection is scales the number of channels if needed to match the output
    dimensions of the last convolutional layer
    """

    out_channels: int
    in_channels: int
    seq_len: int
    channel_adjust: nn.Conv1d
    conv1: nn.Conv1d
    conv2: nn.Conv1d

    def __init__(self, in_channels: int, out_channels: int, seq_len: int):
        """Construct a new residual block.

        :param in_channels: the number fo input channels
        :param out_channels: the number of output channels
        :param seq_len: the length of the input sequence
        """
        super().__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.seq_len = seq_len

        self.channel_adjust = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding="same",
        )
        self.conv1 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=4,
            stride=1,
            padding="same",
        )
        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=4,
            stride=1,
            padding="same",
        )

    def forward(self, channel_first_batch_x: torch.Tensor) -> torch.Tensor:
        """Compute neural network output.

        :param channel_first_batch_x: Expected input shape [BATCH_SIZE, IN_CHANNELS, SEQ_LENGTH]
        :return: Output of shape [BATCH_SIZE, OUT_CHANNELS, SEQ_LENGTH]
        """
        assert channel_first_batch_x.ndim == 3
        assert channel_first_batch_x.shape[1] == self.in_channels
        assert channel_first_batch_x.shape[2] == self.seq_len

        residual = channel_first_batch_x

        # scale input_channels to output channels if needed
        if self.out_channels != self.in_channels:
            residual = self.channel_adjust(channel_first_batch_x)

        # We want centered 0 padding
        # With Kernel Size 4, Stride 1, Padding 0, Dilation 1 --> Output has 3 elements less than input
        # ==> Pad 2 Elements on Left, 1 on Right
        channel_first_batch_x = self.conv1(channel_first_batch_x)
        channel_first_batch_x = F.relu(channel_first_batch_x)
        channel_first_batch_x = self.conv2(channel_first_batch_x)

        channel_first_batch_x = residual + channel_first_batch_x
        return channel_first_batch_x


class CNNResNet(nn.Module):
    """A PyTorch implementation of the ResNet model following the embedding of the 'B-ResNet' model."""

    seq_len: int
    embedding_dim: int
    residual_block_output_channels: int
    residual_conv: ResidualITsec
    sigmoid_output: bool

    def __init__(
        self, embedding_dim: int, residual_block_output_channels: int, seq_len: int
    ):
        """Construct a new ResNet model.

        :param embedding_dim: the dimensionality of the embedding
        :param residual_block_output_channels: the number of output channels of the residual block
        :param seq_len: the input sequence length
        """
        super().__init__()
        self.seq_len = seq_len
        self.residual_block_output_channels = residual_block_output_channels
        self.embedding_dim = embedding_dim

        self.residual_conv = ResidualITsec(
            in_channels=embedding_dim,
            out_channels=residual_block_output_channels,
            seq_len=seq_len,
        )

        # pooling size is 4, with 'same' padding
        in_features = residual_block_output_channels * (math.ceil(seq_len / 4))

        self.linear_out = nn.Linear(in_features, 1)

        self.sigmoid_output = False

    @contextlib.contextmanager
    def with_sigmoid_output(self):
        """Open a context in which this model returns sigmoids instead of logits."""
        self.sigmoid_output = True
        yield
        self.sigmoid_output = False

    def forward(self, character_first_batch: torch.Tensor) -> torch.Tensor:
        """Compute neural network output.

        :param character_first_batch: Expected input shape [BATCH_SIZE, SEQ_LEN, EMBEDDING_DIM]
        :return: Output of shape [BATCH_SIZE]. By default, the output is piped through a sigmoid.
        """
        # Expected input Format [BATCH_SIZE, SEQ_LEN, EMBEDDING_DIM]
        assert character_first_batch.ndim == 3
        assert character_first_batch.shape[1] == self.seq_len
        assert character_first_batch.shape[2] == self.embedding_dim

        # x: (batch_size, seq_len, embedding_dim) -> (batch_size, embedding_dim, seq_len) (in_channels=embedding_dim)
        # The Conv1D Expects an input of format [BATCH_SIZE, EMBEDDING_DIM, SEQ_LEN]
        embedding_first_batch = character_first_batch.permute(0, 2, 1)

        embedding_first_batch = self.residual_conv(embedding_first_batch)
        embedding_first_batch = F.relu(embedding_first_batch)
        embedding_first_batch = F.max_pool1d(
            embedding_first_batch, kernel_size=4, padding=2
        )

        flattened_output = embedding_first_batch.reshape(
            embedding_first_batch.size(0), -1
        )

        logits = self.linear_out(flattened_output)

        if self.sigmoid_output:
            return torch.sigmoid(logits)
        else:
            return logits
