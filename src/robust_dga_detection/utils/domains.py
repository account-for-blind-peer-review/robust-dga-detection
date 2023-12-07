import torch
import numpy as np

char2ix: dict[str, int] = {
    "~": 0,  # Padding character
    "a": 1,
    "b": 2,
    "c": 3,
    "d": 4,
    "e": 5,
    "f": 6,
    "g": 7,
    "h": 8,
    "i": 9,
    "j": 10,
    "k": 11,
    "l": 12,
    "m": 13,
    "n": 14,
    "o": 15,
    "p": 16,
    "q": 17,
    "r": 18,
    "s": 19,
    "t": 20,
    "u": 21,
    "v": 22,
    "w": 23,
    "x": 24,
    "y": 25,
    "z": 26,
    "0": 27,
    "1": 28,
    "2": 29,
    "3": 30,
    "4": 31,
    "5": 32,
    "6": 33,
    "7": 34,
    "8": 35,
    "9": 36,
    "-": 37,
    "_": 38,
    ".": 39,
}

ix2char = {ix: char for char, ix in char2ix.items()}
ix2char_numpy = np.array([ix2char[key] for key in sorted(ix2char.keys())])

unrestricted_characters = set("abcdefghijklmnopqrstuvwxyz0123456789")


def encode_domain(domain: str, padding_target: int | None = 63) -> torch.Tensor:
    """Encode a domain to a tensor.

    Encodes the domain to a tensor by mapping each character to unique integer. If a padding_target is provided
    the input will be padded to the left with zeros.

    :param domain: The domain to encode.
    :param padding_target: The target length to pad the output to.
    :return: The encoded domain.
    """
    prepared_domain = domain.lower()

    if padding_target is not None:
        assert padding_target >= len(domain), "Cannot apply negative padding"
        domain_vector = torch.zeros(padding_target, dtype=torch.int)
    else:
        domain_vector = torch.zeros(len(domain), dtype=torch.int)

    domain_start_idx = len(domain_vector) - len(domain)

    for i in range(len(domain)):
        domain_vector[domain_start_idx + i] = char2ix[prepared_domain[i]]
    return domain_vector


def decode_domains(encoded_domains: torch.Tensor) -> list[str]:
    """Decode domains from a 2d-tensor.

    Decodes a 2d tensor to a list of domain. Stops at the first padding character (0).

    :param encoded_domains: a 2D Tensor of Shape [NUMBER_OF_DOMAINS, DOMAIN_LENGTH].
    :return: The decoded domains.
    """
    seq_len = encoded_domains.shape[1]
    encoded_domains = encoded_domains.cpu().numpy()
    decoded_domains = ix2char_numpy[encoded_domains]
    lengths = np.argmax(np.flip((encoded_domains == 0), axis=1), axis=1)
    has_padding_character = np.any(encoded_domains == 0, axis=1)
    lengths[~has_padding_character] = seq_len
    outputs = [
        "".join(decoded_domains[i, seq_len - lengths[i] : seq_len])
        if (lengths[i] > 0)
        else ""
        for i in range(encoded_domains.shape[0])
    ]
    return outputs
