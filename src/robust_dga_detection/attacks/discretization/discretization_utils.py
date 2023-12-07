import torch


def replace_with_zeros_until_len_individual(
    lp_targets: torch.Tensor, lengths: torch.Tensor
) -> torch.Tensor:
    """Cut row lp_targets[i] to length lengths[i] for all i.

    :param lp_targets: Int-Tensor of shape (N, SEQ_LEN)
    :param lengths: Int-Tensor of shape (N)
    :return: Int-Tensor of Shape (N, SEQ_LEN) such that the first (SEQ_LEN -lengths[i]) cols are set to 0 (padding)
        for each i.
    """
    assert (
        lp_targets.shape[0] == lengths.shape[0]
        and lp_targets.ndim == 2
        and lengths.ndim == 1
    )
    output = torch.zeros_like(lp_targets)
    for i in range(lp_targets.shape[0]):
        output[i, -lengths[i] :] = lp_targets[i, -lengths[i] :]
    return output


def replace_with_zeros_until_len(lp_targets: torch.Tensor, len: int) -> torch.Tensor:
    """Replace the first (SEQ_LEN - len) columns with zeros.

    :param lp_targets: Int-Tensor of shape (N, SEQ_LEN)
    :param len: the length to cut the tensor to
    :return: Int-Tensor of Shape (N, SEQ_LEN) where the first (SEQ_LEN - len) Cols are set to 0 (padding),
     effectively cutting the domains to length len
    """
    assert 0 < len <= lp_targets.shape[1] and lp_targets.ndim == 2
    output_lp_targets = torch.clone(lp_targets)
    output_lp_targets[:, :-len] = 0
    return output_lp_targets
