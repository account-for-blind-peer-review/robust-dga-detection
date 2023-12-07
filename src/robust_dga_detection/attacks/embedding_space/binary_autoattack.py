from torchattacks.attack import Attack
from torchattacks.attacks.fab import FAB
from torchattacks.wrappers.multiattack import MultiAttack

from robust_dga_detection.attacks.embedding_space.binary_apgd import BinaryAPGD


class BinaryAutoAttack(Attack):
    r"""A version of AutoAttack modified for Binary Classification Tasks.

    AutoAttack from the paper
    'Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks'
    [https://arxiv.org/abs/2003.01690]
    [https://github.com/fra31/auto-attack]

    Implementation based on:
    https://github.com/Harry24k/adversarial-attacks-pytorch/blob/c4da6a95546283992a3d1816ae76a0cd4dfc2d8b/torchattacks/attacks/autoattack.py

    Distance Measure : Linf, L2

    Arguments:
        model (nn.Module): model to attack.
        norm (str) : Lp-norm to minimize. ['Linf', 'L2'] (Default: 'Linf')
        eps (float): maximum perturbation. (Default: 0.3)
        verbose (bool): print progress. (Default: False)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,
            `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.
    """

    def __init__(
        self, model, norm="Linf", eps=8 / 255, random_seed=406069, verbose=False
    ):
        """Create a Binary AutoAttack instance.

        :param model the model to attack
        :param norm Lp norm to minimize ('LInf' or 'L2')
        :param eps maximum perturbation
        :param verbose print progress?
        """
        super().__init__("AutoAttack", model)
        self.norm = norm
        self.eps = eps
        self.n_classes = 2
        self.verbose = verbose
        self.supported_mode = ["default"]

        self._autoattack = MultiAttack(
            [
                BinaryAPGD(
                    model,
                    eps=eps,
                    norm=norm,
                    seed=random_seed,
                    verbose=verbose,
                    loss="ce",
                    n_restarts=1,
                ),
                BinaryAPGD(
                    model,
                    eps=eps,
                    norm=norm,
                    seed=random_seed,
                    verbose=verbose,
                    loss="cw",
                    n_restarts=1,
                ),
                FAB(
                    model,
                    eps=eps,
                    norm=norm,
                    seed=random_seed,
                    verbose=verbose,
                    n_classes=2,
                    n_restarts=1,
                ),
            ]
        )

    def forward(self, images, labels):
        """Overridden."""
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        adv_images = self._autoattack(images, labels)

        return adv_images
