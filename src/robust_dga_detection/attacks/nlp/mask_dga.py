import torch
import torch.nn as nn
import torch.nn.functional as F

from robust_dga_detection.attacks.nlp.nlp_attack import NLPAttack
from robust_dga_detection.utils import domains


class MaskDGA(NLPAttack):
    """The White-Box portion of the MaskDGA attack.

    As seen in the paper "MaskDGA: A Black-box Evasion Technique Against DGA Classifiers and Adversarial Defenses"
    by Lior Sidi, Asaf Nadler, Asaf Shabtai
    https://arxiv.org/abs/1902.08909

    Adapted for attacking E2LD-Only classifiers.
    """

    def forward(
        self,
        input_domains: list[str],
        encoded_domains: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Overriden."""
        # Convert encoded domains tensor to character one-hot tensor
        one_hot_encoded_domains = (
            F.one_hot(
                encoded_domains.long().to(encoded_domains.device),
                num_classes=len(domains.ix2char),
            )
            .float()
            .detach()
        )
        one_hot_encoded_domains.requires_grad = True

        # Embed the one-hot tensor, run it through the model, and compute the gradients to get the JSM
        criterion = nn.BCEWithLogitsLoss()

        logits = self.target_model(one_hot_encoded_domains).flatten()
        loss = criterion(logits, labels.float())
        self.target_model.zero_grad()
        loss.backward()

        jsm = one_hot_encoded_domains.grad

        # Correct JSM to prevent selection of "illegal" characters
        jsm[:, :, domains.char2ix["_"]] = -torch.inf
        jsm[:, :, domains.char2ix["."]] = -torch.inf

        # Not allowed to correct to "-" at the end of a domain
        jsm[:, -1, domains.char2ix["-"]] = -torch.inf
        hyphen_idx = domains.char2ix["-"]

        for i, domain in enumerate(input_domains):
            # Not allowed to correct to "-" at start of a domain
            jsm[i, -len(domain), hyphen_idx] = -torch.inf

            # Not allowed to correct to "-" such that both the third and the fourth character are hyphens
            # THIS IS A SUB-OPTIMAL SOLUTION
            if encoded_domains[i, -len(domain) + 2] == hyphen_idx:
                jsm[i, -len(domain) + 3, hyphen_idx] = -torch.inf
            else:
                jsm[i, -len(domain) + 2, hyphen_idx] = -torch.inf

            # MaskDGA: Only allowed to change characters inside the domain
            # (i.e., len(input_domain) == len(output_domain)
            jsm[i, -len(domain) :, domains.char2ix["~"]] = -torch.inf
            jsm[i, : -len(domain), :] = -torch.inf

        # Compute MaskDGA Output
        output_encoded_domains = encoded_domains.clone()
        max_val, max_ind = torch.max(jsm, dim=2)
        sorted_order = torch.argsort(max_val, descending=True)
        for i, domain in enumerate(input_domains):
            to_replace = sorted_order[i, : len(domain) // 2]
            output_encoded_domains[i, to_replace] = max_ind[i, to_replace].int()

        return output_encoded_domains
