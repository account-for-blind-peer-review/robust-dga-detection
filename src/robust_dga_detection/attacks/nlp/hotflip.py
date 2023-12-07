from collections.abc import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from robust_dga_detection.attacks.nlp.nlp_attack import NLPAttack
from robust_dga_detection.utils import domains


class HotFlip(NLPAttack):
    """The HotFlip attack.

    As seen in the paper "HotFlip: White-Box Adversarial Examples for Text Classification"
    by Javid Ebrahimi, Anyi Rao, Daniel Lowd, Dejing Dou
    https://arxiv.org/abs/1712.06751

    Adapted for attacking E2LD-Only classifiers. Without explicit insert / deletion estimations.
    """

    beam_width: int
    max_flips: int

    def __init__(self, target_model: nn.Module, beam_width: int, max_flips: int):
        """Create a HotFlip Attack instance.

        :param target_model the model to attach
        :param beam_width the width of the beam-search beam
        :param max_flips the maximum number of flips per domain (i.e., maximum search tree depth)
        """
        super().__init__(target_model)
        self.beam_width = beam_width
        self.max_flips = max_flips

    def _prime_attack(
        self, encoded_domains: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = encoded_domains.shape[1]
        num_domains = encoded_domains.shape[0]

        score_matrix = self._calculate_scoring_matrix_on_encoded_domains(
            encoded_domains, labels
        )
        output_encoded_domains = torch.zeros(
            (num_domains * self.beam_width, seq_len),
            device=encoded_domains.device,
            dtype=torch.long,
        )
        output_labels = torch.zeros(
            num_domains * self.beam_width,
            device=encoded_domains.device,
            dtype=torch.long,
        )

        for i in range(num_domains):
            output_encoded_domains[
                i * self.beam_width : (i + 1) * self.beam_width
            ] = self._get_next_generation_beam_targets_for_domain(
                encoded_domains[[i]], score_matrix[[i]]
            )
            output_labels[i * self.beam_width : (i + 1) * self.beam_width] = labels[i]
        return output_encoded_domains, output_labels

    def _perform_beam_search_step(
        self, encoded_domains: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_domains = encoded_domains.shape[0] // self.beam_width

        score_matrix = self._calculate_scoring_matrix_on_encoded_domains(
            encoded_domains, labels
        )
        output_encoded_domains = torch.zeros_like(encoded_domains)

        for i in range(num_domains):
            output_encoded_domains[
                i * self.beam_width : (i + 1) * self.beam_width
            ] = self._get_next_generation_beam_targets_for_domain(
                encoded_domains[i * self.beam_width : (i + 1) * self.beam_width],
                score_matrix[i * self.beam_width : (i + 1) * self.beam_width],
            )

        return output_encoded_domains, labels

    def _calculate_scoring_matrix_on_encoded_domains(
        self, encoded_domains: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        seq_len = encoded_domains.shape[1]
        num_chars = len(domains.char2ix)
        num_domains = encoded_domains.shape[0]

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

        criterion = nn.BCEWithLogitsLoss()
        criterion_no_red = nn.BCEWithLogitsLoss(reduce=False)

        logits = self.target_model(one_hot_encoded_domains).flatten()
        loss = criterion(logits, labels.float())
        self.target_model.zero_grad()
        loss.backward()

        # Calculate position of first padding character
        length_mask = (encoded_domains == 0) * (torch.arange(seq_len, 0, -1) - 1).to(
            encoded_domains.device
        )
        lengths, _ = torch.min(
            torch.where(length_mask > 0, length_mask, seq_len), dim=1
        )

        jsm = one_hot_encoded_domains.grad
        hyphen_idx = domains.char2ix["-"]

        for i in range(num_domains):
            length = lengths[i]
            # Subtract identity
            jsm[i] -= (
                jsm[i, torch.arange(seq_len), encoded_domains[i].long()]
                .expand((num_chars, seq_len))
                .T
            )

            # Not allowed to correct to "-" at start of a domain
            jsm[i, -length, hyphen_idx] = -torch.inf

            # Not allowed to shrink the domain if the second character is a "-"
            if encoded_domains[i, -length + 1] == hyphen_idx:
                jsm[i, -length, domains.char2ix["~"]] = -torch.inf

            # Also not allowed to switch the potential "new" start of the domain to a "-"
            if length < seq_len:
                jsm[i, -length - 1, hyphen_idx] = -torch.inf

            # Handle that the third and fourth character cannot be '-' together:

            # If third character "-" --> Not allowed to switch fourth to "-"
            if encoded_domains[i, -length + 2] == hyphen_idx:
                jsm[i, -length + 3, hyphen_idx] = -torch.inf

            # If fourth character "-" --> Not allowed to switch third to "-"
            if encoded_domains[i, -length + 3] == hyphen_idx:
                jsm[i, -length + 2, hyphen_idx] = -torch.inf

            # If fourth and fifth character both "-" --> Not allowed to shrink the domain
            if (
                encoded_domains[i, -length + 3] == hyphen_idx
                and encoded_domains[i, -length + 4] == hyphen_idx
            ):
                jsm[i, -length, domains.char2ix["~"]] = -torch.inf

            # If second and third character both "-" --> Not allowed to extend the domain
            if (
                encoded_domains[i, -length + 1] == hyphen_idx
                and encoded_domains[i, -length + 2] == hyphen_idx
            ):
                jsm[i, -length - 1, :] = -torch.inf

            # Do not allow to switch to a padding character in the middle of the domain
            # Switching the last char to a padding char is okay though
            jsm[i, -length + 1 :, domains.char2ix["~"]] = -torch.inf

            # Do not allow to switch to anything in the padding area (except for the last padding char)
            if length < seq_len:
                jsm[i, : -length - 1, :] = -torch.inf

        # Correct JSM to prevent selection of "illegal" characters
        jsm[:, :, domains.char2ix["_"]] = -torch.inf
        jsm[:, :, domains.char2ix["."]] = -torch.inf

        # Not allowed to correct to "-" at the end of a domain
        jsm[:, -1, hyphen_idx] = -torch.inf

        # Add the current loss to the JSM to approximate the loss after the change
        jsm += (
            criterion_no_red(logits, labels.float())
            .detach()
            .expand((num_chars, seq_len, num_domains))
            .transpose(0, 2)
        )

        return jsm

    def _get_next_generation_beam_targets_for_domain(
        self, encoded_domains: torch.Tensor, score_matrix: torch.Tensor
    ) -> torch.Tensor:
        topk_val, topk_ind = torch.topk(score_matrix.flatten(), self.beam_width)
        topk_ind = np.array(
            np.unravel_index(topk_ind.cpu().numpy(), score_matrix.shape)
        ).T
        return_domains = []
        for i in range(self.beam_width):
            domain_indx = topk_ind[i, 0]
            position_to_flip = topk_ind[i, 1]
            char_to_flip_to = topk_ind[i, 2]

            new_domain = encoded_domains[domain_indx].detach().clone()
            new_domain[position_to_flip] = char_to_flip_to
            return_domains.append(new_domain)
        return torch.stack(return_domains)

    def _get_output_domains_from_beamsearch(
        self,
        encoded_domains: torch.Tensor,
        beamsearch_matrix: torch.Tensor,
        beamsearch_labels: torch.Tensor,
    ) -> torch.Tensor:
        criterion = nn.BCEWithLogitsLoss(reduce=False)
        output_domains = torch.zeros_like(encoded_domains)
        with torch.no_grad():
            model_value_on_beamsearch_matrix = self.target_model(beamsearch_matrix)
            loss = criterion(
                model_value_on_beamsearch_matrix, beamsearch_labels.float()
            )

        for i in range(encoded_domains.shape[0]):
            best_idx = torch.argmax(
                loss[i * self.beam_width : (i + 1) * self.beam_width]
            )
            output_domains[i] = beamsearch_matrix[i * self.beam_width + best_idx]

        return output_domains

    def iforward(
        self,
        input_domains: list[str],
        encoded_domains: torch.Tensor,
        labels: torch.Tensor,
    ) -> Iterable[torch.Tensor]:
        """Perform an 'iterative' forward that also reports intermediate results for each flip."""
        beamsearch_matrix, beamsearch_labels = self._prime_attack(
            encoded_domains, labels
        )
        yield self._get_output_domains_from_beamsearch(
            encoded_domains, beamsearch_matrix, beamsearch_labels
        )

        for i in range(self.max_flips - 1):
            beamsearch_matrix, beamsearch_labels = self._perform_beam_search_step(
                beamsearch_matrix, beamsearch_labels
            )
            yield self._get_output_domains_from_beamsearch(
                encoded_domains, beamsearch_matrix, beamsearch_labels
            )

    def forward(
        self,
        input_domains: list[str],
        encoded_domains: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Overriden."""
        beamsearch_matrix, beamsearch_labels = self._prime_attack(
            encoded_domains, labels
        )

        for i in range(self.max_flips - 1):
            beamsearch_matrix, beamsearch_labels = self._perform_beam_search_step(
                beamsearch_matrix, beamsearch_labels
            )

        return self._get_output_domains_from_beamsearch(
            encoded_domains, beamsearch_matrix, beamsearch_labels
        )
