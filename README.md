# Robust DGA Detection
**Robust DGA detection is a library for attacking and defending binary DGA detectors with character-level embeddings.**

:warning: **Note**: This repository accompanies a paper that is currently in a blind peer-review process.
Therefore, this repository masks its authors. This repository will move once the paper has been accepted.

This library is the reference-implementation for the attacks and defenses introduced in the paper: "[PAPER_REF_HERE]".

The datasets used in the paper are strictly confidential. Therefore, we cannot publish our datasets or the resulting models.
As a result, this reference implementation is designed to allow researchers to test their own
Binary DGA Detectors with Character-Level embeddings against our attacks or serve as inspiration for creating derived attacks.

## :floppy_disk: Installation
Currently, this package has not been published to PyPi. Nevertheless, you may install it directly from this repository:

`pip install git+https://github.com/account-for-blind-peer-review/robust-dga-detection`

:warning: **Note**: To make your experience using this library more deterministic, we recommend setting the following environment variables:
- `CUBLAS_WORKSPACE_CONFIG=":4096:8"`
- `PYTHONHASHSEED=0`

## :rocket: Getting Started
The easiest way to get started with this library is by looking at the examples. They contain the necessary documentation
for integrating our attacks and defenses:

- [Combining embedding-space attacks with controllable discretization](examples/Applying%20Embedding%20Space%20Attacks%20with%20Discretization.ipynb)
- [Generating Adversarial Domain Names with HotFlip](examples/Generating%20Adversarial%20Domain%20Names%20with%20HotFlip.ipynb)
- [Generating Adversarial Domain Names with MaskDGA-WB](examples/Generating%20Adversarial%20Domain%20Names%20with%20MaskDGA-WB.ipynb)
- [Generating Batches for Adversarial Training](examples/Generating%20Batches%20for%20Adversarial%20Training.ipynb)

## :pencil: Citing & Contact
**REDACTED**
