# SIRENA &mdash; Sum-Integral REductioN Algorithm

[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-00aa00.svg)](https://arxiv.org/abs/XXXX.XXXXX)

SIRENA is a Python package (with C++ support) for the automatic reduction of sum-integrals, the finite temperature analogs of Feynman integrals in quantum field theory.
It finds momentum shift symmetries among sum-integrals, and exploits integration-by-parts (IBP) relations [\[arXiv:1207.4042\]](https://arxiv.org/abs/1207.4042) via the Laporta algorithm
[\[arXiv:hep-ph/0102033\]](https://arxiv.org/abs/hep-ph/0102033) to express any sum-integral as linear combinations of a set of master sum-integrals which are left to evaluate.

It is the first publicly available code to adapt these techniques to the particular Matsubara sum structure in sum-integrals, which requires a careful treatment of bosonic and fermionic cases separately.
As such, it bridges the gap between theoretical developments and their practical implementation, and provides a simple interface that allows the user to reduce any set of sum-integrals with minimal interaction.

---

## Installation

SIRENA can be installed from the official PyPI repository via the following shell command:

```shell
$ pip install sirena-ibp
```

(Python 3.10+ is required)

---

## Usage

The correct installation of the package can be tested via the shell command

```shell
$ sirena --demo
```

A comprehensive guide to SIRENA can be found in the original publication [\[arXiv:X\]](https://arxiv.org/abs/X). Some input examples are available in the program files, in the `inputs/` directory.

---

## Authors

- **Luis Gil** - _Universidad de Granada_
- **Javier López Miras** - _Universidad de Granada_
- **Adrián Moreno-Sánchez** - _Universidad de Granada_

---

## License

SIRENA is free software under the terms of the GNU General Public License v3.0.

---

## Reference

If you use SIRENA please cite: [\[arXiv:X\]](https://arxiv.org/abs/X).

---

## Acknowledgments

We thank Mikael Chala for sparking this project, and for his continued support during its development. We are also indebted to Pablo Navarrete for useful discussions, and to York Schröder and Philipp Schicho
for providing useful cross-checks with their own private implementations of an IBP reduction algorithm.
