# Hybrid quantum-classical logistic regression for pulsar candidates

This repository contains the research code and reference summary tables
for the HTRU2 pulsar-classification study, *Hybrid Quantum-Classical Logistic
Regression for Calibrated Classification of Pulsar Candidates*.

Repository: <https://github.com/Chanelie/qlr-pulsar-classification>

The study compares three hybrid quantum-classical logistic regression (QLR)
variants, distinguished by their encoding strategies, with logistic regression,
a support vector machine with a radial basis function (RBF) kernel, XGBoost,
and a quantum support vector machine. The analysis covers discrimination,
probability calibration, performance at low false-positive rates, and
simulation runtime.

## Scope

This repository contains the complete implementation and reproducibility
workflow used to produce the reference results.

## Repository layout

```text
qlr/          reusable model, data, metric, and plotting code
scripts/      experiment and analysis entry points
data/         HTRU2 data and attribution
tables/       reference summary tables
docs/         provenance, reproducibility guidance, and result records
tests/        lightweight regression tests
run.py        single user-facing command entry point
```

Generated caches, figures, and logs are not tracked. They are created locally
when the experiments and analysis commands are run.

## Installation

Python 3.10 was used for the experiments. Install the recorded dependencies
with:

```bash
python -m pip install -r requirements.txt
```

## Dataset

The included HTRU2 dataset contains 17,898 candidates and eight real-valued
features. It is distributed by the UCI Machine Learning Repository under
CC BY 4.0. Dataset attribution and its SHA-256 checksum are recorded in
`data/README.md`.

## Citation

Citation metadata and the final repository URL are provided in `CITATION.cff`.

## Licence

No open-source software licence has yet been selected; see `LICENSE`.
The bundled HTRU2 dataset retains its separate CC BY 4.0 licence and
attribution.
