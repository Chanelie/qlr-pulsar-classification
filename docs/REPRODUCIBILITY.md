# Reproducibility guide

## Recorded environment

The completed experimental grid used Python 3.10.20, PennyLane 0.42.3,
PennyLane-Lightning 0.42.0, NumPy 2.2.6, SciPy 1.15.3, pandas 2.3.3,
scikit-learn 1.7.2, and XGBoost 3.2.0. The full dependency list is in
`requirements.txt` and `environment.yml`.

## Experimental protocol

- fixed stratified 80/20 train/test split with random state 42;
- stratified training subsets at N = 200, 500, and 1,000;
- experiment seeds 0, 1, and 2 for the main grid;
- circuit depths L = 1, 2, 3, 5, and 10;
- analytic simulation (`shots=None`) with `lightning.qubit`;
- parameter-shift gradients;
- 60 maximum epochs, batch size 16, learning rate 0.01;
- L2 coefficient 1e-4;
- QLR angle scale pi/4.

Coverage and interpretation notes are recorded in `docs/RESULTS_RECORD.md`.

## Commands

Check the planned grid without starting jobs:

```bash
python run.py plan
python run.py run-grid --dry-run
```

Run one configuration:

```bash
python run.py run-one --kind qlr --samples 200 --layers 3 --encoding angle --seed 0
```

Run one baseline/QSVM block:

```bash
python run.py run-one --kind baselines --samples 200 --seed 0
```

The Linux grid launcher requires GNU `screen` and `taskset`. Set
`QLR_PYTHON` to the intended environment interpreter and inspect the configured
CPU list before launching. Runtime comparisons are sensitive to concurrent
machine load.

## Generate results and figures

Binary caches are not included in the repository. Run the experiment grid to
create them under `caches/`, then aggregate the results and generate figures:

```bash
python run.py run-grid
python run.py aggregate
python run.py figures --set all
```

The full grid is computationally expensive and the launcher requires Linux,
GNU `screen`, and `taskset`. Do not combine caches from different experiment
runs when comparing or reporting results.
