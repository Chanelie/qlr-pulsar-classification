# Script guide

## Main workflow

| Script | Purpose |
|---|---|
| `run_one.py` | Train one QLR configuration or one classical/QSVM block. |
| `grid_plan.py` | Produce the complete main-grid job list and scheduling estimate. |
| `run_grid.sh` | Launch the Linux main grid with isolated processes. |
| `aggregate_all.py` | Verify cache coverage and aggregate the main grid. |
| `build_all_layer_summary_tables.py` | Build cross-depth summary CSV files. |
| `generate_figures.py` | Generate figures from saved outputs without training. |

Use the top-level `run.py` interface for these operations whenever possible.

## Supported additional analyses

| Script | Purpose |
|---|---|
| `baselines_weighting_comparison_N1000.py` | Compare weighted and unweighted classical baselines. |
| `posthoc_calibration_train.py` | Fit post-hoc calibration using training data only. |
| `bootstrap_test_set.py` | Run a paired, class-stratified bootstrap of test predictions. |
| `bootstrap_summary.py` | Summarise the bootstrap output. |
| `run_split_robustness.py` | Run one varying-split robustness job. |
| `split_robustness_analysis.py` | Aggregate the varying-split study. |
| `ten_seed_analysis_N200.py` | Aggregate the ten-seed fixed-split study. |
| `paired_seed_stats.py` | Shared statistical helpers for robustness analyses. |
| `regen_deltas_N1000_L3.py` | Regenerate paired-difference tables for N=1000, L=3. |

Only the supported experiment and analysis scripts listed above are included.
