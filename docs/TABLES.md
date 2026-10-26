# Reference table guide

The CSV files under `tables/` are reference outputs from the corrected
workflow. They are included for transparency and comparison with independently
reproduced results.

## Main-grid tables

- `qlr_mean_*` and `qlr_se_*`: QLR metrics by training size and depth;
- `baselines_mean_*` and `baselines_se_*`: classical/QSVM summaries;
- `deltas_mean_*` and `deltas_se_*`: paired differences;
- `summary_*_all_layers.csv`: discrimination summaries across depths;
- `calibration_*_all_layers.csv`: calibration summaries across depths;
- `strict_fpr_*_all_layers.csv`: low-FPR summaries across depths;
- `runtime_*_all_layers.csv`: recorded runtime summaries.

## Additional-analysis tables

- `calibrated_baselines_train_*`: training-data-fitted post-hoc calibration;
- `bootstrap_intervals.csv`: paired bootstrap intervals;
- `ten_seed_N200_L3.csv`: ten-seed fixed-split analysis;
- `split_robustness_N200_L3.csv`: varying-split analysis.

The additional-analysis tables extend the main experimental grid with
calibration and robustness checks.
