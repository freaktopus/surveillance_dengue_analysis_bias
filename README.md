# Surveillance Dengue Analysis Bias

## Purpose

This repository holds the analysis pipeline for a study of surveillance bias in dengue forecasting. The study uses weekly dengue case counts for the districts of Nepal from 2022 to 2025.

The pipeline does these tasks:

1. It groups the districts into three clusters by surveillance level.
2. It adds three types of surveillance bias to the clean case counts.
3. It measures how each bias changes the forecasts of an RNN model and an LSTM model.
4. It tests a bias-aware correction filter. The filter uses only prior knowledge of the bias.
5. It compares the neural network models with a seasonal SEIR model.
6. It tests the trained models on data from Kaski district.

All code is in the folder `v_05`.

## Study design

| Item | Setting |
|---|---|
| Districts | Nepal has 77 districts. KASKI is excluded by choice from the main analysis. The main analysis uses 76 districts. KASKI is used only in the generalizability test (steps 12a to 14). |
| Time unit | ISO week. The case snapshots cover 2022 to 2025. |
| Training and test years | Training uses 2022 to 2024. The test year is 2025. Steps 13 and 14 use a different cutoff (see step 13). |
| Clusters | Three clusters from k-means (random seed 42). The inputs are urbanization percentage, health facilities per 100,000 people, and population density. |
| Bias types | B1 is the reporting rate. B2 is the reporting lag. B3 is the detection (ascertainment) probability. The type "combined" applies all three. |
| Monte Carlo draws | 100 draws of the bias parameters for each district. |
| Models | RNN and LSTM. Each model has one recurrent layer with 16 hidden units. The input is the last 8 weeks. The output is a one-week-ahead forecast. |
| Scores | RMSE, MAE, mean forecast bias, correlation, and negative binomial negative log-likelihood (NB-NLL). |
| Scoring reference | Forecasts from biased series are scored against the clean case counts. |

### Clusters

The district counts are those before KASKI is removed.

| Cluster | Label | Districts |
|---|---|---|
| 1 | Urban / High Surveillance | 2 |
| 2 | Peri-urban / Mid Surveillance | 24 |
| 3 | Rural / Low Surveillance | 51 |

### Bias prior parameters

Step 02 sets these values. Step 04 samples bias parameters from them.

| Cluster | B1 reporting rate (Beta) | B2 lag in days (Gamma: shape, scale) | B3 detection probability (Beta) |
|---|---|---|---|
| 1 | Beta(17, 3), mean 0.85 | Gamma(1.0, 1.0), mean 1 day | Beta(15, 5), mean 0.75 |
| 2 | Beta(6, 4), mean 0.60 | Gamma(2.5, 2.0), mean 5 days | Beta(5, 5), mean 0.50 |
| 3 | Beta(3.5, 6.5), mean 0.35 | Gamma(3.5, 4.0), mean 14 days | Beta(2.5, 7.5), mean 0.25 |

## Repository content

| Path | Content |
|---|---|
| `v_05/` | All pipeline scripts, numbered in run order. |
| `v_05/common.py` | Shared code: paths, data loaders, models, metrics, and the correction functions. Other scripts import this file. Do not run it alone. |
| `v_05/inputs/district_covariates.csv` | District covariates. One row for each of the 77 districts. |
| `v_05/v_05_ouput_final_run/` | Log and manifest of the final run. The folder name is spelled as in the repository. |
| `LICENSE` | MIT License. |

The scripts create the folders `v_05/outputs` and `v_05/figures` when they run.

## Requirements

- Python 3.11.2. The final run used this version.
- These libraries: numpy, pandas, scipy, scikit-learn, matplotlib, seaborn, and PyTorch.
- The repository has no file that lists library versions.
- Step 11 uses Helvetica-compatible fonts (Nimbus Sans) when it finds them. If it does not find them, it uses DejaVu Sans.

## Input data

**Included.** The file `v_05/inputs/district_covariates.csv` has these columns: district, province, population, area_km2, urbanization_pct, population_density, and health_facility_per100k.

**Not included.** The case count files are not in this repository.

- Steps 03 and 12a read CSV files from a folder named `data` at the repository root.
- Step 12b reads CSV files from a folder named `data_test_kaski` at the repository root.
- Each file is named `YYYY-MM-DD.csv`. Each file holds the year-to-date cumulative case count for each district (column TOTAL).
- Two file layouts are accepted. The simple layout has the columns District and TOTAL. The wide layout has the columns SN, PROVINCE, DISTRICT, the month columns, and TOTAL.
- The final run used 170 files, from 2022-01-31 to 2025-12-07.

Steps 01 and 02 run with the included file only. Step 03 and all later steps need the case count files.

## Sequence of operations

Run the steps in the order below. The file `v_05/v_05_ouput_final_run/run_manifest.txt` records the order of the final run: 01, 02, 03, 04, 05, 06, 07, 08, 09, 10, 11, 12a, 12b, 13, 14, 15.

Each script finds its folders from its own file location. The working folder does not matter. The log shows that steps 01 to 06 ran without options.

Read the section "Folder notes" before you run steps 07, 09, 10, 11, and 15.

| Step | Script | What it does | Needs |
|---|---|---|---|
| 01 | `01_data_ingestion.py` | Checks the district covariate file. Confirms the 77 district names and the required columns. | Covariate file |
| 02 | `02_spatial_clustering.py` | Groups the districts into 3 clusters. Sets the bias prior parameters for each cluster. Makes the cluster figures. | Covariate file |
| 03 | `03_case_count_ingestion.py` | Reads the case snapshot files. Converts the cumulative counts to incident counts. Spreads the counts over ISO weeks. Removes KASKI. | `data/` folder |
| 04 | `04_bias_injection.py` | Adds B1, B2, B3, and combined bias to the weekly panel. Makes 100 draws. Uses random seed 42. | Outputs of steps 02 and 03 |
| 05 | `05_model_selection.py` | Selects the model settings with walk-forward validation on 2022 to 2024. It never uses 2025. | Outputs of steps 02 and 03 |
| 06 | `06_model_training.py` | Trains the RNN and LSTM models on clean and biased series. Scores the 2025 forecasts against the clean case counts. | Outputs of steps 02, 03, and 04 |
| 07 | `07_model_diagnostics.py` | Makes diagnostic figures and a bias magnitude table from the step 06 results. | Outputs of steps 02, 03, and 06 |
| 08 | `08_sier_recalibrated.py` | Fits a seasonal SEIR model with rolling refit. Refits every 4 weeks on the last 26 weeks. Runs two loss arms: rmse and nbnll. (The file name uses the spelling "sier". The model is a SEIR model.) | Outputs of steps 02 and 03 |
| 09 | `09_bias_aware_filter.py` | Corrects the biased series with the operational filter. The filter uses prior mean parameters only. Then trains the RNN and LSTM models and scores the 2025 forecasts. | Outputs of steps 02, 03, and 04 |
| 10 | `10_adaptive_lambda_shrinkage.py` | Sets the shrinkage weight lambda from the prior only, with no clean labels. The rule is lambda = clamp(1 − kappa × CV, 0, 1), with kappa = 1. CV is the coefficient of variation of the correction factor. | Outputs of steps 02, 03, and 04 |
| 11 | `11_figures_tables.py` | Makes manuscript Figures 1, 2, and 4. Figure 1 shows the SEIR fit and forecasts. Figure 2 compares RNN and LSTM under bias. Figure 4 shows the bias propagation tradeoff. | Outputs of steps 06 and 08 |
| 12a | `12_a_panel_generator_kaski_v_01.py` | Makes the weekly KASKI panel v01 from the `data/` folder. The window is 2022-01-01 to 2024-10-21. | `data/` folder |
| 12b | `12_b_panel_generator_kaski_v_02.py` | Makes the weekly KASKI panel v02 from the `data_test_kaski/` folder. The weeks are the same as in v01. | `data_test_kaski/` folder |
| 13 | `13_retrain.py` | Retrains the models for cluster 2 with a hard cutoff at 2024-10-21 (ISO week 2024-W43). Later weeks get forward passes only. Saves the model checkpoints. | Outputs of steps 02, 03, and 04 |
| 14 | `14_c_generalizability_test.py` | Runs the saved models on the KASKI v01 series (under-reported) and the KASKI v02 series (corrected). Scores the forecasts and makes figures and a report. | Outputs of steps 12a, 12b, and 13 |
| 15 | `15_error_metric.py` | Scores the operational filter with NB-NLL. The dispersion comes from the method of moments (default) or a profile fit. | Outputs of steps 02, 03, and 04 |

### Output files

All paths are relative to `v_05/`.

| Step | Output location | Files |
|---|---|---|
| 01 | `outputs/` | `district_covariates.csv`, `ingestion_report.txt` |
| 02 | `outputs/` and `figures/` | `district_clusters.csv`, `bias_parameters.csv`, `cluster_summary.csv`, `cluster_scores.png`, `cluster_3d_scatter.png` |
| 03 | `outputs/` | `panel_cumulative.csv`, `panel_weekly.csv`, `test_panel_file.csv`, `ingestion_report.txt` |
| 04 | `outputs/` | `bias_draw_params.csv`, `biased_B1.csv`, `biased_B2.csv`, `biased_B3.csv`, `biased_combined.csv` |
| 05 | `outputs/05b_model_training/` | `model_selection_walkforward.csv`, `model_selection_summary.csv`, and 4 figures in `figures/` |
| 06 | `outputs/nbnll_model_training/` | `model_results.csv`, `model_predictions.csv`, `bias_propagation_summary.csv` |
| 07 | `outputs/05b_model_training/` | `bias_magnitude_summary.csv` and the diagnostic figures in `figures/` |
| 08 | `outputs/05b_model_training/` | `seir_recalibrated_rmse_*.csv` and `seir_recalibrated_nbnll_*.csv` (parameters, fitted, metrics). Copies of the rmse files are saved as `seir_recalibrated_parameters.csv`, `seir_recalibrated_fitted.csv`, and `seir_recalibrated_metrics.csv`. |
| 09 | `outputs/` | `v05_bias_filter_results.csv`, `v05_bias_filter_summary.csv` |
| 10 | `outputs/` | `v05_lambda_rule.csv`, `v05_lambda_results.csv`, `v05_lambda_rule_summary.csv` |
| 11 | `figures/` | Figures 1, 2, and 4 as PNG and PDF files |
| 12a | `outputs/` | `panel_weekly_kaski_v_01.csv`, `panel_cumulative_kaski_v_01.csv` |
| 12b | `outputs/` | `panel_weekly_kaski_v_02.csv`, `panel_cumulative_kaski_v_02.csv` |
| 13 | `outputs/nbnll_13_retrain_cluster2/` | `retrain_results.csv`, `retrain_predictions.csv`, `retrain_summary.csv`, `manifest.json`, and the checkpoints in `weights_cluster2_retrain/` |
| 14 | `outputs/nbnll_14_generalizability/` | `report_generalizability.md`, `run_manifest.json`, `tables/` (`per_checkpoint_metrics.csv`, `checkpoint_ratios.csv`, `summary_by_model.csv`, `weekly_ensemble.csv`, `predictions_full_sequence.csv`), and `figures/` |
| 15 | `outputs/` | `v05_bias_filter_nll_results.csv`, `v05_bias_filter_nll_summary.csv` |

### Script options

Each option has a default. The defaults are shown in brackets.

| Step | Options |
|---|---|
| 05 | `--mode` (ablation; the other values are full and hidden_only), `--seeds` (3), `--hidden`, `--epochs`, `--lrs`, `--batches`, `--max_combos` |
| 08 | `--losses` (rmse and nbnll), `--clusters` (1 2 3), `--n-restarts-cal` (30), `--n-restarts-rolling` (16), `--seed` (0), `--no-legacy` |
| 09, 10 | `--draws` (100), `--seeds` (2), `--clusters` (1 2 3), `--conditions` (B1 B2 B3 combined) |
| 13 | `--conditions` (clean, B1, B2, B3, combined), `--models` (RNN, LSTM), `--n-draws` (100), `--seeds-clean` (10), `--seeds-biased` (3), `--epochs` (150), `--transform` (log1p) |
| 14 | `--v01`, `--v02`, `--retrain-dir`, `--outdir`, `--alpha-mode` (mom), `--start` (2024-01-01), `--end` (2024-10-25), `--conditions` (clean only), `--models` (RNN, LSTM), `--seeds`, `--max-draws`, `--plot-start` (2023-01-01), `--no-clip` |
| 15 | `--draws` (100), `--seeds` (2), `--clusters` (1 2 3), `--conditions` (B1 B2 B3 combined), `--alpha-mode` (mom; the other value is profile) |

Steps 01 to 04, 06, 07, 11, 12a, and 12b have no options. Their settings are constants in the script.

## Design rules that limit information leakage

- Step 05 uses data from 2022 to 2024 only. It never uses 2025.
- The operational correction in steps 09, 10, and 15 uses prior mean parameters only. It does not use the sampled true parameters in `bias_draw_params.csv`. The clean case counts are used for scoring only.
- The correction runs on the full series first. The code then splits the corrected series into training, validation, and test parts (`correct_full_then_split` in `common.py`).
- For B2 and combined bias, the correction builds each week from observations that come later. The test results for these two types are retrospective. They do not show real-time performance. The code trims the weeks next to the training and validation boundaries. It does not trim the test part.
- The code joins forecasts to the clean case counts by week date, not by array position. A duplicate week causes an error. An empty overlap between forecasts and case counts also causes an error.
- The NB-NLL dispersion is estimated by the method of moments from the training window, not from the forecast period (default setting).
- Steps 09, 10, and 15 use the model settings in `common.py`: 8-week input, 16 hidden units, learning rate 0.001, batch size 16, and a maximum of 80 epochs with early stopping (patience 15). Step 06 uses a maximum of 150 epochs.

## Folder notes

These notes describe how the code reads and writes folders in the current version. They affect the run order.

1. **Case count folders.** Steps 03, 12a, and 12b need the folders `data` and `data_test_kaski`. These folders are not in the repository (see "Input data").
2. **Steps 09, 10, and 15.** These steps read `panel_weekly.csv` from the folder `outputs` at the repository root. Step 03 writes this file to `v_05/outputs`. Before you run steps 09, 10, and 15, copy `panel_weekly.csv` to a folder named `outputs` at the repository root. These steps read all other inputs from `v_05/outputs` and write their results there.
3. **Steps 07 and 11.** These steps read `model_results.csv`, `model_predictions.csv`, and `bias_propagation_summary.csv` from `v_05/outputs/05b_model_training/`. Step 06 writes these files to `v_05/outputs/nbnll_model_training/`. Before you run steps 07 and 11, copy the three files to `v_05/outputs/05b_model_training/`. Step 11 also reads the SEIR files that step 08 writes to this folder.
4. **Step 14.** The default input paths point to the outputs of steps 12a, 12b, and 13 in `v_05/outputs/`. Run steps 12a, 12b, and 13 first.
5. **Shared file name.** Steps 01 and 03 both write `ingestion_report.txt` in `v_05/outputs/`. The file from step 03 replaces the file from step 01.

## Run record

The file `v_05/v_05_ouput_final_run/run_manifest.txt` records the date (2026-09-14), the Python version (3.11.2), the git revision (45fc37e), and the step order of the final run. The file `pipeline_run.log` is the console log of that run. The log in this repository ends at the start of step 06.

Random seeds:

| Step | Seed |
|---|---|
| 02 | 42 (k-means) |
| 04, 06, 09, 10, 15 | 42 (NumPy) |
| 08 | 0 (default of `--seed`) |
| Model training | Each model run has its own seed. The scripts set the number of seeds (see "Script options"). |

## License

This project uses the MIT License. See the file `LICENSE`.
