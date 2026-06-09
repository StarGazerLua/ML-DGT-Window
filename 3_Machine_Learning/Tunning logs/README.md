# Hyperparameter Tuning Logs

This directory contains the training logs during the Random Forest hyperparameter sweep. All configurations share `random_state=40` and the same zone-stratified LOCO evaluation subsets (10 cities for Model 1, 100 cities for Model 2).

## Directory layout

```
logs/
├── model1/<param_tag>.log         # Logs for Model 1 (10-city, 11-feature setup)
└── model2_all/<param_tag>.log     # Logs for Model 2 (500-city, all 11 features)
```

Each `.log` file contains: the hyperparameter banner, the held-out city list, the Parallel-dispatch summary, per-city RMSE / MAE / MAPE / R², and the aggregate (mean / std) across cities.

## Experiment configurations

| param_tag | n_estimators | max_depth | min_samples_split | min_samples_leaf |
|---|---|---|---|---|
| ne50_d8_split2_leaf1_rs40    | 50  | 8  | 2  | 1 |
| ne100_d8_split2_leaf1_rs40   | 100 | 8  | 2  | 1 |
| ne150_d8_split4_leaf2_rs40   | 150 | 8  | 4  | 2 |
| ne150_d10_split6_leaf3_rs40  | 150 | 10 | 6  | 3 |
| ne150_d14_split10_leaf5_rs40 | 150 | 14 | 10 | 5 |
| ne300_d10_split6_leaf3_rs40  | 300 | 10 | 6  | 3 |

Design rationale: the `n_estimators` axis covers 50–300, while `max_depth / min_samples_split / min_samples_leaf` are increased together along a diagonal to probe the joint effect of tree depth and regularization strength.

## Final results (sorted by LOCO MAPE_mean ascending)

### Model 1 (10 zone-stratified cities)

| param_tag | MAE_mean | RMSE_mean | **MAPE_mean** |
|---|---|---|---|
| ne150_d10_split6_leaf3_rs40   | 61.35 | 65.07 | **6.07%** |
| ne150_d14_split10_leaf5_rs40  | 63.50 | 66.79 | 6.23% |
| ne300_d10_split6_leaf3_rs40   | 65.30 | 69.03 | 6.47% |
| ne50_d8_split2_leaf1_rs40     | 64.09 | 69.17 | 6.52% |
| ne100_d8_split2_leaf1_rs40    | 64.19 | 69.08 | 6.54% |
| ne150_d8_split4_leaf2_rs40    | 64.31 | 69.08 | 6.59% |

### Model 2 / model2_all (100 zone-stratified cities)

| param_tag | MAE_mean | RMSE_mean | **MAPE_mean** |
|---|---|---|---|
| ne150_d14_split10_leaf5_rs40  | 58.04 | 60.41 | **5.75%** |
| ne300_d10_split6_leaf3_rs40   | 58.53 | 63.40 | 5.94% |
| ne150_d10_split6_leaf3_rs40   | 59.51 | 64.34 | 6.01% |
| ne150_d8_split4_leaf2_rs40    | 66.85 | 75.09 | 7.20% |
| ne50_d8_split2_leaf1_rs40     | 67.38 | 75.67 | 7.20% |
| ne100_d8_split2_leaf1_rs40    | 67.26 | 75.51 | 7.23% |

## Final choice

We adopt **`results_rf_ne150_d10_split6_leaf3_rs40`** (`n_estimators=150, max_depth=10, min_samples_split=6, min_samples_leaf=3, random_state=40`) as the final configuration for both models. Reasons:

- On Model 1 it achieves MAPE = 6.07 %, the lowest across all sweep configurations.
- On Model 2 it reaches MAPE = 6.01 %, only ~0.3 pp behind the per-model best (`ne150_d14_split10_leaf5_rs40`, 5.75 %).
