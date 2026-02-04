================================================================================
ENHANCED MODEL TRAINING - TOTAL & CATEGORY-SPECIFIC
================================================================================

1. LOADING PREPARED DATA...
✓ X_train: (52033, 39)
✓ X_val: (15000, 39)
✓ total_spend: train=52033, val=15000
✓ electronics_spend: train=52033, val=15000
✓ grocery_spend: train=52033, val=15000
✓ sports_spend: train=52033, val=15000

2. TRAINING MODELS FOR ALL TARGETS...

================================================================================
TASK 1: TOTAL 30-DAY SPEND PREDICTION
================================================================================

================================================================================
TRAINING MODELS FOR: TOTAL_SPEND
================================================================================

Target Statistics:
  Training samples: 52033
  Validation samples: 15000
  Training mean: 144959.34 INR
  Validation mean: 144713.36 INR
  Non-zero training: 50114 (96.3%)
  Non-zero validation: 14448 (96.3%)

--------------------------------------------------------------------------------
1. RANDOM FOREST REGRESSOR
--------------------------------------------------------------------------------

Random Forest:
  MAE:  2371.55 INR
  RMSE: 1176.67 INR
  R²:   0.9438
  MAPE: 17.74% (for non-zero targets)

Top 10 Most Important Features:
  total_loyalty_points: 0.1013
  min_transaction_value: 0.0696
  max_transaction_value: 0.0653
  purchase_momentum: 0.0624
  avg_days_between_transactions: 0.0488
  avg_transaction_value: 0.0482
  std_transaction_value: 0.0482
  avg_items_per_transaction: 0.0413
  spend_30d_sports: 0.0390
  weekend_transaction_rate: 0.0387

--------------------------------------------------------------------------------
2. XGBOOST REGRESSOR
--------------------------------------------------------------------------------

XGBoost:
  MAE:  840.52 INR
  RMSE: 707.29 INR
  R²:   0.9639
  MAPE: 9.10% (for non-zero targets)

--------------------------------------------------------------------------------
3. LIGHTGBM REGRESSOR
--------------------------------------------------------------------------------

LightGBM:
  MAE:  2354.33 INR
  RMSE: 1451.78 INR
  R²:   0.9163
  MAPE: 18.69% (for non-zero targets)



🏆 BEST MODEL FOR TOTAL_SPEND: XGBoost
✓ Saved total spend models

================================================================================
TASK: ELECTRONICS SPEND PREDICTION
================================================================================

================================================================================
TRAINING MODELS FOR: ELECTRONICS_SPEND
================================================================================

Target Statistics:
  Training samples: 52033
  Validation samples: 15000
  Training mean: 26303.50 INR
  Validation mean: 26248.18 INR
  Non-zero training: 37151 (71.4%)
  Non-zero validation: 10607 (70.7%)

--------------------------------------------------------------------------------
1. RANDOM FOREST REGRESSOR
--------------------------------------------------------------------------------

Random Forest:
  MAE:  3635.01 INR
  RMSE: 2017.14 INR
  R²:   0.9459
  MAPE: 14.38% (for non-zero targets)

Top 10 Most Important Features:
  total_loyalty_points: 0.0965
  purchase_momentum: 0.0696
  min_transaction_value: 0.0653
  max_transaction_value: 0.0645
  avg_transaction_value: 0.0522
  avg_days_between_transactions: 0.0460
  std_transaction_value: 0.0460
  avg_items_per_transaction: 0.0447
  spend_30d_grocery: 0.0416
  spend_30d_sports: 0.0398

--------------------------------------------------------------------------------
2. XGBOOST REGRESSOR
--------------------------------------------------------------------------------

XGBoost:
  MAE:  2406.77 INR
  RMSE: 1826.10 INR
  R²:   0.9580
  MAPE: 9.77% (for non-zero targets)

--------------------------------------------------------------------------------
3. LIGHTGBM REGRESSOR
--------------------------------------------------------------------------------

LightGBM:
  MAE:  3760.15 INR
  RMSE: 2327.45 INR
  R²:   0.9261
  MAPE: 12.33% (for non-zero targets)

🏆 BEST MODEL FOR ELECTRONICS_SPEND: XGBoost

✓ Saved electronics models

================================================================================
TASK: GROCERY SPEND PREDICTION
================================================================================

================================================================================
TRAINING MODELS FOR: GROCERY_SPEND
================================================================================

Target Statistics:
  Training samples: 52033
  Validation samples: 15000
  Training mean: 34038.10 INR
  Validation mean: 33757.19 INR
  Non-zero training: 40334 (77.5%)
  Non-zero validation: 11705 (78.0%)

--------------------------------------------------------------------------------
1. RANDOM FOREST REGRESSOR
--------------------------------------------------------------------------------

Random Forest:
  MAE:  3591.66 INR
  RMSE: 2656.00 INR
  R²:   0.9346
  MAPE: 16.61% (for non-zero targets)

Top 10 Most Important Features:
  total_loyalty_points: 0.0955
  purchase_momentum: 0.0754
  min_transaction_value: 0.0596
  max_transaction_value: 0.0583
  avg_days_between_transactions: 0.0483
  std_transaction_value: 0.0473
  avg_transaction_value: 0.0455
  spend_30d_sports: 0.0410
  avg_items_per_transaction: 0.0404
  spend_30d_electronics: 0.0392

--------------------------------------------------------------------------------
2. XGBOOST REGRESSOR
--------------------------------------------------------------------------------

XGBoost:
  MAE:  2163.83 INR
  RMSE: 1164.70 INR
  R²:   0.9626
  MAPE: 8.37% (for non-zero targets)

--------------------------------------------------------------------------------
3. LIGHTGBM REGRESSOR
--------------------------------------------------------------------------------

LightGBM:
  MAE:  2730.63 INR
  RMSE: 2817.12 INR
  R²:   0.9253
  MAPE: 15.07% (for non-zero targets)


🏆 BEST MODEL FOR GROCERY_SPEND: XGBoost

✓ Saved grocery models

================================================================================
TASK: SPORTS SPEND PREDICTION
================================================================================

================================================================================
TRAINING MODELS FOR: SPORTS_SPEND
================================================================================

Target Statistics:
  Training samples: 52033
  Validation samples: 15000
  Training mean: 31067.24 INR
  Validation mean: 31121.33 INR
  Non-zero training: 40596 (78.0%)
  Non-zero validation: 11650 (77.7%)

--------------------------------------------------------------------------------
1. RANDOM FOREST REGRESSOR
--------------------------------------------------------------------------------

Random Forest:
  MAE:  2759.66 INR
  RMSE: 1435.03 INR
  R²:   0.9544
  MAPE: 12.32% (for non-zero targets)

Top 10 Most Important Features:
  total_loyalty_points: 0.0960
  max_transaction_value: 0.0682
  min_transaction_value: 0.0656
  purchase_momentum: 0.0652
  avg_days_between_transactions: 0.0506
  avg_transaction_value: 0.0495
  std_transaction_value: 0.0461
  avg_items_per_transaction: 0.0451
  weekend_transaction_rate: 0.0389
  spend_30d_sports: 0.0378

--------------------------------------------------------------------------------
2. XGBOOST REGRESSOR
--------------------------------------------------------------------------------

XGBoost:
  MAE:  1358.74 INR
  RMSE: 690.96 INR
  R²:   0.9741
  MAPE: 6.31% (for non-zero targets)

--------------------------------------------------------------------------------
3. LIGHTGBM REGRESSOR
--------------------------------------------------------------------------------

LightGBM:
  MAE:  3106.95 INR
  RMSE: 2359.85 INR
  R²:   0.9239
  MAPE: 12.42% (for non-zero targets)


🏆 BEST MODEL FOR SPORTS_SPEND: XGBoost

✓ Saved sports models

================================================================================
BEST MODELS BY TARGET
================================================================================

TOTAL SPEND:
  Best Model: XGBoost

ELECTRONICS SPEND:
  Best Model: XGBoost

GROCERY SPEND:
  Best Model: XGBoost

SPORTS SPEND:
  Best Model: XGBoost

================================================================================
4. SAVING MODEL METADATA
================================================================================
✓ Saved: model_metadata_enhanced.json

================================================================================
5. DETAILED PREDICTION ANALYSIS
================================================================================

PREDICTION STATISTICS:

Total Spend:
  Actual mean: 144713.36 INR
  Predicted mean: 151145.92 INR
  MAE: 1328.57 INR
  Median error: 913.19 INR