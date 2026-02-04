# ML Pipeline - Quick Reference

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run complete pipeline
cd ml_pipeline
python run_pipeline.py

# Or run steps individually
python 02_data_preparation.py
python 03_train_models.py
```

## 📊 Problem Definition

- **Task**: Predict 30-day customer spend
- **Type**: Regression
- **Cutoff Date**: 2026-01-15
- **Metric**: Mean Absolute Error (MAE)

## 📁 Dataset Files

| File | Description | Key |
|------|-------------|-----|
| customer_details.csv | Customer info | customer_id |
| store_sales_header.csv | Transactions | transaction_id |
| store_sales_line_items.csv | Line items | line_item_id |
| products.csv | Products | product_id |
| stores.csv | Stores | store_id |

## 🎯 Features Used (30-35 total)

### Recency (5)
- days_since_last_transaction
- days_since_first_transaction
- total_transactions
- avg_days_between_transactions
- customer_tenure_days

### Frequency (3)
- transaction_count_7d
- transaction_count_30d
- transaction_count_90d

### Monetary (8)
- total_spend_7d, 30d, 90d
- lifetime_spend
- avg/max/min/std_transaction_value

### Diversity (3)
- num_distinct_products
- num_distinct_categories
- num_distinct_stores

### Behavioral (5)
- total_items_purchased
- avg_items_per_transaction
- promotion_usage_rate
- purchase_momentum
- total_loyalty_points

### Categorical (2 + one-hot)
- loyalty_status_encoded
- segment_id_encoded
- category_* (one-hot)

## ❌ Features Excluded

- customer_id (identifier)
- loyalty_status (encoded)
- segment_id (encoded)
- favorite_category (one-hot)
- *_date columns (converted to days)
- target_30d_spend (target variable)

## 🤖 Models Trained

1. **Random Forest** - Baseline
2. **XGBoost** ⭐ - Likely best
3. **LightGBM** ⭐ - Likely best
4. **Neural Network** - Comparison

## 📈 Evaluation Metrics

- **Primary**: MAE (Mean Absolute Error)
- **Secondary**: RMSE, R²

## 🏆 Expected Best Model

**XGBoost or LightGBM**

Why?
- ✅ Best for tabular data
- ✅ Fast inference
- ✅ Handles non-linearity
- ✅ Feature importance

## 📦 Output Files

### Models
- random_forest_model.pkl
- xgboost_model.json ⭐
- lightgbm_model.pkl ⭐
- neural_network_model.keras
- scaler.pkl
- best_model_metadata.json

### Data
- X_train.csv, X_val.csv
- y_train.csv, y_val.csv

### Analysis
- model_comparison.csv
- prediction_analysis.csv
- features_used.txt
- features_excluded.txt

## 🔌 Integration

```python
# Load best model
with open('models/best_model_metadata.json') as f:
    metadata = json.load(f)

# Load model
import xgboost as xgb
model = xgb.Booster()
model.load_model(f"models/{metadata['model_file']}")

# Extract features (must match training!)
features = extract_features(customer_id)

# Predict
prediction = model.predict(xgb.DMatrix([features]))[0]
```

## ✅ Data Leakage Prevention

- ✅ Features: Only data BEFORE cutoff
- ✅ Target: Only data AFTER cutoff
- ✅ No future information in features
- ✅ Explicit exclusion list

## 📊 Expected Performance

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| XGBoost | 1800-2800 | 2800-4500 | 0.65-0.80 |
| LightGBM | 1800-2800 | 2800-4500 | 0.65-0.80 |
| Random Forest | 2000-3000 | 3000-5000 | 0.60-0.75 |
| Neural Net | 2000-3200 | 3000-5000 | 0.60-0.75 |

## 🎓 Academic Checklist

- [x] Dataset understanding
- [x] Join logic documented
- [x] RFM features
- [x] 30+ features engineered
- [x] Features excluded with reasons
- [x] Target construction (no leakage)
- [x] Train/val split
- [x] 4 models trained (same features)
- [x] MAE/RMSE/R² evaluation
- [x] Model comparison table
- [x] Best model selection
- [x] Production-ready output

## 🔧 Troubleshooting

**Import errors?**
```bash
pip install pandas numpy scikit-learn xgboost lightgbm tensorflow
```

**Out of memory?**
- Reduce n_estimators
- Use smaller neural network
- Sample data

**Poor performance?**
- Check feature engineering
- Verify no data leakage
- Tune hyperparameters

## 📚 Documentation

- **README.md** - Full pipeline docs
- **ML_PIPELINE_DOCUMENTATION.md** - Complete technical spec
- **QUICK_REFERENCE.md** - This file

## 🎯 Next Steps

1. Run pipeline: `python run_pipeline.py`
2. Check results: `cat model_comparison.csv`
3. Review best model: `cat ../models/best_model_metadata.json`
4. Integrate: Follow `integration_instructions.txt`
5. Test: Make predictions with real customers

---

**Quick Links:**
- [Full Documentation](../ML_PIPELINE_DOCUMENTATION.md)
- [Pipeline README](README.md)
- [Integration Example](../INTEGRATION_EXAMPLE.md)
