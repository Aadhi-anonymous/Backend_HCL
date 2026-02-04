# Customer Spend Prediction Service

Complete Flask backend with ML pipeline for predicting customer 30-day spend with category breakdowns.

## 🎯 Features

### API Features
- ✅ **Single Prediction** - Predict spend for one customer
- ✅ **Bulk Prediction** - Upload CSV/Excel with thousands of customers
- ✅ **Multi-Target Predictions** - Total spend + Electronics + Grocery + Sports
- ✅ **Real-time Progress** - Track bulk prediction jobs
- ✅ **Health Monitoring** - Health check endpoints

### ML Features
- ✅ **Rolling Window Dataset** - Multiple observations per customer
- ✅ **45+ Features** - RFM, behavioral, temporal, category-specific
- ✅ **3 ML Models** - Random Forest, XGBoost, LightGBM
- ✅ **Time-Based Validation** - Prevents data leakage
- ✅ **Category-Specific Models** - Separate models for each category

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Copy `.env.example` to `.env` and update:

```bash
cp .env.example .env
```

Required variables:
```env
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
PORT=5000
FLASK_ENV=development
```

### 3. Run Server

```bash
python run.py
```

Server will start on `http://localhost:5000`

### 4. Test API

```bash
# Health check
curl http://localhost:5000/health

# Single prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"customer_id": "1"}'

# Bulk prediction template
curl -o template.csv http://localhost:5000/bulk/template
```

---

## 📡 API Endpoints

### Single Prediction

**POST** `/predict`

Predict 30-day spend for a single customer.

**Request:**
```json
{
  "customer_id": "12345"
}
```

**Response:**
```json
{
  "customer_id": "12345",
  "predicted_30d_spend": {
    "total": 12500.50,
    "electronics": 4500.25,
    "grocery": 6000.15,
    "sports": 2000.10
  },
  "currency": "INR",
  "model_version": "v1.0-enhanced"
}
```

### Bulk Prediction

**POST** `/bulk/predict` - Upload CSV/Excel file
**GET** `/bulk/status/{job_id}` - Check progress
**GET** `/bulk/download/{job_id}` - Download results
**GET** `/bulk/template` - Download CSV template

See **[BULK_PREDICTION_API.md](BULK_PREDICTION_API.md)** for complete documentation.

---

## 🤖 ML Pipeline

### Train Models

```bash
cd ml_pipeline
python run_pipeline.py
```

This will:
1. Create rolling window observations (50K-150K+ examples)
2. Engineer 45+ features
3. Train 12 models (3 algorithms × 4 targets)
4. Select best model for each target
5. Save models and metadata

### Models Trained

| Algorithm | Targets | Speed | Typical Performance |
|-----------|---------|-------|-------------------|
| Random Forest | 4 | Medium | MAE: 2000-3000 |
| **XGBoost** ⭐ | 4 | Fast | MAE: 1500-2500 |
| **LightGBM** ⭐ | 4 | Fastest | MAE: 1500-2500 |

**Total**: 12 models (3 algorithms × 4 targets)

### Prediction Targets

1. **Total Spend** - Overall customer spend
2. **Electronics Spend** - Electronics category
3. **Grocery Spend** - Grocery category
4. **Sports Spend** - Sports category

---

## 📊 Dataset Structure

### Input Tables (pymind_dataset/)

- `customer_details.csv` - Customer information
- `store_sales_header.csv` - Transaction headers
- `store_sales_line_items.csv` - Transaction line items
- `products.csv` - Product catalog
- `stores.csv` - Store information

### Features Engineered (45+)

**Recency (5)**
- days_since_last_transaction
- days_since_first_transaction
- total_transactions
- avg_days_between_transactions
- customer_tenure_days

**Frequency (3)**
- transaction_count_7d, 30d, 90d

**Monetary (8)**
- total_spend_7d, 30d, 90d
- lifetime_spend
- avg/max/min/std_transaction_value
- spend_velocity

**Diversity (3)**
- num_distinct_products, categories, stores

**Category-Specific (9)**
- spend_30d_electronics, grocery, sports
- txn_count_electronics, grocery, sports
- favorite_category_* (one-hot)

**Behavioral (5+)**
- promotion_usage_rate
- purchase_momentum
- weekend_transaction_rate
- avg_items_per_transaction
- etc.

---

## 📁 Project Structure

```
hcl_server/
├── app.py                      # Flask application factory
├── run.py                      # Server entry point
├── config.py                   # Configuration
│
├── routes/                     # API routes
│   ├── health.py              # Health check
│   ├── prediction.py          # Single prediction
│   └── bulk_prediction.py     # Bulk prediction
│
├── services/                   # Business logic
│   └── prediction_service.py  # ML inference
│
├── ml_pipeline/               # ML training pipeline
│   ├── 01_explore_data.py    # Data exploration
│   ├── 02_data_preparation_enhanced.py  # Feature engineering
│   ├── 03_train_models_enhanced.py      # Model training
│   └── run_pipeline.py        # Master script
│
├── models/                    # Trained models (generated)
│   ├── xgboost_total_spend.json
│   ├── xgboost_electronics.json
│   ├── xgboost_grocery.json
│   ├── xgboost_sports.json
│   └── model_metadata_enhanced.json
│
├── pymind_dataset/           # Input data
│   ├── customer_details.csv
│   ├── store_sales_header.csv
│   └── ...
│
└── Documentation/
    ├── README.md                        # This file
    ├── BULK_PREDICTION_API.md          # Bulk API docs
    ├── BULK_PREDICTION_SUMMARY.md      # Quick reference
    └── ENHANCED_ML_PIPELINE_GUIDE.md   # ML pipeline guide
```

---

## 🔌 Integration

### Load Models

```python
import json
import xgboost as xgb

# Load metadata
with open('models/model_metadata_enhanced.json', 'r') as f:
    metadata = json.load(f)

# Load models (cache them)
models = {
    'total': xgb.Booster(),
    'electronics': xgb.Booster(),
    'grocery': xgb.Booster(),
    'sports': xgb.Booster()
}

models['total'].load_model('models/xgboost_total_spend.json')
models['electronics'].load_model('models/xgboost_electronics.json')
models['grocery'].load_model('models/xgboost_grocery.json')
models['sports'].load_model('models/xgboost_sports.json')
```

### Make Predictions

```python
import pandas as pd

def predict_customer_spend(customer_id: str) -> dict:
    # Extract 45 features (implement feature extraction)
    features = extract_features(customer_id)
    
    # Prepare features
    X = pd.DataFrame([features])[metadata['features']]
    dmatrix = xgb.DMatrix(X)
    
    # Predict all targets
    return {
        'customer_id': customer_id,
        'predicted_30d_spend': {
            'total': float(models['total'].predict(dmatrix)[0]),
            'electronics': float(models['electronics'].predict(dmatrix)[0]),
            'grocery': float(models['grocery'].predict(dmatrix)[0]),
            'sports': float(models['sports'].predict(dmatrix)[0])
        },
        'currency': 'INR',
        'model_version': 'v1.0-enhanced'
    }
```

---

## 🧪 Testing

### Run All Tests

```bash
python test.py
```

Tests include:
- ✅ Health check
- ✅ Single prediction (valid/invalid inputs)
- ✅ Bulk prediction upload
- ✅ Job status tracking
- ✅ Results download
- ✅ Template download

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| **README.md** | This file - Overview & quick start |
| **BULK_PREDICTION_API.md** | Complete bulk API documentation with React/Vue examples |
| **BULK_PREDICTION_SUMMARY.md** | Quick reference for bulk predictions |
| **ENHANCED_ML_PIPELINE_GUIDE.md** | ML pipeline details, features, and approach |
| **ml_pipeline/README.md** | ML pipeline step-by-step guide |
| **ml_pipeline/QUICK_REFERENCE.md** | ML quick reference |

---

## 🎯 Business Use Cases

### 1. Revenue Forecasting
Upload all customers → Get total predicted spend → Plan budget

### 2. Targeted Marketing
- High electronics spend → Send electronics promotions
- Low grocery spend → Cross-sell grocery deals
- Multi-category shoppers → VIP treatment

### 3. Inventory Planning
Predict category demand → Optimize stock levels by category

### 4. Customer Segmentation
- High-value customers (total > threshold)
- Category-loyal customers (>70% in one category)
- Cross-sell opportunities (high in one, low in others)

---

## 🔐 Security Considerations

### Current State
- ✅ CORS configured
- ✅ Input validation
- ✅ Error handling
- ✅ Secure file handling

### Before Production
- [ ] Add authentication (JWT/API keys)
- [ ] Implement rate limiting
- [ ] Restrict CORS origins
- [ ] Add request logging
- [ ] Set up monitoring

---

## ⚡ Performance

### Current Implementation
- Synchronous processing
- Works for <1,000 customers in bulk
- Models cached in memory

### Production Recommendations
- Use Celery + Redis for async processing
- Scale horizontally (multiple workers)
- Cloud storage for uploaded files (S3/GCS)
- Model serving with dedicated service

---

## 🚢 Deployment

### Local Development

```bash
python run.py
```

### Production (Gunicorn)

```bash
gunicorn -w 4 -b 0.0.0.0:5000 "app:create_app()"
```

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:create_app()"]
```

Build and run:
```bash
docker build -t customer-spend-api .
docker run -p 5000:5000 --env-file .env customer-spend-api
```

---

## 📊 Expected Performance

### Model Metrics
- **MAE**: 1,500-2,500 INR (excellent)
- **RMSE**: 2,500-4,000 INR
- **R²**: 0.70-0.85

### API Performance
- Single prediction: <50ms
- Bulk prediction (1,000 customers): ~2-5 minutes

---

## 🆘 Troubleshooting

### Server won't start
```bash
# Check configuration
python -c "from config import Config; Config.validate()"

# Verify .env file exists
cat .env
```

### Import errors
```bash
pip install -r requirements.txt
```

### Model training fails
```bash
cd ml_pipeline
python 01_explore_data.py  # Check data
python 02_data_preparation_enhanced.py  # Check features
```

### Bulk upload fails
- Check file size (<10MB)
- Verify file format (CSV, XLS, XLSX)
- Ensure first column has customer IDs

---

## 📞 Support

For issues or questions:
1. Check documentation (README.md, BULK_PREDICTION_API.md)
2. Review example code in test.py
3. Check server logs for errors

---

## 🎓 Key Features Summary

✅ **Complete API** - Single & bulk predictions
✅ **Multi-Target Models** - Total + 3 category predictions
✅ **Production-Ready** - Error handling, validation, documentation
✅ **Scalable Architecture** - Easy to add async processing
✅ **Comprehensive Docs** - API docs, examples, guides
✅ **ML Pipeline** - End-to-end training with 45+ features
✅ **Time-Series Approach** - Rolling windows for robust models
✅ **Easy Integration** - React/Vue examples provided

---

## 📝 License

[Your License Here]

---

**Version**: 1.0
**Last Updated**: February 2026
