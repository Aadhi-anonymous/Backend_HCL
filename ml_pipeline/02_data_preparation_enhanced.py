"""
Enhanced Data Preparation with Rolling Window Approach
Creates a time-series dataset where each customer has multiple observations
Predicts both total spend AND category-specific spend for next 30 days
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = "../pymind_dataset"
LOOKBACK_DAYS = 90  # Use 90 days of history to predict next 30 days
PREDICTION_WINDOW_DAYS = 30
MIN_HISTORY_DAYS = 30  # Minimum history required for a prediction

print("="*80)
print("ENHANCED DATA PREPARATION - ROLLING WINDOW APPROACH")
print("="*80)

# ============================================================================
# 1. LOAD ALL DATASETS
# ============================================================================
print("\n1. LOADING DATASETS...")

customers = pd.read_csv(f"{DATA_DIR}/customer_details.csv")
stores = pd.read_csv(f"{DATA_DIR}/stores.csv")
products = pd.read_csv(f"{DATA_DIR}/products.csv")
sales_header = pd.read_csv(f"{DATA_DIR}/store_sales_header.csv")
sales_line_items = pd.read_csv(f"{DATA_DIR}/store_sales_line_items.csv")

print(f"✓ Customers: {customers.shape}")
print(f"✓ Stores: {stores.shape}")
print(f"✓ Products: {products.shape}")
print(f"✓ Sales Header: {sales_header.shape}")
print(f"✓ Sales Line Items: {sales_line_items.shape}")

# ============================================================================
# 2. JOIN TABLES TO CREATE COMPLETE TRANSACTION DATASET
# ============================================================================
print("\n2. CREATING COMPLETE TRANSACTION DATASET...")

# Join line items with sales header
transactions = sales_line_items.merge(
    sales_header[['transaction_id', 'customer_id', 'store_id', 'transaction_date']],
    on='transaction_id',
    how='left'
)

# Join with products to get category
transactions = transactions.merge(
    products[['product_id', 'product_category', 'unit_price']],
    on='product_id',
    how='left'
)

# Convert transaction_date to datetime
transactions['transaction_date'] = pd.to_datetime(transactions['transaction_date'])
transactions = transactions.sort_values('transaction_date').reset_index(drop=True)

print(f"✓ Complete transactions: {transactions.shape}")
print(f"  Date range: {transactions['transaction_date'].min()} to {transactions['transaction_date'].max()}")
print(f"  Unique customers: {transactions['customer_id'].nunique()}")
print(f"  Unique products: {transactions['product_id'].nunique()}")
print(f"  Product categories: {transactions['product_category'].unique()}")

# ============================================================================
# 3. CREATE ROLLING WINDOW OBSERVATIONS
# ============================================================================
print("\n3. CREATING ROLLING WINDOW OBSERVATIONS...")
print(f"   Strategy: For each customer, create multiple observation points")
print(f"   - Use {LOOKBACK_DAYS} days of history to predict next {PREDICTION_WINDOW_DAYS} days")
print(f"   - Create observations at weekly intervals")

# Get date range
min_date = transactions['transaction_date'].min()
max_date = transactions['transaction_date'].max()

# Calculate total days
total_days = (max_date - min_date).days
print(f"   Total data span: {total_days} days ({min_date.date()} to {max_date.date()})")

# Create observation cutoff dates (weekly intervals)
# Start from min_date + MIN_HISTORY_DAYS, end at max_date - PREDICTION_WINDOW_DAYS
observation_start = min_date + timedelta(days=MIN_HISTORY_DAYS)
observation_end = max_date - timedelta(days=PREDICTION_WINDOW_DAYS)

cutoff_dates = pd.date_range(
    start=observation_start,
    end=observation_end,
    freq='7D'  # Weekly observations
)

print(f"   Creating {len(cutoff_dates)} observation cutoff dates")
print(f"   First cutoff: {cutoff_dates[0].date()}")
print(f"   Last cutoff: {cutoff_dates[-1].date()}")

# ============================================================================
# 4. FEATURE ENGINEERING FUNCTION
# ============================================================================

def extract_features_for_cutoff(customer_id, cutoff_date, transaction_df):
    """
    Extract features for a customer as of a specific cutoff date.
    Uses only data BEFORE the cutoff date.
    """
    # Filter transactions before cutoff for this customer
    customer_txns = transaction_df[
        (transaction_df['customer_id'] == customer_id) &
        (transaction_df['transaction_date'] < cutoff_date)
    ].copy()
    
    if len(customer_txns) == 0:
        return None  # No history
    
    # Time windows
    date_7d = cutoff_date - timedelta(days=7)
    date_30d = cutoff_date - timedelta(days=30)
    date_90d = cutoff_date - timedelta(days=90)
    
    txns_7d = customer_txns[customer_txns['transaction_date'] >= date_7d]
    txns_30d = customer_txns[customer_txns['transaction_date'] >= date_30d]
    txns_90d = customer_txns[customer_txns['transaction_date'] >= date_90d]
    
    features = {}
    
    # ========== RECENCY FEATURES ==========
    features['days_since_last_transaction'] = (cutoff_date - customer_txns['transaction_date'].max()).days
    features['days_since_first_transaction'] = (cutoff_date - customer_txns['transaction_date'].min()).days
    
    # ========== FREQUENCY FEATURES ==========
    features['total_transactions'] = len(customer_txns['transaction_id'].unique())
    features['transaction_count_7d'] = len(txns_7d['transaction_id'].unique())
    features['transaction_count_30d'] = len(txns_30d['transaction_id'].unique())
    features['transaction_count_90d'] = len(txns_90d['transaction_id'].unique())
    
    # Average days between transactions
    unique_dates = customer_txns['transaction_date'].unique()
    if len(unique_dates) > 1:
        features['avg_days_between_transactions'] = np.mean(np.diff(sorted(unique_dates))) / np.timedelta64(1, 'D')
    else:
        features['avg_days_between_transactions'] = 0
    
    # ========== MONETARY FEATURES ==========
    features['total_spend_7d'] = txns_7d['line_item_amount'].sum()
    features['total_spend_30d'] = txns_30d['line_item_amount'].sum()
    features['total_spend_90d'] = txns_90d['line_item_amount'].sum()
    features['lifetime_spend'] = customer_txns['line_item_amount'].sum()
    
    # Transaction-level statistics
    txn_totals = customer_txns.groupby('transaction_id')['line_item_amount'].sum()
    features['avg_transaction_value'] = txn_totals.mean()
    features['max_transaction_value'] = txn_totals.max()
    features['min_transaction_value'] = txn_totals.min()
    features['std_transaction_value'] = txn_totals.std() if len(txn_totals) > 1 else 0
    
    # ========== DIVERSITY FEATURES ==========
    features['num_distinct_products'] = customer_txns['product_id'].nunique()
    features['num_distinct_categories'] = customer_txns['product_category'].nunique()
    features['num_distinct_stores'] = customer_txns['store_id'].nunique()
    
    # ========== ITEM/QUANTITY FEATURES ==========
    features['total_items_purchased'] = customer_txns['quantity'].sum()
    features['avg_items_per_transaction'] = customer_txns.groupby('transaction_id')['quantity'].sum().mean()
    features['max_items_in_transaction'] = customer_txns.groupby('transaction_id')['quantity'].sum().max()
    
    # ========== PROMOTION FEATURES ==========
    features['promotion_usage_count'] = customer_txns['promotion_id'].notna().sum()
    features['promotion_usage_rate'] = customer_txns['promotion_id'].notna().mean()
    
    # ========== CATEGORY-SPECIFIC FEATURES ==========
    # Spend by category in last 30 days
    category_spend_30d = txns_30d.groupby('product_category')['line_item_amount'].sum()
    for category in ['Electronics', 'Grocery', 'Sports']:
        features[f'spend_30d_{category.lower()}'] = category_spend_30d.get(category, 0)
    
    # Transaction count by category
    category_txns = customer_txns.groupby('product_category')['transaction_id'].nunique()
    for category in ['Electronics', 'Grocery', 'Sports']:
        features[f'txn_count_{category.lower()}'] = category_txns.get(category, 0)
    
    # Most purchased category
    top_category = customer_txns.groupby('product_category')['line_item_amount'].sum().idxmax()
    for category in ['Electronics', 'Grocery', 'Sports']:
        features[f'favorite_category_{category.lower()}'] = 1 if top_category == category else 0
    
    # ========== TREND/MOMENTUM FEATURES ==========
    # Compare recent vs older spend
    if len(customer_txns) >= 2:
        mid_point = len(customer_txns) // 2
        recent_spend = customer_txns.iloc[mid_point:]['line_item_amount'].mean()
        older_spend = customer_txns.iloc[:mid_point]['line_item_amount'].mean()
        if older_spend > 0:
            features['purchase_momentum'] = (recent_spend - older_spend) / older_spend
        else:
            features['purchase_momentum'] = 0
    else:
        features['purchase_momentum'] = 0
    
    # Spend velocity (spend per day)
    total_days = features['days_since_first_transaction']
    features['spend_velocity'] = features['lifetime_spend'] / total_days if total_days > 0 else 0
    
    # ========== TIME-BASED FEATURES ==========
    # Day of week preference
    customer_txns['day_of_week'] = customer_txns['transaction_date'].dt.dayofweek
    dow_counts = customer_txns['day_of_week'].value_counts()
    features['most_common_day_of_week'] = dow_counts.index[0] if len(dow_counts) > 0 else 0
    
    # Weekend vs weekday preference
    customer_txns['is_weekend'] = customer_txns['day_of_week'].isin([5, 6]).astype(int)
    features['weekend_transaction_rate'] = customer_txns['is_weekend'].mean()
    
    return features

def calculate_target_for_cutoff(customer_id, cutoff_date, transaction_df):
    """
    Calculate target variables for a customer after the cutoff date.
    Uses data AFTER the cutoff date for the prediction window.
    """
    target_end_date = cutoff_date + timedelta(days=PREDICTION_WINDOW_DAYS)
    
    # Filter transactions in the prediction window
    future_txns = transaction_df[
        (transaction_df['customer_id'] == customer_id) &
        (transaction_df['transaction_date'] >= cutoff_date) &
        (transaction_df['transaction_date'] < target_end_date)
    ].copy()
    
    targets = {}
    
    # Total spend in next 30 days
    targets['target_30d_total_spend'] = future_txns['line_item_amount'].sum()
    
    # Category-specific spend
    category_spend = future_txns.groupby('product_category')['line_item_amount'].sum()
    for category in ['Electronics', 'Grocery', 'Sports']:
        targets[f'target_30d_{category.lower()}_spend'] = category_spend.get(category, 0)
    
    # Number of transactions
    targets['target_30d_transaction_count'] = future_txns['transaction_id'].nunique()
    
    return targets

# ============================================================================
# 5. CREATE DATASET WITH ALL OBSERVATIONS
# ============================================================================
print("\n4. BUILDING DATASET WITH ROLLING WINDOWS...")

all_data = []
customers_list = transactions['customer_id'].unique()

total_observations = len(customers_list) * len(cutoff_dates)
print(f"   Maximum possible observations: {total_observations}")

processed = 0
for customer_id in customers_list:
    for cutoff_date in cutoff_dates:
        # Extract features
        features = extract_features_for_cutoff(customer_id, cutoff_date, transactions)
        
        if features is None:
            continue  # Not enough history
        
        # Calculate targets
        targets = calculate_target_for_cutoff(customer_id, cutoff_date, transactions)
        
        # Combine
        observation = {
            'customer_id': customer_id,
            'cutoff_date': cutoff_date,
            **features,
            **targets
        }
        
        all_data.append(observation)
        processed += 1
    
    # Progress indicator
    if (customers_list.tolist().index(customer_id) + 1) % 100 == 0:
        print(f"   Processed {customers_list.tolist().index(customer_id) + 1}/{len(customers_list)} customers...")

print(f"\n✓ Total observations created: {len(all_data)}")

# Convert to DataFrame
df = pd.DataFrame(all_data)

print(f"\n✓ Final dataset shape: {df.shape}")
print(f"  Rows (observations): {df.shape[0]}")
print(f"  Columns (features + targets): {df.shape[1]}")

# ============================================================================
# 6. ADD CUSTOMER METADATA
# ============================================================================
print("\n5. ADDING CUSTOMER METADATA...")

# Merge with customer details
customers_clean = customers[['customer_id', 'loyalty_status', 'total_loyalty_points', 'segment_id']].copy()

df = df.merge(customers_clean, on='customer_id', how='left')

# Encode categorical variables
df['loyalty_status_encoded'] = df['loyalty_status'].map({
    'Bronze': 1, 'Silver': 2, 'Gold': 3, 'Platinum': 4
})
df['segment_id_encoded'] = df['segment_id'].map({
    'LR': 1, 'AR': 2, 'HR': 3
})

print(f"✓ Added customer metadata")
print(f"  Final shape: {df.shape}")

# ============================================================================
# 7. DATA QUALITY CHECKS
# ============================================================================
print("\n6. DATA QUALITY CHECKS...")

print(f"\nTarget Variable Statistics:")
print(df['target_30d_total_spend'].describe())

print(f"\nTarget Distribution:")
print(f"  Observations with target > 0: {(df['target_30d_total_spend'] > 0).sum()} ({(df['target_30d_total_spend'] > 0).mean()*100:.1f}%)")
print(f"  Observations with target = 0: {(df['target_30d_total_spend'] == 0).sum()} ({(df['target_30d_total_spend'] == 0).mean()*100:.1f}%)")

print(f"\nCategory-Specific Targets:")
for category in ['electronics', 'grocery', 'sports']:
    col = f'target_30d_{category}_spend'
    active = (df[col] > 0).sum()
    print(f"  {category.title()}: {active} active ({active/len(df)*100:.1f}%)")

# Missing values
missing = df.isnull().sum()
if missing.sum() > 0:
    print(f"\nMissing Values:")
    print(missing[missing > 0])
    
    # Fill missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    print(f"✓ Filled missing values with 0")

# ============================================================================
# 8. FEATURE SELECTION
# ============================================================================
print("\n7. DEFINING FEATURES FOR TRAINING...")

# Exclude columns
exclude_columns = [
    'customer_id',          # Identifier
    'cutoff_date',          # Date (already used for feature calculation)
    'loyalty_status',       # Encoded
    'segment_id',           # Encoded
    'target_30d_total_spend',  # Target
    'target_30d_electronics_spend',  # Target
    'target_30d_grocery_spend',  # Target
    'target_30d_sports_spend',  # Target
    'target_30d_transaction_count'  # Target
]

# Feature columns
feature_columns = [col for col in df.columns if col not in exclude_columns]

print(f"\n✓ Features for training: {len(feature_columns)}")
print("\nFeature Categories:")

recency_features = [col for col in feature_columns if 'days_since' in col or 'days_between' in col]
frequency_features = [col for col in feature_columns if 'count' in col and 'target' not in col]
monetary_features = [col for col in feature_columns if 'spend' in col and 'target' not in col]
diversity_features = [col for col in feature_columns if 'distinct' in col or 'num_' in col]
category_features = [col for col in feature_columns if 'category' in col or any(cat in col for cat in ['electronics', 'grocery', 'sports'])]
behavioral_features = [col for col in feature_columns if any(x in col for x in ['items', 'promotion', 'momentum', 'velocity', 'weekend', 'day_of'])]

print(f"  Recency: {len(recency_features)}")
print(f"  Frequency: {len(frequency_features)}")
print(f"  Monetary: {len(monetary_features)}")
print(f"  Diversity: {len(diversity_features)}")
print(f"  Category-specific: {len(category_features)}")
print(f"  Behavioral: {len(behavioral_features)}")

# ============================================================================
# 9. TRAIN/VALIDATION SPLIT (TIME-BASED)
# ============================================================================
print("\n8. CREATING TIME-BASED TRAIN/VALIDATION SPLIT...")

# Sort by cutoff date
df = df.sort_values('cutoff_date').reset_index(drop=True)

# Use last 20% of time period for validation
split_date = df['cutoff_date'].quantile(0.8)
print(f"   Split date: {split_date.date()}")

train_df = df[df['cutoff_date'] < split_date].copy()
val_df = df[df['cutoff_date'] >= split_date].copy()

print(f"   Training set: {len(train_df)} observations ({len(train_df)/len(df)*100:.1f}%)")
print(f"   Validation set: {len(val_df)} observations ({len(val_df)/len(df)*100:.1f}%)")

# ============================================================================
# 10. ADD POLYNOMIAL AND INTERACTION FEATURES (FOR DEMO OPTIMIZATION)
# ============================================================================
print("\n9A. CREATING POLYNOMIAL AND INTERACTION FEATURES (DEMO OPTIMIZATION)...")
print("   ⚠️  Adding higher-order features for maximum R² score")

# Select key features for polynomial expansion (avoid explosion)
key_features = [
    'total_spend_7d', 'total_spend_30d', 'total_spend_90d',
    'transaction_count_7d', 'transaction_count_30d', 'transaction_count_90d',
    'avg_transaction_value', 'lifetime_spend',
    'days_since_last_transaction', 'days_since_first_transaction',
    'spend_velocity', 'purchase_momentum',
    'spend_30d_electronics', 'spend_30d_grocery', 'spend_30d_sports'
]

# Create squared features
print("   Adding squared features...")
for feat in key_features:
    if feat in train_df.columns:
        train_df[f'{feat}_squared'] = train_df[feat] ** 2
        val_df[f'{feat}_squared'] = val_df[feat] ** 2

# Create cubic features for top monetary features
print("   Adding cubic features for top monetary features...")
monetary_top = ['total_spend_30d', 'lifetime_spend', 'avg_transaction_value']
for feat in monetary_top:
    if feat in train_df.columns:
        train_df[f'{feat}_cubed'] = train_df[feat] ** 3
        val_df[f'{feat}_cubed'] = val_df[feat] ** 3

# Create interaction features
print("   Adding interaction features...")
interactions = [
    ('total_spend_30d', 'transaction_count_30d'),  # Spend * frequency
    ('avg_transaction_value', 'transaction_count_30d'),  # Value * frequency
    ('total_spend_90d', 'days_since_last_transaction'),  # Spend * recency
    ('spend_30d_electronics', 'spend_30d_grocery'),  # Category interactions
    ('spend_30d_electronics', 'spend_30d_sports'),
    ('spend_30d_grocery', 'spend_30d_sports'),
    ('lifetime_spend', 'days_since_first_transaction'),  # Lifetime * tenure
    ('spend_velocity', 'purchase_momentum'),  # Velocity * momentum
]

for feat1, feat2 in interactions:
    if feat1 in train_df.columns and feat2 in train_df.columns:
        train_df[f'{feat1}_X_{feat2}'] = train_df[feat1] * train_df[feat2]
        val_df[f'{feat1}_X_{feat2}'] = val_df[feat1] * val_df[feat2]

# Create ratio features
print("   Adding ratio features...")
# Spend ratios
if 'total_spend_30d' in train_df.columns and 'total_spend_90d' in train_df.columns:
    train_df['spend_ratio_30d_90d'] = train_df['total_spend_30d'] / (train_df['total_spend_90d'] + 1)
    val_df['spend_ratio_30d_90d'] = val_df['total_spend_30d'] / (val_df['total_spend_90d'] + 1)

if 'total_spend_7d' in train_df.columns and 'total_spend_30d' in train_df.columns:
    train_df['spend_ratio_7d_30d'] = train_df['total_spend_7d'] / (train_df['total_spend_30d'] + 1)
    val_df['spend_ratio_7d_30d'] = val_df['total_spend_7d'] / (val_df['total_spend_30d'] + 1)

# Category spend percentages
for category in ['electronics', 'grocery', 'sports']:
    cat_col = f'spend_30d_{category}'
    if cat_col in train_df.columns and 'total_spend_30d' in train_df.columns:
        train_df[f'pct_{category}_of_total'] = train_df[cat_col] / (train_df['total_spend_30d'] + 1)
        val_df[f'pct_{category}_of_total'] = val_df[cat_col] / (val_df['total_spend_30d'] + 1)

# Log transformations for skewed features
print("   Adding log-transformed features...")
log_features = ['lifetime_spend', 'total_spend_30d', 'total_spend_90d', 'avg_transaction_value']
for feat in log_features:
    if feat in train_df.columns:
        train_df[f'{feat}_log'] = np.log1p(train_df[feat])  # log(1+x) to handle zeros
        val_df[f'{feat}_log'] = np.log1p(val_df[feat])

# Replace inf and -inf values
train_df = train_df.replace([np.inf, -np.inf], 0)
val_df = val_df.replace([np.inf, -np.inf], 0)

# Fill any new NaN values
train_df = train_df.fillna(0)
val_df = val_df.fillna(0)

print(f"✓ Added polynomial and interaction features")
print(f"  New training shape: {train_df.shape}")

# Update feature columns to include new features
exclude_columns_updated = [
    'customer_id', 'cutoff_date', 'loyalty_status', 'segment_id',
    'target_30d_total_spend', 'target_30d_electronics_spend',
    'target_30d_grocery_spend', 'target_30d_sports_spend',
    'target_30d_transaction_count'
]
feature_columns = [col for col in train_df.columns if col not in exclude_columns_updated]

print(f"  Total features now: {len(feature_columns)}")

# ============================================================================
# 10B. PREPARE FINAL DATASETS
# ============================================================================
print("\n9B. PREPARING FINAL DATASETS...")

# Total spend prediction
X_train_total = train_df[feature_columns]
y_train_total = train_df['target_30d_total_spend']

X_val_total = val_df[feature_columns]
y_val_total = val_df['target_30d_total_spend']

# Category-specific predictions
target_categories = {
    'electronics': 'target_30d_electronics_spend',
    'grocery': 'target_30d_grocery_spend',
    'sports': 'target_30d_sports_spend'
}

# Save all datasets
print("\n10. SAVING DATASETS...")

# Total spend
X_train_total.to_csv('X_train_total_spend.csv', index=False)
y_train_total.to_csv('y_train_total_spend.csv', index=False)
X_val_total.to_csv('X_val_total_spend.csv', index=False)
y_val_total.to_csv('y_val_total_spend.csv', index=False)

print(f"✓ Saved total spend datasets")

# Category-specific
for category, target_col in target_categories.items():
    y_train_cat = train_df[target_col]
    y_val_cat = val_df[target_col]
    
    y_train_cat.to_csv(f'y_train_{category}_spend.csv', index=False)
    y_val_cat.to_csv(f'y_val_{category}_spend.csv', index=False)
    
    print(f"✓ Saved {category} spend targets")

# Save feature names
with open('features_used.txt', 'w') as f:
    f.write("FEATURES USED FOR TRAINING\n")
    f.write("="*70 + "\n\n")
    
    f.write(f"TOTAL FEATURES: {len(feature_columns)}\n\n")
    
    f.write(f"RECENCY FEATURES ({len(recency_features)}):\n")
    for feat in recency_features:
        f.write(f"  - {feat}\n")
    
    f.write(f"\nFREQUENCY FEATURES ({len(frequency_features)}):\n")
    for feat in frequency_features:
        f.write(f"  - {feat}\n")
    
    f.write(f"\nMONETARY FEATURES ({len(monetary_features)}):\n")
    for feat in monetary_features:
        f.write(f"  - {feat}\n")
    
    f.write(f"\nDIVERSITY FEATURES ({len(diversity_features)}):\n")
    for feat in diversity_features:
        f.write(f"  - {feat}\n")
    
    f.write(f"\nCATEGORY-SPECIFIC FEATURES ({len(category_features)}):\n")
    for feat in category_features:
        f.write(f"  - {feat}\n")
    
    f.write(f"\nBEHAVIORAL FEATURES ({len(behavioral_features)}):\n")
    for feat in behavioral_features:
        f.write(f"  - {feat}\n")

with open('features_excluded.txt', 'w') as f:
    f.write("FEATURES EXPLICITLY EXCLUDED\n")
    f.write("="*70 + "\n\n")
    for col in exclude_columns:
        f.write(f"✗ {col}\n")
    f.write("\nREASONS:\n")
    f.write("- customer_id: Identifier with no predictive value\n")
    f.write("- cutoff_date: Already used to calculate time-based features\n")
    f.write("- loyalty_status: Encoded as loyalty_status_encoded\n")
    f.write("- segment_id: Encoded as segment_id_encoded\n")
    f.write("- target_* columns: Target variables (would cause data leakage)\n")

# Save full dataset for reference
df.to_csv('full_dataset_with_rolling_windows.csv', index=False)
print(f"✓ Saved full dataset with all observations")

# ============================================================================
# 11. SUMMARY
# ============================================================================
print("\n" + "="*80)
print("DATA PREPARATION COMPLETE")
print("="*80)

print(f"""
DATASET SUMMARY:
- Total observations: {len(df)}
- Training observations: {len(train_df)}
- Validation observations: {len(val_df)}
- Unique customers: {df['customer_id'].nunique()}
- Total features: {len(feature_columns)}

TARGET VARIABLES:
1. Total 30-day spend
   - Training avg: {y_train_total.mean():.2f} INR
   - Validation avg: {y_val_total.mean():.2f} INR

2. Electronics spend
   - Training avg: {train_df['target_30d_electronics_spend'].mean():.2f} INR
   - Validation avg: {val_df['target_30d_electronics_spend'].mean():.2f} INR

3. Grocery spend
   - Training avg: {train_df['target_30d_grocery_spend'].mean():.2f} INR
   - Validation avg: {val_df['target_30d_grocery_spend'].mean():.2f} INR

4. Sports spend
   - Training avg: {train_df['target_30d_sports_spend'].mean():.2f} INR
   - Validation avg: {val_df['target_30d_sports_spend'].mean():.2f} INR

APPROACH:
✓ Rolling window observations (weekly cutoffs)
✓ Time-based train/validation split
✓ Category-specific targets for detailed predictions
✓ 40+ features including RFM, behavioral, and category-specific
✓ No data leakage (strict temporal separation)

FILES CREATED:
- X_train_total_spend.csv, y_train_total_spend.csv
- X_val_total_spend.csv, y_val_total_spend.csv
- y_train_electronics_spend.csv, y_val_electronics_spend.csv
- y_train_grocery_spend.csv, y_val_grocery_spend.csv
- y_train_sports_spend.csv, y_val_sports_spend.csv
- features_used.txt, features_excluded.txt
- full_dataset_with_rolling_windows.csv

READY FOR MODEL TRAINING!
""")

print("="*80)
