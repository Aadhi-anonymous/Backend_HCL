"""
Prediction Service
Handles ML model inference and feature engineering for customer spend prediction.
"""
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
import xgboost as xgb
import json
import os
from flask import current_app

# Global model cache
_MODEL_CACHE = {}
_METADATA_CACHE = None


def _get_supabase_client():
    """Get Supabase client from Flask app context"""
    return current_app.supabase


def _load_model_metadata():
    """Load model metadata from JSON file"""
    global _METADATA_CACHE
    
    if _METADATA_CACHE is None:
        metadata_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'model_metadata_enhanced.json')
        with open(metadata_path, 'r') as f:
            _METADATA_CACHE = json.load(f)
    
    return _METADATA_CACHE


def _load_xgboost_model(target_type: str):
    """
    Load XGBoost model for a specific target.
    Uses caching to avoid reloading.
    
    Args:
        target_type: One of 'total_spend', 'electronics', 'grocery', 'sports'
    
    Returns:
        Loaded XGBoost model
    """
    global _MODEL_CACHE
    
    if target_type in _MODEL_CACHE:
        return _MODEL_CACHE[target_type]
    
    # Construct model path
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', f'xgboost_{target_type}.json')
    
    # Load model
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    
    # Cache it
    _MODEL_CACHE[target_type] = model
    
    return model


def _fetch_customer_transactions(customer_id: str, supabase_client) -> pd.DataFrame:
    """
    Fetch all transactions for a customer from Supabase.
    
    Args:
        customer_id: Customer ID to fetch transactions for
        supabase_client: Supabase client instance
        
    Returns:
        DataFrame with transaction history
    """
    # Fetch sales headers for this customer
    sales_response = supabase_client.table('store_sales_header')\
        .select('transaction_id, customer_id, store_id, transaction_date')\
        .eq('customer_id', str(customer_id))\
        .execute()
    
    if not sales_response.data or len(sales_response.data) == 0:
        return pd.DataFrame()  # No transactions found
    
    sales_df = pd.DataFrame(sales_response.data)
    transaction_ids = sales_df['transaction_id'].tolist()
    
    # Fetch line items for these transactions
    line_items_response = supabase_client.table('store_sales_line_items')\
        .select('transaction_id, product_id, quantity, line_item_amount, promotion_id')\
        .in_('transaction_id', transaction_ids)\
        .execute()
    
    if not line_items_response.data:
        return pd.DataFrame()
    
    line_items_df = pd.DataFrame(line_items_response.data)
    
    # Fetch product information
    product_ids = line_items_df['product_id'].unique().tolist()
    products_response = supabase_client.table('products')\
        .select('product_id, product_category, unit_price')\
        .in_('product_id', product_ids)\
        .execute()
    
    products_df = pd.DataFrame(products_response.data)
    
    # Merge everything together
    transactions = line_items_df.merge(sales_df, on='transaction_id', how='left')
    transactions = transactions.merge(products_df, on='product_id', how='left')
    
    # Convert date to datetime
    transactions['transaction_date'] = pd.to_datetime(transactions['transaction_date'])
    
    return transactions


def _fetch_customer_details(customer_id: str, supabase_client) -> Dict[str, Any]:
    """
    Fetch customer details from Supabase.
    
    Args:
        customer_id: Customer ID
        supabase_client: Supabase client instance
        
    Returns:
        Dictionary with customer details
    """
    response = supabase_client.table('customer_details')\
        .select('customer_id, loyalty_status, total_loyalty_points, segment_id')\
        .eq('customer_id', str(customer_id))\
        .execute()
    
    if response.data and len(response.data) > 0:
        return response.data[0]
    
    return {}


def _extract_features(customer_id: str, supabase_client) -> Dict[str, float]:
    """
    Extract all features for a customer from the database.
    
    Args:
        customer_id: Customer identifier
        supabase_client: Supabase client instance
        
    Returns:
        Dictionary of features matching the training data
    """
    # Fetch transaction history
    transactions = _fetch_customer_transactions(customer_id, supabase_client)
    
    if len(transactions) == 0:
        raise ValueError(f"No transaction history found for customer {customer_id}")
    
    # Fetch customer details
    customer_details = _fetch_customer_details(customer_id, supabase_client)
    
    # Use current date as cutoff
    cutoff_date = datetime.now(timezone.utc)
    
    # Filter transactions before cutoff
    customer_txns = transactions[transactions['transaction_date'] < cutoff_date].copy()
    
    if len(customer_txns) == 0:
        raise ValueError(f"No valid transactions found for customer {customer_id}")
    
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
    if len(customer_txns) > 0:
        top_category = customer_txns.groupby('product_category')['line_item_amount'].sum().idxmax()
        for category in ['Electronics', 'Grocery', 'Sports']:
            features[f'favorite_category_{category.lower()}'] = 1 if top_category == category else 0
    else:
        for category in ['Electronics', 'Grocery', 'Sports']:
            features[f'favorite_category_{category.lower()}'] = 0
    
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
    
    # ========== CUSTOMER METADATA ==========
    features['total_loyalty_points'] = customer_details.get('total_loyalty_points', 0)
    
    # Encode loyalty status
    loyalty_status_map = {'Bronze': 1, 'Silver': 2, 'Gold': 3, 'Platinum': 4}
    loyalty_status = customer_details.get('loyalty_status', 'Bronze')
    features['loyalty_status_encoded'] = loyalty_status_map.get(loyalty_status, 1)
    
    # Encode segment ID
    segment_map = {'LR': 1, 'AR': 2, 'HR': 3}
    segment_id = customer_details.get('segment_id', 'LR')
    features['segment_id_encoded'] = segment_map.get(segment_id, 1)
    
    return features


def predict_customer_spend(customer_id: str) -> Dict[str, Any]:
    """
    Predict 30-day spend for a given customer.
    
    Args:
        customer_id: Unique identifier for the customer
        
    Returns:
        Dictionary containing prediction results
    """
    try:
        # Get Supabase client
        supabase_client = _get_supabase_client()
        
        # Extract features
        features = _extract_features(customer_id, supabase_client)
        
        # Load metadata to get feature order
        metadata = _load_model_metadata()
        feature_names = metadata['features']
        
        # Create feature vector in correct order
        feature_vector = []
        for feat_name in feature_names:
            feature_vector.append(features.get(feat_name, 0))
        
        # Convert to numpy array and reshape for prediction
        X = np.array(feature_vector).reshape(1, -1)
        
        # Load models and make predictions
        model_total = _load_xgboost_model('total_spend')
        model_electronics = _load_xgboost_model('electronics')
        model_grocery = _load_xgboost_model('grocery')
        model_sports = _load_xgboost_model('sports')
        
        # Make predictions
        pred_total = float(model_total.predict(X)[0])
        pred_electronics = float(model_electronics.predict(X)[0])
        pred_grocery = float(model_grocery.predict(X)[0])
        pred_sports = float(model_sports.predict(X)[0])
        
        # Ensure non-negative predictions
        pred_total = max(0, pred_total)
        pred_electronics = max(0, pred_electronics)
        pred_grocery = max(0, pred_grocery)
        pred_sports = max(0, pred_sports)
        
        # Round to 2 decimal places
        pred_total = round(pred_total, 2)
        pred_electronics = round(pred_electronics, 2)
        pred_grocery = round(pred_grocery, 2)
        pred_sports = round(pred_sports, 2)
        
        return {
            "customer_id": str(customer_id),
            "predicted_30d_spend": {
                "total": pred_total,
                "electronics": pred_electronics,
                "grocery": pred_grocery,
                "sports": pred_sports
            },
            "currency": "INR",
            "model_version": "v1.0-xgboost",
            "model_info": {
                "trained_on": metadata.get('training_date'),
                "num_features": len(feature_names)
            }
        }
        
    except ValueError as e:
        # Handle customer not found or no data
        raise ValueError(str(e))
    except Exception as e:
        # Handle other errors
        raise Exception(f"Prediction error: {str(e)}")


def predict_customer_spend_bulk(customer_ids: List[str]) -> List[Dict[str, Any]]:
    """
    Predict 30-day spend for multiple customers (bulk prediction).
    
    Args:
        customer_ids: List of customer identifiers
        
    Returns:
        List of prediction dictionaries
    """
    results = []
    
    for customer_id in customer_ids:
        try:
            prediction = predict_customer_spend(customer_id)
            results.append(prediction)
        except Exception as e:
            # Handle errors gracefully
            results.append({
                "customer_id": str(customer_id),
                "predicted_30d_spend": {
                    "total": None,
                    "electronics": None,
                    "grocery": None,
                    "sports": None
                },
                "currency": "INR",
                "error": str(e),
                "status": "failed"
            })
    
    return results


def validate_customer_exists(customer_id: str) -> bool:
    """
    Validate that a customer exists in the database.
    
    Args:
        customer_id: Customer identifier to validate
        
    Returns:
        True if customer exists, False otherwise
    """
    try:
        supabase_client = _get_supabase_client()
        response = supabase_client.table('customer_details')\
            .select('customer_id')\
            .eq('customer_id', str(customer_id))\
            .execute()
        
        return response.data and len(response.data) > 0
    except Exception:
        return False
