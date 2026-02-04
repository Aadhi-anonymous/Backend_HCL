"""
Data Exploration Script
Explore all CSV files in the pymind_dataset folder to understand the schema
"""
import pandas as pd
import os

# Define data directory
DATA_DIR = "../pymind_dataset"

def explore_csv(filename):
    """Explore a single CSV file"""
    filepath = os.path.join(DATA_DIR, filename)
    print("\n" + "="*80)
    print(f"FILE: {filename}")
    print("="*80)
    
    try:
        df = pd.read_csv(filepath)
        print(f"\nShape: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"\nColumn Names and Types:")
        print(df.dtypes)
        print(f"\nFirst 5 rows:")
        print(df.head())
        print(f"\nMissing Values:")
        print(df.isnull().sum())
        print(f"\nBasic Statistics:")
        print(df.describe())
        
        return df
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None

if __name__ == "__main__":
    # List all CSV files
    csv_files = [
        "customer_details.csv",
        "stores.csv",
        "products.csv",
        "loyalty_rules.csv",
        "promotion_details.csv",
        "store_sales_header.csv",
        "store_sales_line_items.csv"
    ]
    
    print("PYMIND DATASET EXPLORATION")
    print("="*80)
    
    # Explore each file
    datasets = {}
    for filename in csv_files:
        datasets[filename] = explore_csv(filename)
    
    print("\n" + "="*80)
    print("EXPLORATION COMPLETE")
    print("="*80)
