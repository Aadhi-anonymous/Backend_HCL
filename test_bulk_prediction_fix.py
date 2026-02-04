"""
Test script to verify bulk prediction JSON serialization fixes
"""
import pandas as pd
import numpy as np
import json

# Simulate the metadata creation with problematic types
def test_metadata_serialization():
    """Test that metadata with numpy types can be JSON serialized"""
    
    # Create a sample dataframe
    df = pd.DataFrame({
        'customer_id': ['1', '2', '3', '1', '4'],
        'name': ['A', 'B', 'C', 'A', 'D']
    })
    
    # Original implementation (would fail)
    print("Testing ORIGINAL implementation (problematic):")
    metadata_old = {
        'filename': 'test.csv',
        'file_size_bytes': 1024,
        'file_size_mb': round(1024 / (1024 * 1024), 2),
        'total_rows': len(df),
        'total_customers': df.iloc[:, 0].nunique(),
        'columns': list(df.columns),
        'has_duplicates': df.iloc[:, 0].duplicated().any()  # This returns numpy.bool_
    }
    
    print(f"has_duplicates type: {type(metadata_old['has_duplicates'])}")
    print(f"has_duplicates value: {metadata_old['has_duplicates']}")
    
    try:
        json_str = json.dumps(metadata_old)
        print("❌ OLD: Should have failed but didn't!")
    except TypeError as e:
        print(f"✅ OLD: Failed as expected with error: {e}")
    
    print("\n" + "="*70 + "\n")
    
    # Fixed implementation
    print("Testing FIXED implementation:")
    metadata_new = {
        'filename': 'test.csv',
        'file_size_bytes': int(1024),
        'file_size_mb': round(float(1024 / (1024 * 1024)), 2),
        'total_rows': int(len(df)),
        'total_customers': int(df.iloc[:, 0].nunique()),
        'columns': list(df.columns),
        'has_duplicates': bool(df.iloc[:, 0].duplicated().any())  # Convert to Python bool
    }
    
    print(f"has_duplicates type: {type(metadata_new['has_duplicates'])}")
    print(f"has_duplicates value: {metadata_new['has_duplicates']}")
    
    try:
        json_str = json.dumps(metadata_new, indent=2)
        print("✅ NEW: Successfully serialized to JSON!")
        print("\nJSON output:")
        print(json_str)
    except TypeError as e:
        print(f"❌ NEW: Failed with error: {e}")
    
    print("\n" + "="*70 + "\n")


def test_statistics_serialization():
    """Test that statistics with float calculations can be JSON serialized"""
    
    print("Testing statistics serialization:")
    
    # Sample prediction results
    total_spends = [159848.97, 126797.8, 163753.91, 229482.14, 154165.42]
    
    # Fixed implementation
    statistics = {
        'total_predicted_spend': round(float(sum(total_spends)), 2),
        'average_predicted_spend': round(float(sum(total_spends) / len(total_spends)), 2),
        'min_predicted_spend': round(float(min(total_spends)), 2),
        'max_predicted_spend': round(float(max(total_spends)), 2),
        'currency': 'INR'
    }
    
    try:
        json_str = json.dumps(statistics, indent=2)
        print("✅ Successfully serialized statistics to JSON!")
        print("\nJSON output:")
        print(json_str)
    except TypeError as e:
        print(f"❌ Failed with error: {e}")
    
    print("\n" + "="*70 + "\n")


def test_progress_serialization():
    """Test that progress with calculated percentage can be JSON serialized"""
    
    print("Testing progress serialization:")
    
    # Sample progress data
    total_customers = 5
    processed_customers = 5
    successful_predictions = 5
    failed_predictions = 0
    
    progress = {
        'total': int(total_customers),
        'processed': int(processed_customers),
        'successful': int(successful_predictions),
        'failed': int(failed_predictions),
        'percentage': round(float((processed_customers / total_customers) * 100), 2) if total_customers > 0 else 0.0
    }
    
    try:
        json_str = json.dumps(progress, indent=2)
        print("✅ Successfully serialized progress to JSON!")
        print("\nJSON output:")
        print(json_str)
    except TypeError as e:
        print(f"❌ Failed with error: {e}")
    
    print("\n" + "="*70 + "\n")


def test_full_response():
    """Test a complete response object"""
    
    print("Testing full response object:")
    
    df = pd.DataFrame({
        'customer_id': ['1', '2', '3', '4', '5']
    })
    
    metadata = {
        'filename': 'bulk_prediction_template.csv',
        'file_size_bytes': int(1024),
        'file_size_mb': round(float(1024 / (1024 * 1024)), 2),
        'total_rows': int(len(df)),
        'total_customers': int(df.iloc[:, 0].nunique()),
        'columns': list(df.columns),
        'has_duplicates': bool(df.iloc[:, 0].duplicated().any())
    }
    
    response = {
        'job_id': 'test-job-123',
        'status': 'completed',
        'progress': {
            'total': int(5),
            'processed': int(5),
            'successful': int(5),
            'failed': int(0),
            'percentage': round(float(100.0), 2)
        },
        'metadata': metadata,
        'statistics': {
            'total_predicted_spend': round(float(834048.24), 2),
            'average_predicted_spend': round(float(166809.65), 2),
            'min_predicted_spend': round(float(126797.8), 2),
            'max_predicted_spend': round(float(229482.14), 2),
            'currency': 'INR'
        },
        'created_at': '2026-02-04T21:24:24',
        'updated_at': '2026-02-04T21:24:28',
        'completed_at': '2026-02-04T21:24:28',
        'processing_time_seconds': 4.2
    }
    
    try:
        json_str = json.dumps(response, indent=2)
        print("✅ Successfully serialized full response to JSON!")
        print("\nJSON output:")
        print(json_str)
    except TypeError as e:
        print(f"❌ Failed with error: {e}")
    
    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    print("\n" + "="*70)
    print("BULK PREDICTION JSON SERIALIZATION TEST")
    print("="*70 + "\n")
    
    test_metadata_serialization()
    test_statistics_serialization()
    test_progress_serialization()
    test_full_response()
    
    print("\n" + "="*70)
    print("ALL TESTS COMPLETED!")
    print("="*70 + "\n")
