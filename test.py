#!/usr/bin/env python3
"""
Test script for the Customer Spend Prediction API
"""
import requests
import json
import time
import pandas as pd
import os

BASE_URL = "http://localhost:5000"


def test_health_endpoint():
    """Test the health check endpoint"""
    print("\n" + "=" * 60)
    print("Testing /health endpoint")
    print("=" * 60)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    print("✅ Health check passed!")


def test_predict_endpoint_success():
    """Test the predict endpoint with valid input"""
    print("\n" + "=" * 60)
    print("Testing /predict endpoint (valid input)")
    print("=" * 60)
    
    payload = {"customer_id": "CUST_001"}
    response = requests.post(
        f"{BASE_URL}/predict",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Request: {json.dumps(payload, indent=2)}")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200
    data = response.json()
    assert "customer_id" in data
    assert "predicted_30d_spend" in data
    assert "currency" in data
    assert "model_version" in data
    assert data["currency"] == "INR"
    print("✅ Prediction test passed!")


def test_predict_endpoint_missing_customer_id():
    """Test the predict endpoint with missing customer_id"""
    print("\n" + "=" * 60)
    print("Testing /predict endpoint (missing customer_id)")
    print("=" * 60)
    
    payload = {}
    response = requests.post(
        f"{BASE_URL}/predict",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Request: {json.dumps(payload, indent=2)}")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 400
    assert "error" in response.json()
    print("✅ Validation test passed!")


def test_predict_endpoint_empty_customer_id():
    """Test the predict endpoint with empty customer_id"""
    print("\n" + "=" * 60)
    print("Testing /predict endpoint (empty customer_id)")
    print("=" * 60)
    
    payload = {"customer_id": ""}
    response = requests.post(
        f"{BASE_URL}/predict",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Request: {json.dumps(payload, indent=2)}")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 400
    print("✅ Empty validation test passed!")


def test_predict_endpoint_numeric_customer_id():
    """Test the predict endpoint with numeric customer_id"""
    print("\n" + "=" * 60)
    print("Testing /predict endpoint (numeric customer_id)")
    print("=" * 60)
    
    payload = {"customer_id": 12345}
    response = requests.post(
        f"{BASE_URL}/predict",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Request: {json.dumps(payload, indent=2)}")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200
    print("✅ Numeric customer_id test passed!")


def test_bulk_prediction_template():
    """Test downloading the bulk prediction template"""
    print("\n" + "=" * 60)
    print("Testing /bulk/template endpoint")
    print("=" * 60)
    
    response = requests.get(f"{BASE_URL}/bulk/template")
    print(f"Status Code: {response.status_code}")
    
    assert response.status_code == 200
    assert 'text/csv' in response.headers['Content-Type']
    print("✅ Template download test passed!")


def test_bulk_prediction_upload():
    """Test bulk prediction with file upload"""
    print("\n" + "=" * 60)
    print("Testing /bulk/predict endpoint (file upload)")
    print("=" * 60)
    
    # Create test CSV file
    test_data = pd.DataFrame({
        'customer_id': ['1', '2', '3', '4', '5'],
        'customer_name': ['Test 1', 'Test 2', 'Test 3', 'Test 4', 'Test 5']
    })
    
    test_file = 'test_customers.csv'
    test_data.to_csv(test_file, index=False)
    
    try:
        # Upload file
        with open(test_file, 'rb') as f:
            files = {'file': (test_file, f, 'text/csv')}
            response = requests.post(f"{BASE_URL}/bulk/predict", files=files)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        assert response.status_code == 202
        data = response.json()
        assert 'job_id' in data
        assert data['status'] == 'processing'
        assert 'metadata' in data
        
        job_id = data['job_id']
        print(f"✅ Upload test passed! Job ID: {job_id}")
        
        # Test status endpoint
        print("\nChecking job status...")
        time.sleep(2)  # Wait for processing
        
        status_response = requests.get(f"{BASE_URL}/bulk/status/{job_id}")
        print(f"Status: {json.dumps(status_response.json(), indent=2)}")
        
        assert status_response.status_code == 200
        print("✅ Status check passed!")
        
        # If completed, test download
        if status_response.json()['status'] == 'completed':
            print("\nDownloading results...")
            download_response = requests.get(f"{BASE_URL}/bulk/download/{job_id}")
            assert download_response.status_code == 200
            print("✅ Download test passed!")
        
    finally:
        # Cleanup
        if os.path.exists(test_file):
            os.remove(test_file)


def test_bulk_prediction_invalid_file():
    """Test bulk prediction with invalid file type"""
    print("\n" + "=" * 60)
    print("Testing /bulk/predict endpoint (invalid file)")
    print("=" * 60)
    
    # Create invalid file
    test_file = 'test_invalid.txt'
    with open(test_file, 'w') as f:
        f.write('This is not a valid file')
    
    try:
        with open(test_file, 'rb') as f:
            files = {'file': (test_file, f, 'text/plain')}
            response = requests.post(f"{BASE_URL}/bulk/predict", files=files)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        assert response.status_code == 400
        assert 'error' in response.json()
        print("✅ Invalid file test passed!")
        
    finally:
        if os.path.exists(test_file):
            os.remove(test_file)


if __name__ == "__main__":
    print("\n🧪 Starting API Tests")
    print("Make sure the server is running on http://localhost:5000")
    
    try:
        # Run all tests
        print("\n" + "="*60)
        print("SINGLE PREDICTION TESTS")
        print("="*60)
        
        test_health_endpoint()
        test_predict_endpoint_success()
        test_predict_endpoint_missing_customer_id()
        test_predict_endpoint_empty_customer_id()
        test_predict_endpoint_numeric_customer_id()
        
        print("\n" + "="*60)
        print("BULK PREDICTION TESTS")
        print("="*60)
        
        test_bulk_prediction_template()
        test_bulk_prediction_upload()
        test_bulk_prediction_invalid_file()
        
        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to the server")
        print("Please make sure the Flask app is running on http://localhost:5000")
        print("Run: python run.py")
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
