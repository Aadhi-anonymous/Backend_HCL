"""
Test Prediction Service
Tests the prediction service with actual customer data from database
"""
import requests
import json

# API endpoint
BASE_URL = "http://localhost:5000"

def test_health():
    """Test health endpoint"""
    print("="*80)
    print("Testing Health Endpoint")
    print("="*80)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_single_prediction(customer_id):
    """Test single customer prediction"""
    print("="*80)
    print(f"Testing Single Prediction for Customer: {customer_id}")
    print("="*80)
    
    payload = {
        "customer_id": customer_id
    }
    
    response = requests.post(
        f"{BASE_URL}/predict",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_multiple_customers():
    """Test predictions for multiple customers"""
    # Common customer IDs from the dataset
    customer_ids = ["1", "2", "3", "5", "10"]
    
    print("="*80)
    print(f"Testing Multiple Customers: {customer_ids}")
    print("="*80)
    
    for customer_id in customer_ids:
        test_single_prediction(customer_id)

def test_invalid_customer():
    """Test with invalid customer ID"""
    print("="*80)
    print("Testing Invalid Customer ID")
    print("="*80)
    
    payload = {
        "customer_id": "99999999"
    }
    
    response = requests.post(
        f"{BASE_URL}/predict",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_missing_customer_id():
    """Test with missing customer_id"""
    print("="*80)
    print("Testing Missing Customer ID")
    print("="*80)
    
    payload = {}
    
    response = requests.post(
        f"{BASE_URL}/predict",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

if __name__ == "__main__":
    print("\n" + "="*80)
    print("CUSTOMER SPEND PREDICTION API - TEST SUITE")
    print("="*80 + "\n")
    
    try:
        # Test health
        test_health()
        
        # Test valid customer
        test_single_prediction("1")
        
        # Test multiple customers
        # test_multiple_customers()
        
        # Test invalid customer
        # test_invalid_customer()
        
        # Test missing customer_id
        # test_missing_customer_id()
        
        print("="*80)
        print("TEST SUITE COMPLETED")
        print("="*80)
        
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to the API server.")
        print("Please make sure the server is running on http://localhost:5000")
    except Exception as e:
        print(f"ERROR: {str(e)}")
