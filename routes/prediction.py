"""
Prediction routes
Handles customer spend prediction API endpoints.
"""
from flask import Blueprint, request, jsonify
from services.prediction_service import predict_customer_spend

prediction_bp = Blueprint('prediction', __name__)


@prediction_bp.route('/predict', methods=['POST'])
def predict():
    """
    Predict 30-day customer spend.
    
    Request JSON:
        {
            "customer_id": "<string or int>"
        }
    
    Response JSON:
        {
            "customer_id": "...",
            "predicted_30d_spend": null,
            "currency": "INR",
            "model_version": "v0.1-placeholder"
        }
    
    Returns:
        200: Successful prediction
        400: Bad request (missing customer_id)
        500: Internal server error
    """
    try:
        # Parse request body
        data = request.get_json()
        
        # Validate request
        if not data:
            return jsonify({
                "error": "Invalid request body",
                "message": "Request body must be valid JSON"
            }), 400
        
        # Validate customer_id presence
        customer_id = data.get('customer_id')
        if customer_id is None or customer_id == "":
            return jsonify({
                "error": "Missing required field",
                "message": "customer_id is required"
            }), 400
        
        # Call prediction service
        result = predict_customer_spend(customer_id)
        
        return jsonify(result), 200
        
    except ValueError as e:
        # Handle validation errors (customer not found, no data, etc.)
        return jsonify({
            "error": "Customer data error",
            "message": str(e),
            "customer_id": str(customer_id) if customer_id else None
        }), 404
        
    except Exception as e:
        # Handle unexpected errors
        import traceback
        print(f"Prediction error for customer {customer_id}: {str(e)}")
        print(traceback.format_exc())
        
        return jsonify({
            "error": "Prediction failed",
            "message": str(e),
            "customer_id": str(customer_id) if customer_id else None
        }), 500
