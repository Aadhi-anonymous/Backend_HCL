"""
Flask Application Factory
Creates and configures the Flask application for customer spend prediction service.
"""
from flask import Flask, jsonify
from flask_cors import CORS
from supabase import create_client
from config import Config

# Import blueprints
from routes.health import health_bp
from routes.prediction import prediction_bp
from routes.bulk_prediction import bulk_prediction_bp


def create_app():
    """
    Create and configure the Flask application.
    
    Returns:
        Configured Flask app instance
    """
    # Validate configuration
    Config.validate()
    
    # Initialize Flask app
    app = Flask(__name__)
    app.config.from_object(Config)
    
    # Configure CORS
    CORS(app, 
         supports_credentials=True,
         origins=Config.CORS_ORIGINS,
         allow_headers=["Content-Type", "Authorization"],
         methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
    
    # Initialize Supabase client (make it accessible to routes if needed)
    app.supabase = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)
    
    # Register blueprints
    app.register_blueprint(health_bp)
    app.register_blueprint(prediction_bp)
    app.register_blueprint(bulk_prediction_bp)
    
    # Root endpoint (redirects to health)
    @app.route("/", methods=["GET"])
    def root():
        return jsonify({
            "service": "Customer Spend Prediction API",
            "version": "1.0.0",
            "endpoints": {
                "health": "/health (GET)",
                "predict": "/predict (POST)",
                "bulk_predict": "/bulk/predict (POST)",
                "bulk_status": "/bulk/status/<job_id> (GET)",
                "bulk_download": "/bulk/download/<job_id> (GET)",
                "bulk_template": "/bulk/template (GET)"
            }
        }), 200
    
    # Global error handlers
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            "error": "Not found",
            "message": "The requested endpoint does not exist"
        }), 404
    
    @app.errorhandler(405)
    def method_not_allowed(error):
        return jsonify({
            "error": "Method not allowed",
            "message": "The requested method is not allowed for this endpoint"
        }), 405
    
    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({
            "error": "Internal server error",
            "message": "An unexpected error occurred"
        }), 500
    
    return app


if __name__ == "__main__":
    app = create_app()
    app.run(
        host="0.0.0.0", 
        port=Config.PORT, 
        debug=(Config.FLASK_ENV == "development")
    )
