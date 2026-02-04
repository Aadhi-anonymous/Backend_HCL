#!/usr/bin/env python3
"""
Simple run script for the Flask application
"""
from app import create_app
from config import Config

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Starting Customer Spend Prediction Service")
    print("=" * 60)
    
    try:
        app = create_app()
        
        print(f"\n🔗 Server running at: http://localhost:{Config.PORT}")
        print(f"📍 API Info: http://localhost:{Config.PORT}/")
        print(f"💚 Health Check: http://localhost:{Config.PORT}/health")
        print(f"🔮 Prediction: http://localhost:{Config.PORT}/predict (POST)")
        print("-" * 60)
        print(f"\n🧪 Example curl command:")
        print(f'   curl -X POST http://localhost:{Config.PORT}/predict \\')
        print(f'        -H "Content-Type: application/json" \\')
        print(f'        -d \'{{"customer_id": "CUST_001"}}\'')
        print("-" * 60)
        
        app.run(
            host="0.0.0.0",
            port=Config.PORT,
            debug=(Config.FLASK_ENV == "development")
        )
    except ValueError as e:
        print(f"\n❌ Configuration Error: {e}")
        print("\n📝 Please update your .env file with:")
        print("   - SUPABASE_URL")
        print("   - SUPABASE_KEY")
    except Exception as e:
        print(f"\n❌ Error: {e}")
