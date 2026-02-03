#!/usr/bin/env python3
"""
Simple run script for the Flask application
"""
from app import create_app
from config import Config

if __name__ == "__main__":
    print("=" * 50)
    print("🚀 Starting Flask Application")
    print("=" * 50)
    
    try:
        app = create_app()
        
        print(f"\n🔗 Server running at: http://localhost:{Config.PORT}")
        print(f"📍 Health check: http://localhost:{Config.PORT}/")
        print(f"🔍 DB test: http://localhost:{Config.PORT}/test/db")
        print(f"📊 Test API: http://localhost:{Config.PORT}/test")
        print("-" * 50)
        
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
