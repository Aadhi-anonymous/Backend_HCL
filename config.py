import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Flask
    FLASK_ENV = os.getenv("FLASK_ENV", "development")
    PORT = int(os.getenv("PORT", 5000))
    
    # Supabase
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    
    # CORS
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")
    
    @classmethod
    def validate(cls):
        """Validate required configuration"""
        if not cls.SUPABASE_URL or cls.SUPABASE_URL == "your_supabase_url_here":
            raise ValueError("SUPABASE_URL is not configured in .env")
        if not cls.SUPABASE_KEY or cls.SUPABASE_KEY == "your_supabase_anon_key_here":
            raise ValueError("SUPABASE_KEY is not configured in .env")
