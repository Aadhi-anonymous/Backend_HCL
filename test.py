#!/usr/bin/env python3
"""
Quick test script to verify Supabase connection
"""
from supabase import create_client
from config import Config

def test_connection():
    print("=" * 50)
    print("🔍 Testing Supabase Connection")
    print("=" * 50)
    print()
    
    try:
        Config.validate()
        print("✅ Configuration validated")
        print(f"   URL: {Config.SUPABASE_URL}")
        print(f"   Key: {Config.SUPABASE_KEY[:20]}...")
        print()
        
        # Create client
        supabase = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)
        print("✅ Supabase client created")
        print()
        
        # Test query
        print("📊 Testing 'test' table...")
        result = supabase.table("test").select("*").limit(5).execute()
        
        print(f"✅ Query successful!")
        print(f"   Found {len(result.data)} record(s)")
        
        if result.data:
            print("\n📋 Sample records:")
            for record in result.data:
                print(f"   - ID: {record.get('id')}, Created: {record.get('created_at')}")
        else:
            print("\n⚠️  Table is empty (no records)")
        
        print()
        print("=" * 50)
        print("✅ All tests passed!")
        print("=" * 50)
        print("\nYou can now run: python run.py")
        
        return True
        
    except ValueError as e:
        print(f"❌ Configuration Error: {e}")
        print("\n📝 Update your .env file with:")
        print("   SUPABASE_URL=https://xxx.supabase.co")
        print("   SUPABASE_KEY=your_anon_key")
        return False
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Tips:")
        print("   - Check if your Supabase project is active")
        print("   - Verify the URL and KEY are correct")
        print("   - Make sure 'test' table exists in your database")
        return False

if __name__ == "__main__":
    test_connection()
