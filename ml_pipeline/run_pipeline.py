#!/usr/bin/env python3
"""
Run Complete ML Pipeline
Executes data preparation and model training in sequence
"""
import subprocess
import sys
import os

def run_script(script_name, description):
    """Run a Python script and handle errors"""
    print("\n" + "="*80)
    print(f"Running: {description}")
    print("="*80 + "\n")
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=False,
            text=True
        )
        print(f"\n✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} failed!")
        print(f"Error code: {e.returncode}")
        return False
    except Exception as e:
        print(f"\n❌ Error running {description}: {str(e)}")
        return False

def main():
    """Run the complete ML pipeline"""
    print("\n" + "="*80)
    print("CUSTOMER SPEND PREDICTION - ML PIPELINE")
    print("="*80)
    
    # Check if we're in the right directory
    if not os.path.exists('02_data_preparation_enhanced.py'):
        print("\n❌ Error: Please run this script from the ml_pipeline directory")
        print("   cd ml_pipeline && python run_pipeline.py")
        sys.exit(1)
    
    # Step 1: Data Preparation
    if not run_script('02_data_preparation_enhanced.py', 'Step 1: Data Preparation'):
        print("\n❌ Pipeline stopped due to data preparation failure")
        sys.exit(1)
    
    # Step 2: Model Training
    if not run_script('03_train_models_enhanced.py', 'Step 2: Model Training'):
        print("\n❌ Pipeline stopped due to model training failure")
        sys.exit(1)
    
    # Success!
    print("\n" + "="*80)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nResults:")
    print("  📊 Model comparison: model_comparison_all_targets.csv")
    print("  🤖 Trained models: ../models/")
    print("  📋 Metadata: ../models/model_metadata_enhanced.json")
    print("\nNext steps:")
    print("  1. Review model metrics: cat model_comparison_all_targets.csv")
    print("  2. Start the API server: cd .. && python run.py")
    print("  3. Test predictions: python test_prediction.py")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
