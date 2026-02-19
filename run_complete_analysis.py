# ============================================================================
# COMPLETE ANALYSIS RUNNER
# Execute all analysis steps in sequence
# ============================================================================

import subprocess
import sys
import os

print("="*80)
print("SUPPLY CHAIN ANALYTICS - COMPLETE ANALYSIS RUNNER")
print("="*80)

def run_script(script_name, description):
    """Run a Python script and handle errors"""
    print(f"\n{'='*80}")
    print(f"RUNNING: {description}")
    print(f"Script: {script_name}")
    print(f"{'='*80}\n")
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=False,
            text=True,
            check=True
        )
        print(f"\n✓ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error running {script_name}")
        print(f"Error: {e}")
        return False
    except FileNotFoundError:
        print(f"\n✗ Script not found: {script_name}")
        return False

def main():
    """Main execution function"""
    
    print("\n🚀 Starting Complete Supply Chain Analysis Pipeline...")
    print("\nThis will execute the following steps:")
    print("  1. Generate sample data (if needed)")
    print("  2. Data cleaning and EDA")
    print("  3. SQL analysis")
    print("  4. Machine learning model training")
    
    input("\nPress Enter to continue...")
    
    # Check if data file exists
    if not os.path.exists('supply_chain_data.csv'):
        print("\n⚠️ Data file not found. Generating sample data...")
        if not run_script('generate_sample_data.py', 'Sample Data Generation'):
            print("\n✗ Failed to generate sample data. Exiting.")
            return
    else:
        print("\n✓ Data file found: supply_chain_data.csv")
    
    # Step 1: Data Analysis and EDA
    if not run_script('supply_chain_analytics.py', 'Data Cleaning & EDA'):
        print("\n✗ Analysis failed. Exiting.")
        return
    
    # Step 2: SQL Analysis
    if not run_script('sql_analysis.py', 'SQL Business Intelligence Analysis'):
        print("\n✗ SQL analysis failed. Continuing anyway...")
    
    # Step 3: Machine Learning
    if not run_script('ml_demand_forecasting.py', 'Machine Learning Model Training'):
        print("\n✗ ML model training failed. Continuing anyway...")
    
    # Final summary
    print("\n" + "="*80)
    print("✓ COMPLETE ANALYSIS PIPELINE FINISHED!")
    print("="*80)
    
    print("\n📁 OUTPUT FILES GENERATED:")
    print("-"*80)
    
    # Check for output files
    outputs = {
        'supply_chain_data_cleaned.csv': 'Cleaned dataset',
        'outputs/': 'EDA visualizations folder',
        'sql_results/': 'SQL query results folder',
        'demand_forecasting_model.h5': 'Trained ML model',
        'scaler.pkl': 'Feature scaler',
        'feature_names.pkl': 'Feature names'
    }
    
    for file, description in outputs.items():
        if os.path.exists(file):
            print(f"  ✓ {file:<40} - {description}")
        else:
            print(f"  ✗ {file:<40} - Not found")
    
    print("\n📊 NEXT STEPS:")
    print("-"*80)
    print("  1. Review visualizations in 'outputs' folder")
    print("  2. Check SQL results in 'sql_results' folder")
    print("  3. Open PowerBI_Dashboard_Guide.md for dashboard creation")
    print("  4. Read PROJECT_REPORT.md for complete documentation")
    
    print("\n" + "="*80)
    print("Thank you for using Supply Chain Analytics!")
    print("="*80)

if __name__ == "__main__":
    main()
