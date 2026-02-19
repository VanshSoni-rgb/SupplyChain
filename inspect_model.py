#!/usr/bin/env python
# ============================================================================
# MODEL INSPECTOR
# View information about the saved model without loading full data
# ============================================================================

import pickle
import os

print("="*80)
print("MODEL FILE INSPECTOR")
print("="*80)

# ============================================================================
# CHECK ALL MODEL-RELATED FILES
# ============================================================================
print("\n📁 CHECKING MODEL FILES:")
print("-"*80)

files_to_check = {
    'demand_forecasting_model.h5': 'TensorFlow/Keras Neural Network Model',
    'demand_forecasting_model.pkl': 'Scikit-learn Random Forest Model',
    'scaler.pkl': 'Feature Scaler (StandardScaler)',
    'feature_names.pkl': 'List of Feature Names'
}

found_files = {}
for filename, description in files_to_check.items():
    if os.path.exists(filename):
        size = os.path.getsize(filename)
        size_kb = size / 1024
        print(f"✓ {filename:<35} - {description}")
        print(f"  Size: {size_kb:.2f} KB")
        found_files[filename] = description
    else:
        print(f"✗ {filename:<35} - NOT FOUND")

if not found_files:
    print("\n⚠️ No model files found!")
    print("\nPlease run the ML training script first:")
    print("  python ml_demand_forecasting.py")
    exit(0)

# ============================================================================
# INSPECT PICKLE FILES
# ============================================================================
print("\n\n📦 INSPECTING PICKLE FILES:")
print("="*80)

# Inspect scaler.pkl
if 'scaler.pkl' in found_files:
    print("\n1. SCALER.PKL")
    print("-"*80)
    try:
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        print(f"✓ Type: {type(scaler).__name__}")
        print(f"✓ Number of features: {scaler.n_features_in_}")
        print(f"✓ Mean values: {scaler.mean_[:5]}... (showing first 5)")
        print(f"✓ Scale values: {scaler.scale_[:5]}... (showing first 5)")
        print("\nThis scaler normalizes features before prediction.")
    except Exception as e:
        print(f"✗ Error reading scaler.pkl: {e}")

# Inspect feature_names.pkl
if 'feature_names.pkl' in found_files:
    print("\n2. FEATURE_NAMES.PKL")
    print("-"*80)
    try:
        with open('feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        print(f"✓ Type: {type(feature_names).__name__}")
        print(f"✓ Number of features: {len(feature_names)}")
        print(f"\n✓ Feature list:")
        for i, name in enumerate(feature_names, 1):
            print(f"   {i:2d}. {name}")
        print("\nThese are the features the model expects for predictions.")
    except Exception as e:
        print(f"✗ Error reading feature_names.pkl: {e}")

# Inspect demand_forecasting_model.pkl
if 'demand_forecasting_model.pkl' in found_files:
    print("\n3. DEMAND_FORECASTING_MODEL.PKL")
    print("-"*80)
    try:
        with open('demand_forecasting_model.pkl', 'rb') as f:
            model = pickle.load(f)
        print(f"✓ Type: {type(model).__name__}")
        
        # If it's a Random Forest, show details
        if hasattr(model, 'n_estimators'):
            print(f"✓ Number of trees: {model.n_estimators}")
        if hasattr(model, 'max_depth'):
            print(f"✓ Max depth: {model.max_depth}")
        if hasattr(model, 'n_features_in_'):
            print(f"✓ Number of features: {model.n_features_in_}")
        if hasattr(model, 'feature_importances_'):
            print(f"\n✓ Top 5 Most Important Features:")
            if 'feature_names.pkl' in found_files:
                with open('feature_names.pkl', 'rb') as f:
                    feature_names = pickle.load(f)
                importances = list(zip(feature_names, model.feature_importances_))
                importances.sort(key=lambda x: x[1], reverse=True)
                for i, (name, importance) in enumerate(importances[:5], 1):
                    print(f"   {i}. {name:<30} {importance:.4f}")
        
        print("\nThis is the trained machine learning model.")
        print("Use load_and_use_model.py to make predictions.")
    except Exception as e:
        print(f"✗ Error reading demand_forecasting_model.pkl: {e}")

# Inspect TensorFlow model if exists
if 'demand_forecasting_model.h5' in found_files:
    print("\n4. DEMAND_FORECASTING_MODEL.H5")
    print("-"*80)
    print("✓ This is a TensorFlow/Keras model file")
    print("✓ Cannot inspect without loading TensorFlow")
    print("\nTo view architecture, run:")
    print("  python load_and_use_model.py")

# ============================================================================
# USAGE INSTRUCTIONS
# ============================================================================
print("\n\n💡 HOW TO USE THESE FILES:")
print("="*80)

print("""
These files work together to make predictions:

1. MODEL FILE (.pkl or .h5)
   - Contains the trained machine learning model
   - Cannot be opened in text editor (binary format)
   - Load with pickle or keras

2. SCALER.PKL
   - Normalizes input features
   - Must be applied before prediction
   - Ensures features are on same scale

3. FEATURE_NAMES.PKL
   - Lists required features in correct order
   - Ensures data is formatted correctly
   - Must match training data structure

TO MAKE PREDICTIONS:
  python load_and_use_model.py

TO VIEW MODEL DETAILS:
  python inspect_model.py (this script)

TO RETRAIN MODEL:
  python ml_demand_forecasting.py
""")

print("\n" + "="*80)
print("✓ INSPECTION COMPLETE")
print("="*80)
