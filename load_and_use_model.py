#!/usr/bin/env python
# ============================================================================
# LOAD AND USE THE TRAINED MODEL
# This script shows how to load and use the demand forecasting model
# ============================================================================

import pickle
import pandas as pd
import numpy as np
import os

print("="*80)
print("LOAD AND USE DEMAND FORECASTING MODEL")
print("="*80)

# ============================================================================
# STEP 1: CHECK WHICH MODEL FILE EXISTS
# ============================================================================
print("\n📂 STEP 1: CHECKING MODEL FILES")
print("-"*80)

model_file = None
model_type = None

if os.path.exists('demand_forecasting_model.h5'):
    model_file = 'demand_forecasting_model.h5'
    model_type = 'tensorflow'
    print("✓ Found TensorFlow model: demand_forecasting_model.h5")
elif os.path.exists('demand_forecasting_model.pkl'):
    model_file = 'demand_forecasting_model.pkl'
    model_type = 'sklearn'
    print("✓ Found sklearn model: demand_forecasting_model.pkl")
else:
    print("✗ No model file found!")
    print("\nPlease run the ML training script first:")
    print("  python ml_demand_forecasting.py")
    exit(1)

# ============================================================================
# STEP 2: LOAD THE MODEL
# ============================================================================
print("\n📦 STEP 2: LOADING MODEL")
print("-"*80)

try:
    if model_type == 'tensorflow':
        from tensorflow import keras
        model = keras.models.load_model(model_file)
        print(f"✓ TensorFlow model loaded successfully")
        print("\nModel Architecture:")
        model.summary()
    else:
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        print(f"✓ Random Forest model loaded successfully")
        print(f"\nModel Type: {type(model).__name__}")
        print(f"Number of Trees: {model.n_estimators}")
        print(f"Max Depth: {model.max_depth}")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    exit(1)

# ============================================================================
# STEP 3: LOAD SCALER AND FEATURE NAMES
# ============================================================================
print("\n📦 STEP 3: LOADING SCALER AND FEATURES")
print("-"*80)

try:
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("✓ Scaler loaded")
    
    with open('feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    print(f"✓ Feature names loaded ({len(feature_names)} features)")
except Exception as e:
    print(f"✗ Error loading scaler/features: {e}")
    exit(1)

# ============================================================================
# STEP 4: LOAD DATA AND MAKE PREDICTIONS
# ============================================================================
print("\n📊 STEP 4: MAKING SAMPLE PREDICTIONS")
print("-"*80)

try:
    # Load the cleaned data
    df = pd.read_csv('supply_chain_data_cleaned.csv')
    print(f"✓ Data loaded: {len(df)} records")
    
    # Prepare features (same as training)
    X = df[feature_names].fillna(0)
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Make predictions
    predictions = model.predict(X_scaled)
    
    if model_type == 'tensorflow':
        predictions = predictions.flatten()
    
    print(f"✓ Predictions made for {len(predictions)} records")
    
    # Add predictions to dataframe
    df['Predicted_Sales'] = predictions
    
    # Show sample predictions
    print("\n📈 SAMPLE PREDICTIONS:")
    print("-"*80)
    print(f"{'Actual Sales':<20} {'Predicted Sales':<20} {'Difference':<20}")
    print("-"*80)
    
    for i in range(min(10, len(df))):
        actual = df['Number of products sold'].iloc[i]
        predicted = predictions[i]
        diff = actual - predicted
        print(f"{actual:<20.2f} {predicted:<20.2f} {diff:<20.2f}")
    
    # Calculate accuracy metrics
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    
    actual_sales = df['Number of products sold']
    mse = mean_squared_error(actual_sales, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual_sales, predictions)
    r2 = r2_score(actual_sales, predictions)
    
    print("\n📊 MODEL PERFORMANCE:")
    print("-"*80)
    print(f"Mean Squared Error (MSE):  {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"R² Score: {r2:.4f}")
    
    # Save predictions to file
    output_file = 'predictions_output.csv'
    df[['Product type', 'Number of products sold', 'Predicted_Sales']].to_csv(output_file, index=False)
    print(f"\n✓ Predictions saved to: {output_file}")
    
except Exception as e:
    print(f"✗ Error making predictions: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# ============================================================================
# STEP 5: HOW TO USE FOR NEW DATA
# ============================================================================
print("\n\n💡 HOW TO USE THIS MODEL FOR NEW DATA")
print("="*80)

print("""
To make predictions on new data:

1. Prepare your data with the same features:
   - Must have all {num_features} features
   - Feature names: {features}

2. Load the model and scaler:
   ```python
   import pickle
   
   # Load model
   with open('{model_file}', 'rb') as f:
       model = pickle.load(f)
   
   # Load scaler
   with open('scaler.pkl', 'rb') as f:
       scaler = pickle.load(f)
   
   # Load feature names
   with open('feature_names.pkl', 'rb') as f:
       feature_names = pickle.load(f)
   ```

3. Prepare and scale your data:
   ```python
   # Ensure features are in correct order
   X_new = new_data[feature_names].fillna(0)
   
   # Scale features
   X_scaled = scaler.transform(X_new)
   ```

4. Make predictions:
   ```python
   predictions = model.predict(X_scaled)
   ```

That's it! The predictions will be the forecasted number of products sold.
""".format(
    num_features=len(feature_names),
    features=', '.join(feature_names[:5]) + '...',
    model_file=model_file
))

print("\n" + "="*80)
print("✓ MODEL LOADED AND READY TO USE!")
print("="*80)
