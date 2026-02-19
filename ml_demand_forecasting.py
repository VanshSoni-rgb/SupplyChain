# ============================================================================
# MACHINE LEARNING MODEL - DEMAND FORECASTING
# Predicting Number of Products Sold
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Try to import TensorFlow with error handling
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
    print("✓ TensorFlow loaded successfully")
except ImportError as e:
    TF_AVAILABLE = False
    print("="*80)
    print("⚠️ TensorFlow not installed or incompatible with your system")
    print("="*80)
    print("\nOptions to fix:")
    print("1. Install TensorFlow: pip install tensorflow")
    print("2. For older systems: pip install tensorflow==2.10.0")
    print("3. For M1/M2 Mac: pip install tensorflow-macos tensorflow-metal")
    print("\nFalling back to simpler ML model (Random Forest)...")
    print("="*80)
    from sklearn.ensemble import RandomForestRegressor

print("="*80)
print("MACHINE LEARNING MODEL - DEMAND FORECASTING")
print("="*80)

# ============================================================================
# STEP 1: LOAD CLEANED DATA
# ============================================================================
print("\n📂 STEP 1: LOADING CLEANED DATA")
print("-"*80)

df = pd.read_csv('supply_chain_data_cleaned.csv')
print(f"✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ============================================================================
# STEP 2: FEATURE SELECTION & ENCODING
# ============================================================================
print("\n\n⚙️ STEP 2: FEATURE PREPARATION")
print("-"*80)

# Select target variable
target_column = 'Number of products sold'

# Check if target exists
if target_column not in df.columns:
    print(f"⚠️ Target column '{target_column}' not found. Available columns:")
    print(df.columns.tolist())
    # Try alternative names
    possible_targets = [col for col in df.columns if 'sold' in col.lower() or 'sales' in col.lower()]
    if possible_targets:
        target_column = possible_targets[0]
        print(f"✓ Using alternative target: {target_column}")

# Select features for modeling
numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()

# Remove target from features
if target_column in numeric_features:
    numeric_features.remove(target_column)

# Select categorical features for encoding
categorical_features = ['Product type', 'Supplier name', 'Transportation modes', 
                        'Shipping carriers', 'Location']
categorical_features = [col for col in categorical_features if col in df.columns]

print(f"\n✓ Target Variable: {target_column}")
print(f"✓ Numeric Features: {len(numeric_features)}")
print(f"✓ Categorical Features: {len(categorical_features)}")

# Create a copy for modeling
df_model = df.copy()

# Encode categorical variables
print("\n🔧 Encoding Categorical Variables...")
label_encoders = {}

for col in categorical_features:
    le = LabelEncoder()
    df_model[col + '_encoded'] = le.fit_transform(df_model[col].astype(str))
    label_encoders[col] = le
    print(f"✓ Encoded: {col}")

# Prepare feature list
encoded_features = [col + '_encoded' for col in categorical_features]
all_features = numeric_features + encoded_features

# Remove any features that might cause data leakage
features_to_remove = ['Revenue generated', 'Profit', 'Revenue_per_Product']
all_features = [f for f in all_features if f not in features_to_remove]

print(f"\n✓ Total Features for Model: {len(all_features)}")

# ============================================================================
# STEP 3: PREPARE DATA FOR TRAINING
# ============================================================================
print("\n\n📊 STEP 3: PREPARING TRAINING DATA")
print("-"*80)

# Create feature matrix and target vector
X = df_model[all_features].fillna(0)
y = df_model[target_column].fillna(0)

print(f"✓ Feature Matrix Shape: {X.shape}")
print(f"✓ Target Vector Shape: {y.shape}")

# Split data into training and testing sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n✓ Training Set: {X_train.shape[0]} samples")
print(f"✓ Testing Set: {X_test.shape[0]} samples")

# Standardize features
print("\n🔧 Standardizing Features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("✓ Features standardized using StandardScaler")

# ============================================================================
# STEP 4: BUILD MACHINE LEARNING MODEL
# ============================================================================
print("\n\n🧠 STEP 4: BUILDING MACHINE LEARNING MODEL")
print("-"*80)

if TF_AVAILABLE:
    # Set random seed for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)

    # Build the neural network model
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )

    print("✓ Neural Network Model Architecture:")
    model.summary()
else:
    # Use Random Forest as fallback
    print("✓ Using Random Forest Regressor (sklearn)")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    print("✓ Model Configuration:")
    print(f"   - Algorithm: Random Forest")
    print(f"   - Trees: 100")
    print(f"   - Max Depth: 10")
    print(f"   - Features: {X_train_scaled.shape[1]}")

# ============================================================================
# STEP 5: TRAIN THE MODEL
# ============================================================================
print("\n\n🎯 STEP 5: TRAINING THE MODEL")
print("-"*80)

if TF_AVAILABLE:
    # Early stopping to prevent overfitting
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    # Train the model
    print("\n🚀 Training in progress...")
    history = model.fit(
        X_train_scaled, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )
    print("\n✓ Training Complete!")
else:
    # Train Random Forest
    print("\n🚀 Training Random Forest model...")
    model.fit(X_train_scaled, y_train)
    print("✓ Training Complete!")
    # Create dummy history for compatibility
    history = None

# ============================================================================
# STEP 6: MODEL EVALUATION
# ============================================================================
print("\n\n📈 STEP 6: MODEL EVALUATION")
print("-"*80)

# Make predictions
y_pred_train = model.predict(X_train_scaled).flatten()
y_pred_test = model.predict(X_test_scaled).flatten()

# Calculate metrics
train_mse = mean_squared_error(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)
train_rmse = np.sqrt(train_mse)
test_rmse = np.sqrt(test_mse)
train_mae = mean_absolute_error(y_train, y_pred_train)
test_mae = mean_absolute_error(y_test, y_pred_test)
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

print("\n📊 MODEL PERFORMANCE METRICS:")
print("-"*80)
print(f"Training Set:")
print(f"  • MSE:  {train_mse:.2f}")
print(f"  • RMSE: {train_rmse:.2f}")
print(f"  • MAE:  {train_mae:.2f}")
print(f"  • R²:   {train_r2:.4f}")
print(f"\nTesting Set:")
print(f"  • MSE:  {test_mse:.2f}")
print(f"  • RMSE: {test_rmse:.2f}")
print(f"  • MAE:  {test_mae:.2f}")
print(f"  • R²:   {test_r2:.4f}")

# ============================================================================
# STEP 7: VISUALIZATIONS
# ============================================================================
print("\n\n📊 STEP 7: CREATING VISUALIZATIONS")
print("-"*80)

import os
if not os.path.exists('outputs'):
    os.makedirs('outputs')

# Visualization 1: Training History (only for TensorFlow)
if TF_AVAILABLE and history is not None:
    print("\n📈 Creating Training History Plot...")
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    plt.title('Model Loss During Training', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE', linewidth=2)
    plt.plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
    plt.title('Model MAE During Training', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MAE', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('outputs/training_history.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: training_history.png")
else:
    print("\n⚠️ Skipping training history plot (not available for Random Forest)")

# Visualization 2: Actual vs Predicted (Test Set)
print("\n📈 Creating Actual vs Predicted Plot...")
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_test, alpha=0.5, s=50)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', lw=2, label='Perfect Prediction')
plt.title('Actual vs Predicted Values (Test Set)', fontsize=16, fontweight='bold')
plt.xlabel('Actual Number of Products Sold', fontsize=12)
plt.ylabel('Predicted Number of Products Sold', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/actual_vs_predicted.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: actual_vs_predicted.png")

# Visualization 3: Prediction Errors
print("\n📈 Creating Prediction Error Distribution...")
errors = y_test - y_pred_test
plt.figure(figsize=(10, 6))
plt.hist(errors, bins=50, edgecolor='black', alpha=0.7)
plt.title('Distribution of Prediction Errors', fontsize=16, fontweight='bold')
plt.xlabel('Prediction Error', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/prediction_errors.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: prediction_errors.png")

# ============================================================================
# STEP 8: SAVE MODEL
# ============================================================================
print("\n\n💾 STEP 8: SAVING MODEL")
print("-"*80)

# Save the trained model
if TF_AVAILABLE:
    model.save('demand_forecasting_model.h5')
    print("✓ Model saved as: demand_forecasting_model.h5")
else:
    import pickle
    with open('demand_forecasting_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("✓ Model saved as: demand_forecasting_model.pkl")

# Save the scaler
import pickle
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✓ Scaler saved as: scaler.pkl")

# Save feature names
with open('feature_names.pkl', 'wb') as f:
    pickle.dump(all_features, f)
print("✓ Feature names saved as: feature_names.pkl")

# ============================================================================
# STEP 9: SAMPLE PREDICTIONS
# ============================================================================
print("\n\n🔮 STEP 9: SAMPLE PREDICTIONS")
print("-"*80)

# Show some sample predictions
sample_size = 10
sample_indices = np.random.choice(len(y_test), sample_size, replace=False)

print("\nSample Predictions vs Actual Values:")
print("-"*60)
print(f"{'Actual':<15} {'Predicted':<15} {'Error':<15}")
print("-"*60)

for idx in sample_indices:
    actual = y_test.iloc[idx]
    predicted = y_pred_test[idx]
    error = actual - predicted
    print(f"{actual:<15.2f} {predicted:<15.2f} {error:<15.2f}")

print("\n" + "="*80)
print("✓ DEMAND FORECASTING MODEL COMPLETE!")
print("="*80)
print(f"\n📊 Final Model Performance (Test Set):")
print(f"   • R² Score: {test_r2:.4f}")
print(f"   • RMSE: {test_rmse:.2f}")
print(f"   • MAE: {test_mae:.2f}")
if TF_AVAILABLE:
    print(f"\n✓ Neural Network model ready for deployment!")
else:
    print(f"\n✓ Random Forest model ready for deployment!")
print("✓ All model files saved successfully!")
