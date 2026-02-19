# ============================================================================
# SUPPLY CHAIN MANAGEMENT DATA ANALYTICS & MACHINE LEARNING PROJECT
# Fashion & Beauty Startup Supply Chain Analysis
# ============================================================================

# Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("="*80)
print("SUPPLY CHAIN ANALYTICS PROJECT - FASHION & BEAUTY STARTUP")
print("="*80)

# ============================================================================
# STEP 1: DATA LOADING
# ============================================================================
print("\n📂 STEP 1: LOADING DATA")
print("-"*80)

# Load the dataset
df = pd.read_csv("supply_chain_data.csv")

print(f"✓ Dataset loaded successfully!")
print(f"✓ Total Records: {len(df)}")
print(f"✓ Total Columns: {len(df.columns)}")

# Display first few rows
print("\n📊 First 5 Rows of Dataset:")
print(df.head())

# Display dataset information
print("\n📋 Dataset Information:")
print(df.info())

# Display statistical summary
print("\n📈 Statistical Summary:")
print(df.describe())

# ============================================================================
# STEP 2: DATA CLEANING
# ============================================================================
print("\n\n🧹 STEP 2: DATA CLEANING")
print("-"*80)

# Check for missing values
print("\n🔍 Checking Missing Values:")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])

if missing_values.sum() == 0:
    print("✓ No missing values found!")
else:
    # Fill missing values with appropriate methods
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['float64', 'int64']:
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
    print("✓ Missing values handled!")

# Check for duplicates
print(f"\n🔍 Checking Duplicates:")
duplicates = df.duplicated().sum()
print(f"Total Duplicates: {duplicates}")

if duplicates > 0:
    df.drop_duplicates(inplace=True)
    print("✓ Duplicates removed!")
else:
    print("✓ No duplicates found!")

print(f"\n✓ Clean Dataset Shape: {df.shape}")

# ============================================================================
# STEP 3: FEATURE ENGINEERING
# ============================================================================
print("\n\n⚙️ STEP 3: FEATURE ENGINEERING")
print("-"*80)

# Create new features
print("\n🔧 Creating New Features:")

# 1. Profit Calculation
df['Profit'] = df['Revenue generated'] - df['Manufacturing costs']
print("✓ Created: Profit = Revenue - Manufacturing Costs")

# 2. Cost per Shipment
df['Cost_per_Shipment'] = df['Shipping costs'] / (df['Order quantities'] + 1)
print("✓ Created: Cost per Shipment")

# 3. Inventory Gap
df['Inventory_Gap'] = df['Stock levels'] - df['Order quantities']
print("✓ Created: Inventory Gap")

# 4. Revenue per Product
df['Revenue_per_Product'] = df['Revenue generated'] / (df['Number of products sold'] + 1)
print("✓ Created: Revenue per Product")

# 5. Profit Margin
df['Profit_Margin'] = (df['Profit'] / df['Revenue generated']) * 100
print("✓ Created: Profit Margin (%)")

# 6. Total Lead Time
df['Total_Lead_Time'] = df['Lead times'] + df['Shipping times']
print("✓ Created: Total Lead Time")

print(f"\n✓ Total Features Now: {len(df.columns)}")

# ============================================================================
# STEP 4: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================
print("\n\n📊 STEP 4: EXPLORATORY DATA ANALYSIS")
print("-"*80)

# Create output directory for plots
import os
if not os.path.exists('outputs'):
    os.makedirs('outputs')
    print("✓ Created 'outputs' folder for visualizations")

# EDA 1: Revenue by Product Type
print("\n📈 Creating Visualization 1: Revenue by Product Type")
plt.figure(figsize=(12, 6))
revenue_by_product = df.groupby('Product type')['Revenue generated'].sum().sort_values(ascending=False)
revenue_by_product.plot(kind='bar', color='steelblue')
plt.title('Total Revenue by Product Type', fontsize=16, fontweight='bold')
plt.xlabel('Product Type', fontsize=12)
plt.ylabel('Revenue Generated ($)', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('outputs/revenue_by_product_type.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: revenue_by_product_type.png")

# EDA 2: Manufacturing Costs by Supplier
print("\n📈 Creating Visualization 2: Manufacturing Costs by Supplier")
plt.figure(figsize=(12, 6))
cost_by_supplier = df.groupby('Supplier name')['Manufacturing costs'].mean().sort_values(ascending=False).head(10)
cost_by_supplier.plot(kind='barh', color='coral')
plt.title('Average Manufacturing Costs by Top 10 Suppliers', fontsize=16, fontweight='bold')
plt.xlabel('Average Manufacturing Cost ($)', fontsize=12)
plt.ylabel('Supplier Name', fontsize=12)
plt.tight_layout()
plt.savefig('outputs/manufacturing_costs_by_supplier.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: manufacturing_costs_by_supplier.png")

# EDA 3: Price vs Manufacturing Cost
print("\n📈 Creating Visualization 3: Price vs Manufacturing Cost")
plt.figure(figsize=(10, 6))
plt.scatter(df['Manufacturing costs'], df['Price'], alpha=0.5, color='green')
plt.title('Price vs Manufacturing Cost Analysis', fontsize=16, fontweight='bold')
plt.xlabel('Manufacturing Cost ($)', fontsize=12)
plt.ylabel('Price ($)', fontsize=12)
plt.tight_layout()
plt.savefig('outputs/price_vs_manufacturing_cost.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: price_vs_manufacturing_cost.png")

# EDA 4: Defect Rate Distribution
print("\n📈 Creating Visualization 4: Defect Rate Distribution")
plt.figure(figsize=(10, 6))
plt.hist(df['Defect rates'], bins=30, color='red', alpha=0.7, edgecolor='black')
plt.title('Distribution of Defect Rates', fontsize=16, fontweight='bold')
plt.xlabel('Defect Rate (%)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.tight_layout()
plt.savefig('outputs/defect_rate_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: defect_rate_distribution.png")

# EDA 5: Shipping Costs by Carrier
print("\n📈 Creating Visualization 5: Shipping Costs by Carrier")
plt.figure(figsize=(12, 6))
shipping_by_carrier = df.groupby('Shipping carriers')['Shipping costs'].mean().sort_values(ascending=False)
shipping_by_carrier.plot(kind='bar', color='purple')
plt.title('Average Shipping Costs by Carrier', fontsize=16, fontweight='bold')
plt.xlabel('Shipping Carrier', fontsize=12)
plt.ylabel('Average Shipping Cost ($)', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('outputs/shipping_costs_by_carrier.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: shipping_costs_by_carrier.png")

# EDA 6: Transportation Mode Frequency
print("\n📈 Creating Visualization 6: Transportation Mode Distribution")
plt.figure(figsize=(10, 6))
transport_counts = df['Transportation modes'].value_counts()
plt.pie(transport_counts, labels=transport_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Transportation Mode Distribution', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/transportation_mode_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: transportation_mode_distribution.png")

# EDA 7: Stock Levels vs Order Quantities
print("\n📈 Creating Visualization 7: Stock Levels vs Order Quantities")
plt.figure(figsize=(10, 6))
plt.scatter(df['Order quantities'], df['Stock levels'], alpha=0.6, color='orange')
plt.title('Stock Levels vs Order Quantities', fontsize=16, fontweight='bold')
plt.xlabel('Order Quantities', fontsize=12)
plt.ylabel('Stock Levels', fontsize=12)
plt.tight_layout()
plt.savefig('outputs/stock_vs_orders.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: stock_vs_orders.png")

# EDA 8: Production Volume by Location
print("\n📈 Creating Visualization 8: Production Volume by Location")
plt.figure(figsize=(12, 6))
prod_by_location = df.groupby('Location')['Production volumes'].sum().sort_values(ascending=False)
prod_by_location.plot(kind='bar', color='teal')
plt.title('Total Production Volume by Location', fontsize=16, fontweight='bold')
plt.xlabel('Location', fontsize=12)
plt.ylabel('Production Volume', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('outputs/production_by_location.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: production_by_location.png")

# Business Metrics Summary
print("\n\n📊 KEY BUSINESS METRICS:")
print("-"*80)
print(f"💰 Total Revenue: ${df['Revenue generated'].sum():,.2f}")
print(f"💵 Total Manufacturing Costs: ${df['Manufacturing costs'].sum():,.2f}")
print(f"💎 Total Profit: ${df['Profit'].sum():,.2f}")
print(f"📊 Average Profit Margin: {df['Profit_Margin'].mean():.2f}%")
print(f"📦 Average Stock Level: {df['Stock levels'].mean():.2f} units")
print(f"📋 Total Orders: {df['Order quantities'].sum():,.0f} units")
print(f"🛍️ Total Products Sold: {df['Number of products sold'].sum():,.0f} units")
print(f"🚚 Average Shipping Time: {df['Shipping times'].mean():.2f} days")
print(f"⏱️ Average Lead Time: {df['Lead times'].mean():.2f} days")
print(f"⚠️ Average Defect Rate: {df['Defect rates'].mean():.2f}%")
print(f"💲 Average Shipping Cost: ${df['Shipping costs'].mean():.2f}")

print("\n✓ EDA Complete! All visualizations saved in 'outputs' folder.")

# Save cleaned dataset
df.to_csv('supply_chain_data_cleaned.csv', index=False)
print("\n✓ Cleaned dataset saved as: supply_chain_data_cleaned.csv")

print("\n" + "="*80)
print("✓ DATA ANALYSIS COMPLETE!")
print("="*80)
