# ============================================================================
# SAMPLE DATA GENERATOR
# Generate realistic supply chain data for testing
# ============================================================================

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

print("="*80)
print("GENERATING SAMPLE SUPPLY CHAIN DATA")
print("="*80)

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Define data parameters
num_records = 1000

# Define categories
product_types = ['Skincare', 'Haircare', 'Cosmetics', 'Fragrances', 'Accessories']
suppliers = ['Supplier A', 'Supplier B', 'Supplier C', 'Supplier D', 'Supplier E', 
             'Supplier F', 'Supplier G', 'Supplier H']
locations = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Miami', 
             'Seattle', 'Boston', 'Atlanta', 'Dallas', 'San Francisco']
shipping_carriers = ['FedEx', 'UPS', 'DHL', 'USPS', 'Amazon Logistics']
transportation_modes = ['Air', 'Road', 'Rail', 'Sea']
routes = ['Route A', 'Route B', 'Route C', 'Route D', 'Route E']
inspection_results = ['Pass', 'Pass', 'Pass', 'Fail']  # 75% pass rate

print(f"\n Generating {num_records} records...")

# Generate data
data = {
    'Product type': np.random.choice(product_types, num_records),
    'SKU': [f'SKU-{str(i).zfill(6)}' for i in range(1, num_records + 1)],
    'Price': np.random.uniform(10, 200, num_records).round(2),
    'Availability': np.random.randint(0, 2, num_records),
    'Number of products sold': np.random.randint(10, 500, num_records),
    'Stock levels': np.random.randint(50, 1000, num_records),
    'Lead times': np.random.uniform(1, 30, num_records).round(2),
    'Order quantities': np.random.randint(20, 600, num_records),
    'Shipping times': np.random.uniform(1, 15, num_records).round(2),
    'Shipping carriers': np.random.choice(shipping_carriers, num_records),
    'Shipping costs': np.random.uniform(5, 100, num_records).round(2),
    'Supplier name': np.random.choice(suppliers, num_records),
    'Location': np.random.choice(locations, num_records),
    'Production volumes': np.random.randint(100, 5000, num_records),
    'Manufacturing costs': np.random.uniform(5, 150, num_records).round(2),
    'Defect rates': np.random.uniform(0, 10, num_records).round(2),
    'Transportation modes': np.random.choice(transportation_modes, num_records),
    'Routes': np.random.choice(routes, num_records),
    'Costs': np.random.uniform(50, 500, num_records).round(2),
    'Inspection results': np.random.choice(inspection_results, num_records)
}

# Create DataFrame
df = pd.DataFrame(data)

# Calculate Revenue generated
df['Revenue generated'] = (df['Number of products sold'] * df['Price']).round(2)

# Reorder columns
column_order = [
    'Product type', 'SKU', 'Price', 'Availability', 'Number of products sold',
    'Revenue generated', 'Stock levels', 'Lead times', 'Order quantities',
    'Shipping times', 'Shipping carriers', 'Shipping costs', 'Supplier name',
    'Location', 'Production volumes', 'Manufacturing costs', 'Defect rates',
    'Transportation modes', 'Routes', 'Costs', 'Inspection results'
]

df = df[column_order]

# Save to CSV
df.to_csv('supply_chain_data.csv', index=False)

print(f"✓ Generated {len(df)} records")
print(f"✓ Saved as: supply_chain_data.csv")

# Display summary
print("\n DATA SUMMARY:")
print("-"*80)
print(f"Total Records: {len(df)}")
print(f"Total Columns: {len(df.columns)}")
print(f"\nProduct Types: {df['Product type'].nunique()}")
print(f"Suppliers: {df['Supplier name'].nunique()}")
print(f"Locations: {df['Location'].nunique()}")
print(f"Shipping Carriers: {df['Shipping carriers'].nunique()}")
print(f"\nTotal Revenue: ${df['Revenue generated'].sum():,.2f}")
print(f"Average Price: ${df['Price'].mean():.2f}")
print(f"Total Products Sold: {df['Number of products sold'].sum():,}")

print("\n✓ Sample data generation complete!")
print("="*80)
