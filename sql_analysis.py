# ============================================================================
# SQL ANALYSIS USING PANDAS (DuckDB Optional)
# Business Intelligence Queries for Supply Chain Data
# ============================================================================

import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Try to import DuckDB, fall back to pandas if not available
try:
    import duckdb
    USE_DUCKDB = True
    print("✓ Using DuckDB for SQL queries")
except ImportError:
    USE_DUCKDB = False
    print("⚠️ DuckDB not installed. Using pandas for queries (works fine!)")
    print("   To install DuckDB later: pip install duckdb\n")

print("="*80)
print("SQL ANALYSIS - SUPPLY CHAIN BUSINESS INTELLIGENCE")
print("="*80)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\n📂 STEP 1: LOADING DATA")
print("-"*80)

# Load the cleaned dataset
df = pd.read_csv('supply_chain_data_cleaned.csv')
print(f"✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Initialize DuckDB connection if available
if USE_DUCKDB:
    con = duckdb.connect(':memory:')
    print("✓ DuckDB connection established")

# ============================================================================
# STEP 2: BUSINESS INTELLIGENCE QUERIES
# ============================================================================
print("\n\n📊 STEP 2: EXECUTING SQL QUERIES")
print("="*80)

# Query 1: Total Revenue
print("\n" + "="*80)
print("QUERY 1: TOTAL REVENUE GENERATED")
print("="*80)

if USE_DUCKDB:
    query1 = """
    SELECT 
        SUM("Revenue generated") AS total_revenue,
        COUNT(*) AS total_transactions,
        AVG("Revenue generated") AS avg_revenue_per_transaction
    FROM df
    """
    print("\nSQL Query:")
    print(query1)
    result1 = con.execute(query1).df()
else:
    print("\nUsing pandas equivalent")
    result1 = pd.DataFrame({
        'total_revenue': [df['Revenue generated'].sum()],
        'total_transactions': [len(df)],
        'avg_revenue_per_transaction': [df['Revenue generated'].mean()]
    })

print("\nResults:")
print(result1.to_string(index=False))

# Query 2: Revenue by Location
print("\n" + "="*80)
print("QUERY 2: REVENUE BY LOCATION")
print("="*80)

if USE_DUCKDB:
    query2 = """
    SELECT 
        Location,
        SUM("Revenue generated") AS total_revenue,
        COUNT(*) AS num_transactions,
        AVG("Revenue generated") AS avg_revenue
    FROM df
    GROUP BY Location
    ORDER BY total_revenue DESC
    LIMIT 10
    """
    print("\nSQL Query:")
    print(query2)
    result2 = con.execute(query2).df()
else:
    print("\nUsing pandas equivalent")
    result2 = df.groupby('Location').agg({
        'Revenue generated': ['sum', 'count', 'mean']
    }).reset_index()
    result2.columns = ['Location', 'total_revenue', 'num_transactions', 'avg_revenue']
    result2 = result2.sort_values('total_revenue', ascending=False).head(10)

print("\nResults:")
print(result2.to_string(index=False))

# Query 3: Average Manufacturing Cost by Supplier
print("\n" + "="*80)
print("QUERY 3: AVERAGE MANUFACTURING COST BY SUPPLIER")
print("="*80)

if USE_DUCKDB:
    query3 = """
    SELECT 
        "Supplier name",
        AVG("Manufacturing costs") AS avg_manufacturing_cost,
        COUNT(*) AS num_products,
        SUM("Production volumes") AS total_production_volume
    FROM df
    GROUP BY "Supplier name"
    ORDER BY avg_manufacturing_cost DESC
    LIMIT 10
    """
    print("\nSQL Query:")
    print(query3)
    result3 = con.execute(query3).df()
else:
    print("\nUsing pandas equivalent")
    grouped = df.groupby('Supplier name')
    result3 = pd.DataFrame({
        'Supplier name': grouped['Manufacturing costs'].mean().index,
        'avg_manufacturing_cost': grouped['Manufacturing costs'].mean().values,
        'num_products': grouped.size().values,
        'total_production_volume': grouped['Production volumes'].sum().values
    }).sort_values('avg_manufacturing_cost', ascending=False).head(10)

print("\nResults:")
print(result3.to_string(index=False))

# Query 4: Products with Highest Defect Rates
print("\n" + "="*80)
print("QUERY 4: PRODUCTS WITH HIGHEST DEFECT RATES")
print("="*80)

if USE_DUCKDB:
    query4 = """
    SELECT 
        "Product type",
        AVG("Defect rates") AS avg_defect_rate,
        COUNT(*) AS num_records,
        AVG("Manufacturing costs") AS avg_cost
    FROM df
    GROUP BY "Product type"
    ORDER BY avg_defect_rate DESC
    LIMIT 10
    """
    print("\nSQL Query:")
    print(query4)
    result4 = con.execute(query4).df()
else:
    print("\nUsing pandas equivalent")
    grouped = df.groupby('Product type')
    result4 = pd.DataFrame({
        'Product type': grouped['Defect rates'].mean().index,
        'avg_defect_rate': grouped['Defect rates'].mean().values,
        'num_records': grouped.size().values,
        'avg_cost': grouped['Manufacturing costs'].mean().values
    }).sort_values('avg_defect_rate', ascending=False).head(10)

print("\nResults:")
print(result4.to_string(index=False))

# Query 5: Shipping Cost Summary by Carrier
print("\n" + "="*80)
print("QUERY 5: SHIPPING COST SUMMARY BY CARRIER")
print("="*80)

if USE_DUCKDB:
    query5 = """
    SELECT 
        "Shipping carriers",
        SUM("Shipping costs") AS total_shipping_cost,
        AVG("Shipping costs") AS avg_shipping_cost,
        AVG("Shipping times") AS avg_shipping_time,
        COUNT(*) AS num_shipments
    FROM df
    GROUP BY "Shipping carriers"
    ORDER BY total_shipping_cost DESC
    """
    print("\nSQL Query:")
    print(query5)
    result5 = con.execute(query5).df()
else:
    print("\nUsing pandas equivalent")
    grouped = df.groupby('Shipping carriers')
    result5 = pd.DataFrame({
        'Shipping carriers': grouped['Shipping costs'].sum().index,
        'total_shipping_cost': grouped['Shipping costs'].sum().values,
        'avg_shipping_cost': grouped['Shipping costs'].mean().values,
        'avg_shipping_time': grouped['Shipping times'].mean().values,
        'num_shipments': grouped.size().values
    }).sort_values('total_shipping_cost', ascending=False)

print("\nResults:")
print(result5.to_string(index=False))

# Query 6: Product Performance Analysis
print("\n" + "="*80)
print("QUERY 6: PRODUCT PERFORMANCE ANALYSIS")
print("="*80)

if USE_DUCKDB:
    query6 = """
    SELECT 
        "Product type",
        SUM("Number of products sold") AS total_sold,
        SUM("Revenue generated") AS total_revenue,
        AVG(Price) AS avg_price,
        SUM(Profit) AS total_profit
    FROM df
    GROUP BY "Product type"
    ORDER BY total_revenue DESC
    """
    print("\nSQL Query:")
    print(query6)
    result6 = con.execute(query6).df()
else:
    print("\nUsing pandas equivalent")
    grouped = df.groupby('Product type')
    result6 = pd.DataFrame({
        'Product type': grouped['Number of products sold'].sum().index,
        'total_sold': grouped['Number of products sold'].sum().values,
        'total_revenue': grouped['Revenue generated'].sum().values,
        'avg_price': grouped['Price'].mean().values,
        'total_profit': grouped['Profit'].sum().values
    }).sort_values('total_revenue', ascending=False)

print("\nResults:")
print(result6.to_string(index=False))

# Query 7: Inventory Analysis
print("\n" + "="*80)
print("QUERY 7: INVENTORY ANALYSIS")
print("="*80)

if USE_DUCKDB:
    query7 = """
    SELECT 
        "Product type",
        AVG("Stock levels") AS avg_stock_level,
        AVG("Order quantities") AS avg_order_quantity,
        AVG("Inventory_Gap") AS avg_inventory_gap,
        AVG("Lead times") AS avg_lead_time
    FROM df
    GROUP BY "Product type"
    ORDER BY avg_stock_level DESC
    """
    print("\nSQL Query:")
    print(query7)
    result7 = con.execute(query7).df()
else:
    print("\nUsing pandas equivalent")
    grouped = df.groupby('Product type')
    result7 = pd.DataFrame({
        'Product type': grouped['Stock levels'].mean().index,
        'avg_stock_level': grouped['Stock levels'].mean().values,
        'avg_order_quantity': grouped['Order quantities'].mean().values,
        'avg_inventory_gap': grouped['Inventory_Gap'].mean().values,
        'avg_lead_time': grouped['Lead times'].mean().values
    }).sort_values('avg_stock_level', ascending=False)

print("\nResults:")
print(result7.to_string(index=False))

# Query 8: Transportation Mode Analysis
print("\n" + "="*80)
print("QUERY 8: TRANSPORTATION MODE ANALYSIS")
print("="*80)

if USE_DUCKDB:
    query8 = """
    SELECT 
        "Transportation modes",
        COUNT(*) AS frequency,
        AVG(Costs) AS avg_transportation_cost,
        AVG("Shipping times") AS avg_shipping_time
    FROM df
    GROUP BY "Transportation modes"
    ORDER BY frequency DESC
    """
    print("\nSQL Query:")
    print(query8)
    result8 = con.execute(query8).df()
else:
    print("\nUsing pandas equivalent")
    grouped = df.groupby('Transportation modes')
    result8 = pd.DataFrame({
        'Transportation modes': grouped.size().index,
        'frequency': grouped.size().values,
        'avg_transportation_cost': grouped['Costs'].mean().values,
        'avg_shipping_time': grouped['Shipping times'].mean().values
    }).sort_values('frequency', ascending=False)

print("\nResults:")
print(result8.to_string(index=False))

# Query 9: Profitability Analysis
print("\n" + "="*80)
print("QUERY 9: PROFITABILITY ANALYSIS BY PRODUCT")
print("="*80)

if USE_DUCKDB:
    query9 = """
    SELECT 
        "Product type",
        SUM(Profit) AS total_profit,
        AVG(Profit) AS avg_profit_per_unit,
        SUM("Revenue generated") AS total_revenue,
        SUM("Manufacturing costs") AS total_manufacturing_cost,
        (SUM(Profit) / SUM("Revenue generated") * 100) AS profit_margin_pct
    FROM df
    GROUP BY "Product type"
    ORDER BY total_profit DESC
    """
    print("\nSQL Query:")
    print(query9)
    result9 = con.execute(query9).df()
else:
    print("\nUsing pandas equivalent")
    grouped = df.groupby('Product type')
    result9 = pd.DataFrame({
        'Product type': grouped['Profit'].sum().index,
        'total_profit': grouped['Profit'].sum().values,
        'avg_profit_per_unit': grouped['Profit'].mean().values,
        'total_revenue': grouped['Revenue generated'].sum().values,
        'total_manufacturing_cost': grouped['Manufacturing costs'].sum().values
    })
    result9['profit_margin_pct'] = (result9['total_profit'] / result9['total_revenue'] * 100)
    result9 = result9.sort_values('total_profit', ascending=False)

print("\nResults:")
print(result9.to_string(index=False))

# Query 10: Lead Time Performance
print("\n" + "="*80)
print("QUERY 10: LEAD TIME PERFORMANCE BY SUPPLIER")
print("="*80)

if USE_DUCKDB:
    query10 = """
    SELECT 
        "Supplier name",
        AVG("Lead times") AS avg_lead_time,
        MIN("Lead times") AS min_lead_time,
        MAX("Lead times") AS max_lead_time,
        COUNT(*) AS num_orders
    FROM df
    GROUP BY "Supplier name"
    ORDER BY avg_lead_time ASC
    LIMIT 10
    """
    print("\nSQL Query:")
    print(query10)
    result10 = con.execute(query10).df()
else:
    print("\nUsing pandas equivalent")
    grouped = df.groupby('Supplier name')
    result10 = pd.DataFrame({
        'Supplier name': grouped['Lead times'].mean().index,
        'avg_lead_time': grouped['Lead times'].mean().values,
        'min_lead_time': grouped['Lead times'].min().values,
        'max_lead_time': grouped['Lead times'].max().values,
        'num_orders': grouped.size().values
    }).sort_values('avg_lead_time').head(10)

print("\nResults:")
print(result10.to_string(index=False))

# ============================================================================
# STEP 3: SAVE QUERY RESULTS
# ============================================================================
print("\n\n💾 STEP 3: SAVING QUERY RESULTS")
print("="*80)

import os
if not os.path.exists('sql_results'):
    os.makedirs('sql_results')

# Save all results to CSV
result1.to_csv('sql_results/01_total_revenue.csv', index=False)
result2.to_csv('sql_results/02_revenue_by_location.csv', index=False)
result3.to_csv('sql_results/03_manufacturing_cost_by_supplier.csv', index=False)
result4.to_csv('sql_results/04_highest_defect_rates.csv', index=False)
result5.to_csv('sql_results/05_shipping_cost_summary.csv', index=False)
result6.to_csv('sql_results/06_product_performance.csv', index=False)
result7.to_csv('sql_results/07_inventory_analysis.csv', index=False)
result8.to_csv('sql_results/08_transportation_mode_analysis.csv', index=False)
result9.to_csv('sql_results/09_profitability_analysis.csv', index=False)
result10.to_csv('sql_results/10_lead_time_performance.csv', index=False)

print("\n✓ All query results saved in 'sql_results' folder")

# ============================================================================
# STEP 4: EXECUTIVE SUMMARY
# ============================================================================
print("\n\n📊 EXECUTIVE SUMMARY")
print("="*80)

if USE_DUCKDB:
    summary_query = """
    SELECT 
        COUNT(DISTINCT "Product type") AS total_product_types,
        COUNT(DISTINCT "Supplier name") AS total_suppliers,
        COUNT(DISTINCT Location) AS total_locations,
        COUNT(DISTINCT "Shipping carriers") AS total_carriers,
        SUM("Revenue generated") AS total_revenue,
        SUM(Profit) AS total_profit,
        AVG("Defect rates") AS avg_defect_rate,
        AVG("Lead times") AS avg_lead_time,
        AVG("Shipping times") AS avg_shipping_time
    FROM df
    """
    summary = con.execute(summary_query).df()
else:
    summary = pd.DataFrame({
        'total_product_types': [df['Product type'].nunique()],
        'total_suppliers': [df['Supplier name'].nunique()],
        'total_locations': [df['Location'].nunique()],
        'total_carriers': [df['Shipping carriers'].nunique()],
        'total_revenue': [df['Revenue generated'].sum()],
        'total_profit': [df['Profit'].sum()],
        'avg_defect_rate': [df['Defect rates'].mean()],
        'avg_lead_time': [df['Lead times'].mean()],
        'avg_shipping_time': [df['Shipping times'].mean()]
    })

print("\n📈 KEY BUSINESS METRICS:")
print("-"*80)
for col in summary.columns:
    value = summary[col].values[0]
    if 'total' in col.lower() or 'revenue' in col.lower() or 'profit' in col.lower():
        print(f"{col}: {value:,.2f}")
    else:
        print(f"{col}: {value:.2f}")

# Close connection if using DuckDB
if USE_DUCKDB:
    con.close()

print("\n" + "="*80)
print("✓ SQL ANALYSIS COMPLETE!")
print("="*80)
print("\n✓ All query results saved in 'sql_results' folder")
print("✓ Ready for Power BI integration")
