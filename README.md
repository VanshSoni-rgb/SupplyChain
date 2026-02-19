# 📊 Supply Chain Management Data Analytics & Machine Learning Project

A comprehensive end-to-end supply chain analytics solution for a Fashion & Beauty startup, featuring data analysis, machine learning demand forecasting, SQL business intelligence, and Power BI dashboarding.

---

## 🎯 Project Overview

This project demonstrates intermediate-level data analytics skills through a complete supply chain management analysis pipeline. It includes:

- **Data Cleaning & Preprocessing**: Handling missing values, duplicates, and data type conversions
- **Exploratory Data Analysis (EDA)**: Visual insights into supply chain performance
- **Feature Engineering**: Creating business-relevant metrics
- **Machine Learning**: Neural network model for demand forecasting
- **SQL Analysis**: Business intelligence queries using DuckDB
- **Power BI Dashboard**: Interactive visualizations for stakeholders

---

## 📁 Project Structure

```
supply-chain-analytics/
│
├── supply_chain_data.csv              # Raw dataset
├── supply_chain_data_cleaned.csv      # Cleaned dataset
│
├── supply_chain_analytics.py          # Main EDA script
├── ml_demand_forecasting.py           # ML model training
├── sql_analysis.py                    # SQL queries
├── generate_sample_data.py            # Sample data generator
├── run_complete_analysis.py           # Complete pipeline runner
│
├── demand_forecasting_model.h5        # Trained ML model
├── scaler.pkl                         # Feature scaler
├── feature_names.pkl                  # Feature names
│
├── outputs/                           # EDA visualizations
│   ├── revenue_by_product_type.png
│   ├── manufacturing_costs_by_supplier.png
│   ├── price_vs_manufacturing_cost.png
│   ├── defect_rate_distribution.png
│   ├── shipping_costs_by_carrier.png
│   ├── transportation_mode_distribution.png
│   ├── training_history.png
│   ├── actual_vs_predicted.png
│   └── prediction_errors.png
│
├── sql_results/                       # SQL query outputs
│   ├── 01_total_revenue.csv
│   ├── 02_revenue_by_location.csv
│   ├── 03_manufacturing_cost_by_supplier.csv
│   └── ... (10 query results)
│
├── PowerBI_Dashboard_Guide.md         # Power BI instructions
├── PROJECT_REPORT.md                  # Complete project report
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Power BI Desktop (for dashboard creation)

### Installation

1. **Clone or download this project**

2. **Install required packages**:

**Option 1: Automatic (Recommended)**
```bash
python install_dependencies.py
```

**Option 2: Manual**
```bash
pip install -r requirements.txt
```

**Note:** If TensorFlow or DuckDB fail to install, don't worry! The project will automatically use alternatives (Random Forest and pandas). See `TROUBLESHOOTING.md` for help.

3. **Generate sample data** (if you don't have the dataset):
```bash
python generate_sample_data.py
```

4. **Run complete analysis**:
```bash
python run_complete_analysis.py
```

Or run individual scripts:
```bash
python supply_chain_analytics.py
python sql_analysis.py
python ml_demand_forecasting.py
```

**Having issues?** Check `TROUBLESHOOTING.md` for solutions to common problems.

---

## 📊 Dataset Description

The dataset contains supply chain data for a Fashion & Beauty startup with the following fields:

| Column | Description |
|--------|-------------|
| Product type | Category of product (Skincare, Haircare, etc.) |
| SKU | Stock Keeping Unit identifier |
| Price | Product price ($) |
| Availability | Product availability (0/1) |
| Number of products sold | Sales quantity |
| Revenue generated | Total revenue ($) |
| Stock levels | Current inventory |
| Lead times | Time from order to delivery (days) |
| Order quantities | Order size |
| Shipping times | Delivery duration (days) |
| Shipping carriers | Logistics provider |
| Shipping costs | Shipping expenses ($) |
| Supplier name | Supplier identifier |
| Location | Geographic location |
| Production volumes | Manufacturing quantity |
| Manufacturing costs | Production expenses ($) |
| Defect rates | Quality defect percentage |
| Transportation modes | Transport method |
| Routes | Shipping route |
| Costs | Transportation costs ($) |
| Inspection results | Quality check outcome |

---

## 🔍 Analysis Components

### 1. Data Cleaning & EDA
**Script**: `supply_chain_analytics.py`

- Missing value handling
- Duplicate removal
- Feature engineering (Profit, Cost per Shipment, Inventory Gap)
- 6 key visualizations
- Business metrics summary

**Outputs**: 
- `supply_chain_data_cleaned.csv`
- Visualizations in `outputs/` folder

### 2. SQL Analysis
**Script**: `sql_analysis.py`

10 business intelligence queries:
1. Total Revenue
2. Revenue by Location
3. Manufacturing Cost by Supplier
4. Highest Defect Rates
5. Shipping Cost Summary
6. Product Performance
7. Inventory Analysis
8. Transportation Mode Analysis
9. Profitability Analysis
10. Lead Time Performance

**Outputs**: Query results in `sql_results/` folder

### 3. Machine Learning Model
**Script**: `ml_demand_forecasting.py`

- **Objective**: Predict number of products sold
- **Model**: Neural Network (Keras)
- **Architecture**: 
  - Dense(128, relu) + Dropout(0.2)
  - Dense(64, relu) + Dropout(0.2)
  - Dense(32, relu)
  - Dense(1)
- **Training**: 50 epochs, 80/20 train-test split
- **Evaluation**: MSE, RMSE, MAE, R²

**Outputs**:
- `demand_forecasting_model.h5`
- `scaler.pkl`
- `feature_names.pkl`
- Training visualizations

### 4. Power BI Dashboard
**Guide**: `PowerBI_Dashboard_Guide.md`

Three-page dashboard:
- **Page 1**: Executive Overview (KPIs, revenue charts)
- **Page 2**: Logistics Analysis (shipping, transportation)
- **Page 3**: Inventory & Quality (stock, defects)

---

## 📈 Key Findings

### Business Metrics
- Total revenue and profitability analysis
- Average lead times and shipping costs
- Defect rate monitoring
- Inventory gap identification

### Insights
- Product type performance comparison
- Supplier cost efficiency
- Shipping carrier optimization opportunities
- Quality control areas for improvement

---

## 🛠️ Technologies Used

- **Python**: Core programming language
- **Pandas & NumPy**: Data manipulation
- **Matplotlib & Seaborn**: Data visualization
- **Scikit-learn**: Data preprocessing
- **TensorFlow/Keras**: Deep learning
- **DuckDB**: SQL analysis
- **Power BI**: Interactive dashboards

---

## 📚 Documentation

- **MODEL_USAGE_GUIDE.md**: How to load and use the trained model
- **PowerBI_Dashboard_Guide.md**: Complete Power BI setup instructions
- **PROJECT_REPORT.md**: Detailed project documentation
- **TROUBLESHOOTING.md**: Solutions to common issues
- **FIXES_APPLIED.md**: Documentation of TensorFlow/DuckDB fixes
- **Code Comments**: Inline documentation in all scripts

---

## 🎓 Learning Outcomes

This project demonstrates:
- Data cleaning and preprocessing techniques
- Exploratory data analysis best practices
- Feature engineering for business metrics
- Neural network implementation
- SQL query writing for business intelligence
- Dashboard design principles
- End-to-end project workflow

---

## 📝 Usage Examples

### Load and Explore Data
```python
import pandas as pd

# Load cleaned data
df = pd.read_csv('supply_chain_data_cleaned.csv')

# View summary
print(df.describe())
print(df.info())
```

### Load and Use the Trained Model
```bash
# Use the helper script (easiest way)
python load_and_use_model.py

# Or inspect model details
python inspect_model.py
```

**Note:** The `.pkl` files are binary pickle files that contain Python objects. They cannot be opened in text editors. Use the provided scripts to load and use them. See `MODEL_USAGE_GUIDE.md` for detailed instructions.

### Make Predictions with Code
```python
import pickle
import pandas as pd

# Load model and scaler
with open('demand_forecasting_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

# Prepare new data and predict
# X_new = ... (your new data with required features)
# X_scaled = scaler.transform(X_new)
# predictions = model.predict(X_scaled)
```

### Run SQL Queries
```python
import duckdb
import pandas as pd

df = pd.read_csv('supply_chain_data_cleaned.csv')
con = duckdb.connect(':memory:')

result = con.execute("""
    SELECT 
        "Product type",
        SUM("Revenue generated") AS total_revenue
    FROM df
    GROUP BY "Product type"
    ORDER BY total_revenue DESC
""").df()

print(result)
```

---

## 🤝 Contributing

This is a student portfolio project. Feel free to:
- Fork the repository
- Modify for your own datasets
- Extend with additional analyses
- Improve visualizations

---

## 📧 Contact

**Project Type**: Academic Portfolio Project  
**Level**: Intermediate Data Analytics  
**Domain**: Supply Chain Management  
**Industry**: Fashion & Beauty

---

## 📄 License

This project is created for educational purposes.

---

## ⭐ Acknowledgments

- Dataset structure inspired by real-world supply chain systems
- Analysis techniques based on industry best practices
- Dashboard design follows Power BI guidelines

---

**Last Updated**: 2024  
**Version**: 1.0  
**Status**: Complete ✓

---
