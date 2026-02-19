#!/usr/bin/env python
# ============================================================================
# INSTALLATION TEST SCRIPT
# Quickly verify that your setup is working
# ============================================================================

import sys

print("="*80)
print("SUPPLY CHAIN ANALYTICS - INSTALLATION TEST")
print("="*80)

# Test Python version
print("\n🐍 Testing Python Version...")
print(f"   Python {sys.version}")
if sys.version_info >= (3, 8):
    print("   ✓ Python version OK (3.8+)")
else:
    print("   ✗ Python version too old. Need 3.8+")
    sys.exit(1)

# Test core packages
print("\n📦 Testing Core Packages...")
core_packages = {
    'pandas': 'Data manipulation',
    'numpy': 'Numerical operations',
    'matplotlib': 'Visualization',
    'seaborn': 'Statistical plots',
    'sklearn': 'Machine learning'
}

failed_core = []
for package, description in core_packages.items():
    try:
        __import__(package)
        print(f"   ✓ {package:<15} - {description}")
    except ImportError:
        print(f"   ✗ {package:<15} - NOT INSTALLED")
        failed_core.append(package)

# Test optional packages
print("\n🔧 Testing Optional Packages...")
optional_packages = {
    'tensorflow': 'Deep learning (optional - will use Random Forest if missing)',
    'duckdb': 'SQL queries (optional - will use pandas if missing)'
}

for package, description in optional_packages.items():
    try:
        __import__(package)
        print(f"   ✓ {package:<15} - {description}")
    except ImportError:
        print(f"   ⚠ {package:<15} - NOT INSTALLED")
        print(f"      {description}")

# Test file existence
print("\n📁 Testing Project Files...")
import os

required_files = [
    'generate_sample_data.py',
    'supply_chain_analytics.py',
    'sql_analysis.py',
    'ml_demand_forecasting.py',
    'run_complete_analysis.py',
    'requirements.txt',
    'README.md'
]

missing_files = []
for file in required_files:
    if os.path.exists(file):
        print(f"   ✓ {file}")
    else:
        print(f"   ✗ {file} - MISSING")
        missing_files.append(file)

# Test data file
print("\n📊 Testing Data File...")
if os.path.exists('supply_chain_data.csv'):
    print("   ✓ supply_chain_data.csv exists")
else:
    print("   ⚠ supply_chain_data.csv not found")
    print("      Run: python generate_sample_data.py")

# Summary
print("\n" + "="*80)
print("TEST SUMMARY")
print("="*80)

if failed_core:
    print("\n❌ INSTALLATION INCOMPLETE")
    print(f"\n   Missing core packages: {', '.join(failed_core)}")
    print(f"\n   Install them with:")
    print(f"   pip install {' '.join(failed_core)}")
    print("\n   Or run: python install_dependencies.py")
elif missing_files:
    print("\n⚠️ SOME FILES MISSING")
    print(f"\n   Missing files: {', '.join(missing_files)}")
    print("\n   Please download the complete project")
else:
    print("\n✅ INSTALLATION SUCCESSFUL!")
    print("\n   All core packages installed ✓")
    print("   All project files present ✓")
    print("\n   You're ready to run the project!")
    print("\n   Next step:")
    if not os.path.exists('supply_chain_data.csv'):
        print("   1. python generate_sample_data.py")
        print("   2. python run_complete_analysis.py")
    else:
        print("   python run_complete_analysis.py")

print("\n" + "="*80)
