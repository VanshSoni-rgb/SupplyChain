#!/usr/bin/env python
# ============================================================================
# DEPENDENCY INSTALLER WITH ERROR HANDLING
# Helps install packages with proper error messages
# ============================================================================

import subprocess
import sys

print("="*80)
print("SUPPLY CHAIN ANALYTICS - DEPENDENCY INSTALLER")
print("="*80)

# Core dependencies (required)
core_packages = [
    'pandas>=1.3.0',
    'numpy>=1.21.0',
    'matplotlib>=3.4.0',
    'seaborn>=0.11.0',
    'scikit-learn>=0.24.0',
    'openpyxl>=3.0.0'
]

# Optional dependencies
optional_packages = {
    'tensorflow': 'tensorflow>=2.10.0',
    'duckdb': 'duckdb>=0.8.0'
}

def install_package(package):
    """Install a single package"""
    try:
        print(f"\n📦 Installing {package}...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
        print(f"✓ {package} installed successfully")
        return True
    except subprocess.CalledProcessError:
        print(f"✗ Failed to install {package}")
        return False

def main():
    print("\n🚀 STEP 1: Installing Core Dependencies")
    print("-"*80)
    
    failed_core = []
    for package in core_packages:
        if not install_package(package):
            failed_core.append(package)
    
    if failed_core:
        print(f"\n⚠️ Some core packages failed to install: {failed_core}")
        print("Please install them manually: pip install " + " ".join(failed_core))
    else:
        print("\n✓ All core dependencies installed successfully!")
    
    print("\n\n🔧 STEP 2: Installing Optional Dependencies")
    print("-"*80)
    print("These are optional but recommended for full functionality\n")
    
    # Try TensorFlow
    print("\n📦 Attempting to install TensorFlow...")
    print("   (This may take a few minutes)")
    tf_success = install_package(optional_packages['tensorflow'])
    
    if not tf_success:
        print("\n⚠️ TensorFlow installation failed")
        print("   Don't worry! The project will use Random Forest instead")
        print("\n   To install TensorFlow manually, try:")
        print("   - Windows/Linux: pip install tensorflow")
        print("   - Mac M1/M2: pip install tensorflow-macos tensorflow-metal")
        print("   - Older systems: pip install tensorflow==2.10.0")
    
    # Try DuckDB
    print("\n📦 Attempting to install DuckDB...")
    duckdb_success = install_package(optional_packages['duckdb'])
    
    if not duckdb_success:
        print("\n⚠️ DuckDB installation failed")
        print("   Don't worry! The project will use pandas for SQL queries")
        print("\n   To install DuckDB manually: pip install duckdb")
    
    # Summary
    print("\n\n" + "="*80)
    print("INSTALLATION SUMMARY")
    print("="*80)
    
    print("\n✓ Core Dependencies: ", end="")
    if not failed_core:
        print("ALL INSTALLED")
    else:
        print(f"MISSING {len(failed_core)}")
    
    print("✓ TensorFlow: ", end="")
    print("INSTALLED" if tf_success else "NOT INSTALLED (will use Random Forest)")
    
    print("✓ DuckDB: ", end="")
    print("INSTALLED" if duckdb_success else "NOT INSTALLED (will use pandas)")
    
    print("\n" + "="*80)
    
    if not failed_core:
        print("✓ You're ready to run the project!")
        print("\nNext step: python run_complete_analysis.py")
    else:
        print("⚠️ Please install missing core dependencies first")
        print(f"\nRun: pip install {' '.join(failed_core)}")
    
    print("="*80)

if __name__ == "__main__":
    main()
