# 🚗 Vehicle Sales Price Prediction - Production ML System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live_Demo-FF4B4B.svg)](https://vehicle-sales-prediction.streamlit.app)
[![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-20BEFF.svg)](https://www.kaggle.com/datasets/syedanwarafridi/vehicle-sales-data)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **End-to-end production ML system achieving 96.82% R² accuracy in vehicle price prediction**

**[🚀 Live Demo](https://vehicle-sales-prediction.streamlit.app)** | [📊 Notebooks](notebooks/) | [📖 Documentation](#documentation)

<div align="center">
  <img src="images/dashboard_preview.png" alt="Dashboard Preview" width="800"/>
</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Live Demo](#-live-demo)
- [Architecture](#-architecture)
- [Performance Metrics](#-performance-metrics)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Technical Stack](#-technical-stack)
- [Methodology](#-methodology)
- [Results & Insights](#-results--insights)
- [Production Features](#-production-features)
- [Documentation](#-documentation)
- [Contributing](#-contributing)

---

## 🎯 Overview

A **production-grade machine learning system** that predicts vehicle selling prices with exceptional accuracy. Built following industry best practices, this project demonstrates the complete ML lifecycle from data exploration to production deployment with monitoring.

### Business Impact
- 📈 **96.82% R² accuracy** - Explains 96.82% of price variance
- 💰 **$887 average error** on $13,611 mean price (6.5% error rate)
- 🚗 **558,825 vehicles** analyzed across 64 US states
- ⚡ **Real-time predictions** via interactive Streamlit dashboard

### What Makes This Special
- ✅ **Complete ML Pipeline**: Data → Model → Deployment → Monitoring
- ✅ **Production Ready**: Drift detection, monitoring, alerting
- ✅ **Model Explainability**: SHAP-like analysis, fairness assessment
- ✅ **Best Practices**: Hyperparameter tuning, cross-validation, proper splits

---

## ✨ Key Features

### 🤖 Advanced ML Pipeline
- **XGBoost** model optimized with Optuna (50 Bayesian trials)
- **17 engineered features** including age-odometer interactions
- **Prevented data leakage** through rigorous validation
- **Production preprocessing** with label encoding pipeline

### 📊 Interactive Dashboard
- **Real-time predictions** with confidence scoring
- **Feature importance** visualization
- **Market analysis** by state, make, body type
- **Sample predictions** showcase across price ranges

### 🔍 Model Explainability
- Feature importance analysis (MMR: 61.7%, Body: 9.0%)
- Performance segmentation (by state, make, vehicle age)
- Error analysis and diagnostics
- Fairness assessment across demographics

### 📈 Production Monitoring
- **Statistical drift detection** (KS test, PSI, Chi-square)
- **Baseline profiling** for 17 features
- **3-tier alert system** (HIGH/MEDIUM/LOW severity)
- **Automated retraining** triggers

---

## 🚀 Live Demo

**Try the interactive dashboard:** [https://vehicle-sales-prediction.streamlit.app](https://vehicle-sales-prediction.streamlit.app)

Features:
- Enter vehicle details (year, make, model, mileage, etc.)
- Get instant price predictions
- Compare with market values (MMR)
- Explore model insights and performance

---

## 🏗️ Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                     DATA LAYER                              │
│  Kaggle Dataset (558K records) → Cleaning → Feature Eng.   │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                   MODELING LAYER                            │
│  Train/Val/Test Split → Hyperparameter Tuning (Optuna)     │
│  Model Selection → XGBoost (Best: R²=0.968)                │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                EXPLAINABILITY LAYER                         │
│  Feature Importance → SHAP Analysis → Fairness Check       │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                 DEPLOYMENT LAYER                            │
│  Streamlit Dashboard → Real-time Predictions                │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                 MONITORING LAYER                            │
│  Drift Detection → Alerting → Retraining Triggers          │
└─────────────────────────────────────────────────────────────┘
```

---

## 🏆 Performance Metrics

### Overall Model Performance

| Metric | Train | Validation | Test |
|--------|-------|------------|------|
| **MAE** | $709 | $887 | $887 |
| **R²** | 0.985 | 0.975 | **0.968** |
| **MAPE** | 13.4% | 12.4% | 12.3% |
| **RMSE** | $1,202 | $1,542 | $1,542 |

### Accuracy by Segment

| Segment | Best Performance | MAPE |
|---------|------------------|------|
| **Price Range** | $30k-$50k | 3.19% |
| **State** | Pennsylvania | 7.05% |
| **Make** | BMW | 7.61% |
| **Vehicle Age** | 2-5 years | 5.85% |

### Feature Importance (Top 5)

1. **MMR** (61.7%) - Manheim Market Report value
2. **Body Type** (9.0%) - Vehicle category
3. **Age-Odometer** (8.5%) - Wear indicator
4. **Make** (5.9%) - Manufacturer
5. **Model** (3.0%) - Specific model

---

## ⚡ Quick Start

### Prerequisites
```bash
Python 3.8+
pip
8GB+ RAM recommended
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/YOUR-USERNAME/vehicle-sales-prediction.git
cd vehicle-sales-prediction
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download dataset**
- Visit: [Kaggle Dataset](https://www.kaggle.com/datasets/syedanwarafridi/vehicle-sales-data)
- Download `car_prices.csv`
- Place in: `data/raw/car_prices.csv`

4. **Run notebooks** (optional - pre-trained model included)
```bash
jupyter notebook notebooks/
```
Execute notebooks 00-05 in sequence.

5. **Launch dashboard**
```bash
cd app
streamlit run app.py
```
Open browser to: `http://localhost:8501`

### Quick Prediction (Python API)
```python
from app.predictor import VehiclePricePredictor

# Initialize
predictor = VehiclePricePredictor()

# Prepare vehicle data
vehicle = {
    'year': 2015,
    'make': 'Toyota',
    'model_grouped': 'Camry',
    'body': 'Sedan',
    'transmission': 'Automatic',
    'odometer': 50000,
    'condition': 35.0,
    'state': 'Ca',
    'color': 'Black',
    'interior': 'Black',
    'seller_grouped': 'Other_Seller',
    'trim_grouped': 'Se',
    'mmr': 12000.0
}

# Predict
result = predictor.predict(vehicle)
print(f"Predicted Price: ${result['predicted_price']:,.2f}")
# Output: Predicted Price: $14,250.00
```

---

## 📁 Project Structure
```
vehicle-sales-prediction/
│
├── notebooks/              # 6 comprehensive Jupyter notebooks
│   ├── 00_data_overview.ipynb           # EDA & validation
│   ├── 01_data_cleaning.ipynb           # Preprocessing pipeline
│   ├── 02_modeling.ipynb                # Model training & tuning
│   ├── 03_explainability.ipynb          # Feature importance & fairness
│   ├── 04_dashboard_prep.ipynb          # Deployment preparation
│   └── 05_monitoring.ipynb              # Drift detection system
│
├── app/                    # Production Streamlit dashboard
│   ├── app.py             # Main dashboard application
│   ├── predictor.py       # Prediction interface class
│   └── artifacts/         # Model artifacts (pkl, json)
│
├── models/                 # Trained models & metadata
│   ├── final/
│   │   ├── xgboost_optimized.pkl       # Final model
│   │   ├── model_metadata.pkl          # Performance metrics
│   │   └── feature_importance.csv      # Feature rankings
│   └── preprocessing/
│       └── label_encoders.pkl          # Categorical encoders
│
├── data/                   # Data directory (download from Kaggle)
│   ├── raw/               # Original dataset
│   ├── interim/           # Intermediate processing
│   └── processed/         # Final cleaned data
│
├── artifacts/              # Analysis outputs
│   └── explainability/    # Visualizations & reports
│
├── monitoring/             # Production monitoring
│   ├── baseline_profiles.pkl          # Distribution baselines
│   ├── monitoring_config.json         # Alert configurations
│   └── drift_monitoring_dashboard.png # Monitoring visualizations
│
└── docs/                   # Additional documentation
```

---

## 🛠️ Technical Stack

| Category | Technologies |
|----------|-------------|
| **ML Framework** | XGBoost, Scikit-learn, LightGBM, CatBoost |
| **Optimization** | Optuna (Bayesian hyperparameter tuning) |
| **Dashboard** | Streamlit, Plotly, Matplotlib, Seaborn |
| **Data Processing** | Pandas, NumPy |
| **Monitoring** | Custom drift detection (SciPy stats) |
| **Development** | Jupyter, Python 3.8+ |

---

## 📚 Methodology

### 1. Data Collection & Exploration
- **Dataset**: 558,837 vehicle sales records (2014-2015)
- **Source**: [Kaggle - Vehicle Sales Data](https://www.kaggle.com/datasets/syedanwarafridi/vehicle-sales-data)
- **Features**: 16 original attributes
- **EDA**: Distributions, correlations, outlier detection

### 2. Data Preprocessing
- **Cleaning**: Removed 12 records with missing target (99.998% retention)
- **Standardization**: 87 → 47 body types, fixed transmission errors
- **Outlier Treatment**: Capped 81 odometer values at 500k miles
- **Feature Engineering**: Created 5 new features
  - `vehicle_age` = 2015 - year
  - `log_odometer` = log(odometer + 1)
  - `age_odo_interaction` = age × odometer / 10000
  - `has_date` = date availability flag
- **Encoding**: Label encoding for tree-based models
- **Dimensionality Reduction**: Seller (14k→101), Model (852→201)

### 3. Model Development
- **Train/Val/Test Split**: 70/15/15
- **Models Tested**: 6 algorithms
  - Linear Regression (baseline)
  - Ridge Regression
  - Random Forest
  - LightGBM
  - **XGBoost** ✓ (selected)
  - CatBoost

### 4. Hyperparameter Optimization
- **Tool**: Optuna (Bayesian optimization)
- **Trials**: 50 iterations
- **Objective**: Minimize MAE
- **Best Parameters**:
  - Trees: 448
  - Learning rate: 0.045
  - Max depth: 12
  - Min child weight: 6
  - Subsample: 0.85

### 5. Model Evaluation
- **Metrics**: MAE, RMSE, R², MAPE
- **Cross-Validation**: 5-fold CV
- **Test Performance**: R² = 0.9682
- **Generalization**: Val/Test difference < $1

### 6. Explainability Analysis
- Feature importance extraction
- Performance by segment (state, make, age, price range)
- Error distribution analysis
- Fairness assessment

### 7. Production Deployment
- Streamlit dashboard development
- API creation (`VehiclePricePredictor` class)
- Artifact packaging
- Documentation

### 8. Monitoring System
- Baseline distribution profiling
- Statistical drift detection
- Alert rule configuration
- Drift simulation testing

---

## 📊 Results & Insights

### Key Findings

1. **MMR Dominates Predictions** (61.7% importance)
   - Market reference value is strongest predictor
   - Dealers should focus on accurate MMR assessment

2. **Vehicle Depreciation Patterns**
   - Average: ~$500/year depreciation
   - Accelerates after 8 years
   - Varies significantly by make (BMW holds value better)

3. **Mileage Impact**
   - Each 10k miles reduces price by ~$300
   - Effect varies by vehicle age (newer cars more sensitive)

4. **Regional Price Variance**
   - Texas: 39% MAPE (highest variance)
   - Pennsylvania: 7% MAPE (most predictable)
   - Driven by market conditions, not model issues

5. **Brand Reliability in Predictions**
   - Most predictable: BMW, Toyota, Honda
   - Least predictable: Ford, Dodge (higher variance)

### Business Applications

- **Dealers**: Optimize inventory pricing strategies
- **Buyers**: Negotiate with data-driven confidence
- **Auctions**: Set accurate reserve prices
- **Lenders**: Assess collateral value for loans
- **Insurance**: Determine replacement values

---

## 🔧 Production Features

### Drift Detection System

The monitoring system detects three types of drift:

1. **Feature Drift** - Changes in input distributions
   - **Method**: Kolmogorov-Smirnov test (numeric), Chi-square (categorical)
   - **Threshold**: p-value < 0.05

2. **Prediction Drift** - Changes in output distribution
   - **Method**: Population Stability Index (PSI)
   - **Thresholds**:
     - PSI < 0.1: No drift
     - PSI 0.1-0.2: Moderate drift
     - PSI > 0.2: Significant drift

3. **Performance Drift** - Accuracy degradation
   - **Method**: MAE/MAPE tracking over time
   - **Threshold**: >15% increase triggers alert

### Alert System

**3-Tier Severity Levels:**

- **HIGH** (PSI > 0.25)
  - Action: Immediate notification + model retraining
  - Frequency: Real-time

- **MEDIUM** (PSI 0.1-0.25)
  - Action: Daily digest + investigation
  - Frequency: Daily

- **LOW** (PSI < 0.1)
  - Action: Log to dashboard
  - Frequency: Weekly summary

### Retraining Triggers

Automatic retraining initiated when:
- 3 consecutive days of HIGH alerts
- Prediction PSI > 0.3
- Performance degradation > 15%

---

## 📖 Documentation

- **[Methodology Deep Dive](docs/methodology.md)** - Detailed technical approach
- **[Deployment Guide](docs/deployment_guide.md)** - Production deployment steps
- **[API Reference](docs/api_reference.md)** - Code documentation
- **[Model Card](docs/model_card.md)** - ML documentation standard

### Notebooks Overview

| Notebook | Description | Key Outputs |
|----------|-------------|-------------|
| **00** | Data overview, validation, EDA | Data quality report, visualizations |
| **01** | Data cleaning, preprocessing | Cleaned dataset, encoders |
| **02** | Modeling, evaluation, tuning | Trained models, performance metrics |
| **03** | Explainability, fairness | Feature importance, segment analysis |
| **04** | Dashboard preparation | Artifacts, lookup tables, samples |
| **05** | Monitoring, drift detection | Baseline profiles, alert configurations |

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Dataset**: [Syed Anwar Afridi](https://www.kaggle.com/syedanwarafridi) on Kaggle
- **Tools**: XGBoost, Streamlit, Optuna, Scikit-learn communities
- **Inspiration**: Production ML best practices

---

## 📞 Contact

**GitHub**: [@Stevenshanmukh](https://github.com/Stevenshanmukh)

**Project Link**: [https://github.com/Stevenshanmukh/vehicle-sales-prediction](https://github.com/Stevenshanmukh/vehicle-sales-prediction)

---

<div align="center">

**⭐ Star this repo if you find it helpful!**

**Built with ❤️ and Python**

[🔝 Back to Top](#-vehicle-sales-price-prediction---production-ml-system)

</div>
