# Vehicle Price Prediction Dashboard

## Overview
Production-ready Streamlit dashboard for vehicle price prediction using XGBoost model.

**Model Performance:**
- MAE: $887.48
- R-squared: 0.9682
- MAPE: 12.30%

## Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Run dashboard
streamlit run app.py
```

## Dashboard Features

### 1. Price Prediction Tool
- Interactive form with dropdowns for vehicle attributes
- Real-time price predictions
- Confidence scoring
- MMR comparison

### 2. Model Insights
- Feature importance visualization
- Performance metrics dashboard
- Prediction confidence analysis

### 3. Market Analysis
- Price trends by segment (state, make, body type)
- Popular vehicle combinations
- Market statistics

### 4. Sample Predictions
- Pre-computed examples across price ranges
- Prediction explanations
- Accuracy demonstrations

## File Structure
```
app/
├── app.py                      # Main Streamlit dashboard
├── predictor.py                # Prediction interface class
├── artifacts/
│   ├── xgboost_optimized.pkl   # Trained model
│   ├── label_encoders.pkl      # Categorical encoders
│   ├── model_metadata.pkl      # Model info
│   ├── lookup_tables.pkl       # Dropdown options
│   ├── lookup_tables.json      # Simplified lookups
│   ├── sample_predictions.pkl  # Example predictions
│   ├── sample_predictions.json
│   ├── dashboard_data.pkl      # Performance metrics
│   └── dashboard_data.json
├── assets/                     # Images, logos
└── requirements.txt            # Python dependencies
```

## Usage Example
```python
from predictor import VehiclePricePredictor

# Initialize predictor
predictor = VehiclePricePredictor()

# Make prediction
vehicle = {
    'year': 2012,
    'make': 'Toyota',
    'model_grouped': 'Camry',
    'body': 'Sedan',
    'transmission': 'Automatic',
    'odometer': 50000,
    'condition': 35,
    'state': 'Ca',
    'color': 'Black',
    'interior': 'Black',
    'seller_grouped': 'Other_Seller',
    'trim_grouped': 'Se',
    'mmr': 12000
}

result = predictor.predict(vehicle)
print(f"Predicted Price: ${result['predicted_price']:,.2f}")
```

## Model Details

**Algorithm:** XGBoost Regressor (Optimized with Optuna)

**Features (17 total):**
- Numeric: year, condition, odometer, mmr, vehicle_age, log_odometer, age_odo_interaction, has_date
- Categorical: make, body, transmission, state, color, interior, seller_grouped, model_grouped, trim_grouped

**Training Data:** 558,825 vehicle sales records (2014-2015)

**Key Feature Importance:**
1. MMR (61.7%)
2. Body Type (9.0%)
3. Age-Odometer Interaction (8.5%)
4. Make (5.9%)
5. Model (3.0%)

## Performance by Segment

- **Best Accuracy:** $30k-$50k range (3.19% MAPE)
- **Best State:** PA (7.05% MAPE)
- **Best Make:** BMW (7.61% MAPE)

## Deployment Checklist

- [x] Model trained and optimized
- [x] Artifacts packaged
- [x] Prediction interface created
- [x] Sample data prepared
- [x] Documentation complete
- [ ] Streamlit app.py created
- [ ] Testing completed
- [ ] Production deployment

## Support

For issues or questions, refer to the project notebooks:
- Notebook 00: Data Overview
- Notebook 01: Data Cleaning
- Notebook 02: Modeling
- Notebook 03: Explainability
- Notebook 04: Dashboard Prep (this notebook)

## License

Internal use only - Vehicle Sales & Market Insights Project
