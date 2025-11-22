
"""
Vehicle Price Prediction Interface
Simplified interface for Streamlit dashboard
"""

import pandas as pd
import numpy as np
import pickle

class VehiclePricePredictor:
    """
    Simplified interface for vehicle price prediction in dashboard.
    """
    
    def __init__(self, model_path='app/artifacts/xgboost_optimized.pkl', 
                 encoders_path='app/artifacts/label_encoders.pkl',
                 metadata_path='app/artifacts/model_metadata.pkl'):
        """Load model and preprocessing artifacts."""
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        with open(encoders_path, 'rb') as f:
            self.label_encoders = pickle.load(f)
        
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        
        self.features = self.metadata['features']
        self.numeric_features = self.metadata['numeric_features']
        self.categorical_features = self.metadata['categorical_features']
    
    def validate_input(self, input_data):
        """Validate input data."""
        errors = []
        
        # Check required fields
        required_fields = ['year', 'make', 'body', 'transmission', 'state', 
                          'condition', 'odometer', 'color', 'interior', 
                          'seller_grouped', 'model_grouped', 'trim_grouped', 'mmr']
        
        for field in required_fields:
            if field not in input_data:
                errors.append(f"Missing required field: {field}")
        
        if errors:
            return False, errors
        
        # Validate ranges
        if input_data.get('year', 0) < 1980 or input_data.get('year', 0) > 2025:
            errors.append("Year must be between 1980 and 2025")
        
        if input_data.get('odometer', 0) < 0 or input_data.get('odometer', 0) > 500000:
            errors.append("Odometer must be between 0 and 500,000")
        
        if input_data.get('condition', 0) < 1 or input_data.get('condition', 0) > 49:
            errors.append("Condition must be between 1 and 49")
        
        return len(errors) == 0, errors
    
    def engineer_features(self, input_data):
        """Create engineered features."""
        
        # Vehicle age (reference year 2015)
        input_data['vehicle_age'] = 2015 - input_data['year']
        
        # Log odometer
        input_data['log_odometer'] = np.log1p(input_data['odometer'])
        
        # Age-odometer interaction
        input_data['age_odo_interaction'] = input_data['vehicle_age'] * input_data['odometer'] / 10000
        
        # Has date flag (always 1 for dashboard predictions)
        input_data['has_date'] = 1
        
        return input_data
    
    def encode_features(self, input_data):
        """Encode categorical features."""
        
        encoded_data = input_data.copy()
        
        for feature in self.categorical_features:
            if feature in encoded_data:
                try:
                    value = str(encoded_data[feature])
                    encoded_data[feature] = self.label_encoders[feature].transform([value])[0]
                except:
                    # Use most common value if encoding fails
                    encoded_data[feature] = 0
        
        return encoded_data
    
    def predict(self, input_data):
        """
        Make price prediction.
        
        Args:
            input_data (dict): Vehicle attributes
        
        Returns:
            dict: Prediction result with confidence info
        """
        
        # Validate input
        valid, errors = self.validate_input(input_data)
        if not valid:
            return {'success': False, 'errors': errors}
        
        # Engineer features
        input_data = self.engineer_features(input_data)
        
        # Encode categorical features
        encoded_data = self.encode_features(input_data)
        
        # Create feature vector in correct order
        feature_vector = pd.DataFrame([encoded_data])[self.features]
        
        # Predict
        prediction = self.model.predict(feature_vector)[0]
        
        # Calculate confidence based on similar vehicles
        # (simplified - in production would use more sophisticated method)
        confidence = 'High' if 5000 <= prediction <= 50000 else 'Medium'
        
        return {
            'success': True,
            'predicted_price': float(prediction),
            'confidence': confidence,
            'input_summary': {
                'vehicle': f"{input_data['year']} {input_data['make']} {input_data['model_grouped']}",
                'odometer': f"{input_data['odometer']:,} miles",
                'condition': input_data['condition']
            }
        }
    
    def predict_batch(self, input_list):
        """Predict for multiple vehicles."""
        results = []
        for input_data in input_list:
            results.append(self.predict(input_data))
        return results

# Example usage
if __name__ == "__main__":
    predictor = VehiclePricePredictor()
    
    example_vehicle = {
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
    
    result = predictor.predict(example_vehicle)
    print(f"Prediction: ${result['predicted_price']:,.2f}")
