import pandas as pd
import numpy as np
import joblib

def predict():
    # Load test data
    test_data = pd.read_csv('hidden_test.csv')
    
    # Load model and scaler
    model = joblib.load('model.joblib')
    scaler = joblib.load('scaler.joblib')
    
    # Scale features
    X_test_scaled = scaler.transform(test_data)
    
    # Make predictions
    predictions = model.predict(X_test_scaled)
    
    # Save predictions
    pd.DataFrame({'predictions': predictions}).to_csv('predictions.csv', index=False)

if __name__ == "__main__":
    predict()