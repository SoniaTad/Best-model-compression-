import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the model from the pickle file
model2 = joblib.load('LR_model.pkl')

# Prepare your input data as a NumPy array
input_data  =( np.array([[0, 80.0, 13.0], [0, 79.9, 13.0], [0, 79.9, 12.9],[0, 82.9, 12.9],[0, 90.0, 12.0],[0, 83.0, 11.5],[0, 120.0, 12.8],[0, 140.9, 12.0],[0, 90.9, 9.0],[0, 90.3, 5.4],[0, 91.9, 5.6],[0, 70.8, 5.1],[0, 130.4, 5.0],[0, 94.9, 19.9],
[1, 80.0, 12.0], [1, 79.9, 12.0], [1, 79.9, 11.9],[1, 82.9, 11.9],[1, 90.0, 11.0],[1, 83.0, 9.5],[1, 120.0, 11.8],[1, 140.9, 10.0],[1, 90.9, 9.0],[1, 90.3, 5.4],[1, 91.9, 5.6],[1, 70.8, 5.1],[1, 130.4, 5.0],[1, 94.9, 19.9]]) ) # Replace the values with your actual input data
  # Replace the values with your actual input data

# Load the scaler from the pickle file
scaler = joblib.load('LR_scaler.pkl')

# Scale the input data using the fitted scaler
input_data_scaled = scaler.transform(input_data)

# Make predictions
predictions = model2.predict(input_data_scaled)
print(predictions)
