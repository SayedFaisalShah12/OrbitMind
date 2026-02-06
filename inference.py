import torch
import numpy as np
import pandas as pd
import joblib
import os
from model import OrbitLSTM
from preprocess import create_sequences
import visualize as vis

def run_inference():
    print("\n--- Running Inference & Anomaly Detection ---")
    
    # Load model and scaler
    model = OrbitLSTM(input_size=6, hidden_size=64, num_layers=2, output_size=6)
    model.load_state_dict(torch.load('models/orbit_lstm.pth'))
    model.eval()
    
    scaler = joblib.load('models/scaler.joblib')
    
    # Load anomaly data
    df_anomaly = pd.read_csv('data/orbit_anomaly.csv')
    features = ['x', 'y', 'z', 'vx', 'vy', 'vz']
    data = df_anomaly[features].values
    
    # Normalize
    scaled_data = scaler.transform(data)
    
    # Create sequences
    seq_length = 20
    X, y_true = create_sequences(scaled_data, seq_length)
    
    X_t = torch.FloatTensor(X)
    
    with torch.no_grad():
        y_pred = model(X_t).numpy()
    
    # Calculate Reconstruction Error (MSE)
    errors = np.mean((y_true - y_pred)**2, axis=1)
    
    # Threshold for anomaly (e.g., 5x the mean error)
    threshold = np.mean(errors) + 3 * np.std(errors)
    anomalies = errors > threshold
    
    print(f"Detected {np.sum(anomalies)} anomalous points out of {len(errors)} segments.")
    
    # Visualizing prediction for 'x' coordinate
    # Re-scale back for plotting
    y_true_rescaled = scaler.inverse_transform(y_true)[:, 0]
    y_pred_rescaled = scaler.inverse_transform(y_pred)[:, 0]
    
    vis.plot_prediction_error(y_true_rescaled, y_pred_rescaled, 'X-Coordinate')
    
    # Create a dataframe for the error analysis
    df_error = pd.DataFrame({
        't': df_anomaly['t'].values[seq_length:],
        'error': errors,
        'is_anomaly': anomalies
    })
    df_error.to_csv('data/anomaly_results.csv', index=False)
    print("Anomaly results saved to data/anomaly_results.csv")

if __name__ == "__main__":
    # Check if data and model exist
    if not os.path.exists('data/orbit_nominal.csv'):
        print("Data not found. Run python scripts/data_gen.py first.")
    elif not os.path.exists('models/orbit_lstm.pth'):
        print("Model not found. Run python scripts/train.py first.")
    else:
        run_inference()
