import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import joblib
import os

def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def preprocess_data(file_path, seq_length=10, train_split=0.8):
    df = pd.read_csv(file_path)
    # Target features: x, y, z, vx, vy, vz
    features = ['x', 'y', 'z', 'vx', 'vy', 'vz']
    data = df[features].values
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    X, y = create_sequences(scaled_data, seq_length)
    
    split_idx = int(len(X) * train_split)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Save the scaler
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.joblib')
    
    return X_train, y_train, X_test, y_test, scaler

if __name__ == "__main__":
    X_train, y_train, X_test, y_test, scaler = preprocess_data('data/orbit_nominal.csv')
    print(f"Preprocessed data shapes:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
