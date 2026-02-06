import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import joblib
from preprocess import preprocess_data
from model import OrbitLSTM
from physics import get_orbital_energy

def train_model():
    # Setup
    seq_length = 20
    batch_size = 32
    epochs = 50
    learning_rate = 0.001
    
    # Load and Preprocess
    print("Loading data...")
    X_train, y_train, X_test, y_test, scaler = preprocess_data('data/orbit_nominal.csv', seq_length=seq_length)
    
    # Convert to Tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test)
    
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
    
    # Model
    model = OrbitLSTM(input_size=6, hidden_size=64, num_layers=2, output_size=6)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Scaler parameters for un-normalization (to use in Physics Loss if needed)
    # Scaler: (x - mean) / scale -> x = (scaled * scale) + mean
    means = torch.FloatTensor(scaler.mean_)
    scales = torch.FloatTensor(scaler.scale_)
    
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            
            # Standard MSE Loss
            loss_mse = criterion(outputs, batch_y)
            
            # Physics-Informed Loss (Conceptual implementation)
            # Un-normalize outputs and batch_y to real units
            # pred_real = outputs * scales + means
            # target_real = batch_y * scales + means
            # loss_physics = physics_loss(pred_real, target_real)
            
            loss = loss_mse #+ 0.01 * loss_physics
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {train_loss/len(train_loader):.6f}")

    # Save model
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/orbit_lstm.pth')
    print("Model saved to models/orbit_lstm.pth")

if __name__ == "__main__":
    train_model()
