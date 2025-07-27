from sklearn.preprocessing import StandardScaler
from src.models.config import FEATURE_DATA_PATH, FEATURE_COLS, TARGET_COL, HYPERPARAM_SPACE, MODEL_SAVE_PATH
from src.training.train_xgb import load_data_time_split
from src.models.pytorch_mlp import build_mlp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from src.utils import path_builder
import numpy as np
import pandas as pd
import pickle
import os
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load and split data
X_train, y_train, X_val, y_val, X_test, y_test = load_data_time_split()

# Scale feature data
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled = scaler_X.transform(X_val)
X_test_scaled = scaler_X.transform(X_test)

# Scale target data - reshape to 2D array for StandardScaler
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_val_scaled = scaler_y.transform(y_val.values.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

# Create directory if it doesn't exist
os.makedirs("..\data\mlp", exist_ok=True)
os.makedirs("..\src\model_files", exist_ok=True)

# Save scalers as pickle files (preserves all attributes)
with open(path_builder("data\mlp", "scaler_X.pkl"), "wb") as f:
    pickle.dump(scaler_X, f)

with open(path_builder("data\mlp", "scaler_y.pkl"), "wb") as f:
    pickle.dump(scaler_y, f)

np.save(path_builder("data\mlp", "X_train_scaled.npy"), X_train_scaled)
np.save(path_builder("data\mlp", "X_val_scaled.npy"), X_val_scaled)
np.save(path_builder("data\mlp", "X_test_scaled.npy"), X_test_scaled)
np.save(path_builder("data\mlp", "y_train_scaled.npy"), y_train_scaled)
np.save(path_builder("data\mlp", "y_val_scaled.npy"), y_val_scaled)
np.save(path_builder("data\mlp", "y_test_scaled.npy"), y_test_scaled)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train_scaled)
X_val_tensor = torch.FloatTensor(X_val_scaled)
y_val_tensor = torch.FloatTensor(y_val_scaled)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.FloatTensor(y_test_scaled)

# Create data loaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor.unsqueeze(1))
val_dataset = TensorDataset(X_val_tensor, y_val_tensor.unsqueeze(1))
test_dataset = TensorDataset(X_test_tensor, y_test_tensor.unsqueeze(1))

batch_size = 512
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Initialize model
input_dim = X_train_scaled.shape[1]
model = build_mlp(input_dim, hidden_units=[128, 64], dropout_rate=0.2)
model = model.to(device)

# Define model file paths
model_checkpoint_path = path_builder("src\model_files", "best_pytorch_mlp.pt")
model_final_path = path_builder("src\model_files", "final_pytorch_mlp.pt")
model_config_path = path_builder("src\model_files", "pytorch_mlp_config.json")
os.makedirs(os.path.dirname(model_checkpoint_path), exist_ok=True)
os.makedirs(os.path.dirname(model_final_path), exist_ok=True)
os.makedirs(os.path.dirname(model_config_path), exist_ok=True)

if os.path.exists(model_final_path):
    print(f"Verified: Model file exists at {model_final_path}")
else:
    print(f"Warning: Model file not found at {model_final_path}")

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

# Training loop with early stopping
n_epochs = 100
patience = 10
best_val_loss = float('inf')
patience_counter = 0
history = {'train_loss': [], 'val_loss': []}

# Training loop
for epoch in range(n_epochs):
    # Training phase
    model.train()
    train_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        # Forward pass
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * X_batch.size(0)
    
    train_loss /= len(train_loader.dataset)
    history['train_loss'].append(train_loss)
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            val_loss += criterion(y_pred, y_batch).item() * X_batch.size(0)
    
    val_loss /= len(val_loader.dataset)
    history['val_loss'].append(val_loss)
    
    # Print progress
    if epoch % 5 == 0 or epoch == n_epochs - 1:
        print(f"Epoch {epoch+1}/{n_epochs} - Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")
    
    # Check for early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Save the best model
        torch.save(model.state_dict(), model_checkpoint_path)
        print(f"Epoch {epoch+1}: Val loss improved to {val_loss:.4f}, saved model to {model_checkpoint_path}")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

# Load the best model for final evaluation
model.load_state_dict(torch.load(model_checkpoint_path))

# Save the final model
torch.save(model.state_dict(), model_final_path)
print(f"Model saved to {model_final_path}")

# Save model architecture config
model_config = {
    'input_dim': input_dim,
    'hidden_units': [128, 64],
    'dropout_rate': 0.1
}
with open(model_config_path, "w") as json_file:
    json.dump(model_config, json_file)
print(f"Model config saved to {model_config_path}")

# Save training history
with open("..\models\mlp_training_history.pkl", "wb") as f:
    pickle.dump(history, f)
print("Training history saved")

# Evaluate the model on test data
model.eval()
test_loss = 0.0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        y_pred = model(X_batch)
        test_loss += criterion(y_pred, y_batch).item() * X_batch.size(0)

test_loss /= len(test_loader.dataset)
print(f"Test loss (MSE): {test_loss}")

# Make predictions on test data
model.eval()
y_pred_scaled = []
with torch.no_grad():
    for X_batch, _ in test_loader:
        X_batch = X_batch.to(device)
        batch_preds = model(X_batch).cpu().numpy()
        y_pred_scaled.extend(batch_preds.flatten())

y_pred_scaled = np.array(y_pred_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

# Calculate and print some metrics on the test set
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Test RMSE: {rmse:.4f}")
print(f"Test MAE: {mae:.4f}")
print(f"Test R²: {r2:.4f}")


