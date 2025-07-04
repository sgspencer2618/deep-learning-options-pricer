import torch
import torch.optim as optim
import torch.nn as nn
import os
import numpy as np
import logging
from src.model.config import GRU_MODEL_SAVE_PATH
from src.features.build_features import generate_gru_window_datasets
from src.features.build_features import X_train_path, X_val_path, X_test_path, y_train_path, y_val_path, y_test_path
from src.neural.attentive_gru import AttentiveGRU
from src.neural.dataset import OptionSequenceDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def weighted_loss(y_pred, y_true, weights, loss_fn):
    loss = loss_fn(y_pred, y_true)
    return (loss * weights).mean()

def train_gru(
    model, train_loader, val_loader, epochs=50, lr=1e-3, patience=5, device="cuda"
):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.HuberLoss(delta=1.0)
    best_val_loss = float('inf')
    patience_counter = 0
    logger.info(f"Starting training for {epochs} epochs with learning rate {lr}")
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch+1}/{epochs}")
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            epsilon=1e-3
            alpha=2
            weights = (y_batch.abs() + epsilon) / (y_batch.abs().mean() + epsilon) ** alpha
            loss = weighted_loss(y_pred, y_batch, weights, F.mse_loss)
            loss.backward()
            optimizer.step()
        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                val_loss = criterion(y_pred, y_batch)
                val_losses.append(val_loss.item())
        avg_val_loss = sum(val_losses) / len(val_losses)
        print(f"Epoch {epoch+1}, Val Loss: {avg_val_loss:.4f}")
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), GRU_MODEL_SAVE_PATH)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered!")
                break
    # Load best model
    model.load_state_dict(torch.load(GRU_MODEL_SAVE_PATH))
    return model

def main():
    if not os.path.exists(X_train_path):
        X_train, y_train, X_val, y_val, X_test, y_test = generate_gru_window_datasets()
    else:
        logger.info("Loading pre-generated GRU datasets from disk.")
        X_train = np.load(X_train_path)
        y_train = np.load(y_train_path)
        X_val = np.load(X_val_path)
        y_val = np.load(y_val_path)
        X_test = np.load(X_test_path)
        y_test = np.load(y_test_path)

    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of CUDA devices: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
    
        train_ds = OptionSequenceDataset(X_train, y_train)
        val_ds = OptionSequenceDataset(X_val, y_val)

        train_loader = DataLoader(train_ds, batch_size=128, shuffle=False)
        val_loader = DataLoader(val_ds, batch_size=256)

        model = AttentiveGRU(input_dim=X_train.shape[-1], hidden_dim=128, num_layers=2)
        model = train_gru(model, train_loader, val_loader, epochs=30, lr=1e-3, patience=10, device="cuda")


if __name__ == "__main__":
    logger.info(f"GRU Model save path: {GRU_MODEL_SAVE_PATH}")
    path_exists = os.path.exists(GRU_MODEL_SAVE_PATH)
    if path_exists:
        logger.info(f" GRU Model path exists: {path_exists}")
        main()
    else:
        logger.info(f"GRU Model path does not exist")
