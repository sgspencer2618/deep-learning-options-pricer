import torch
import torch.optim as optim
import torch.nn as nn
import os
import numpy as np
import logging
import optuna
import pickle
from src.models.config import GRU_MODEL_SAVE_PATH
from src.features.build_features import generate_gru_window_datasets
from src.models.config import X_TRAIN_PATH, Y_TRAIN_PATH, X_VAL_PATH, Y_VAL_PATH, X_TEST_PATH, Y_TEST_PATH
from src.models.attentive_gru import AttentiveGRU
from src.datasets.dataset import OptionSequenceDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from src.utils import path_builder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def weighted_loss(y_pred, y_true, weights, loss_fn):
    loss = loss_fn(y_pred, y_true)
    return (loss * weights).mean()

def train_gru(model, train_loader, val_loader, epochs=50, lr=1e-3, patience=5, device="cuda"):
    import pickle
    from sklearn.metrics import mean_squared_error
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.HuberLoss(delta=1.0)
    best_val_loss = float('inf')
    patience_counter = 0
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_rmse': [],
        'val_rmse': []
    }
    
    logger.info(f"Starting training for {epochs} epochs with learning rate {lr}")
    
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch+1}/{epochs}")
        
        # Training phase
        model.train()
        train_losses = []
        train_predictions = []
        train_targets = []
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            
            # Store predictions and targets for RMSE calculation
            train_predictions.extend(y_pred.detach().cpu().numpy().flatten())
            train_targets.extend(y_batch.detach().cpu().numpy().flatten())
            
            epsilon=1e-3
            alpha=2
            weights = (y_batch.abs() + epsilon) / (y_batch.abs().mean() + epsilon) ** alpha
            loss = weighted_loss(y_pred, y_batch, weights, F.mse_loss)
            train_losses.append(loss.item())
            loss.backward()
            optimizer.step()
        
        avg_train_loss = sum(train_losses) / len(train_losses)
        
        # Calculate training RMSE
        train_rmse = np.sqrt(mean_squared_error(train_targets, train_predictions))
        
        # Validation phase
        model.eval()
        val_losses = []
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                val_loss = criterion(y_pred, y_batch)
                val_losses.append(val_loss.item())
                
                # Store predictions and targets for RMSE calculation
                val_predictions.extend(y_pred.cpu().numpy().flatten())
                val_targets.extend(y_batch.cpu().numpy().flatten())
        
        avg_val_loss = sum(val_losses) / len(val_losses)
        
        # Calculate validation RMSE
        val_rmse = np.sqrt(mean_squared_error(val_targets, val_predictions))
        
        # Store metrics in history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_rmse'].append(train_rmse)
        history['val_rmse'].append(val_rmse)
        
        print(f"Epoch {epoch+1}")
        print(f"  Train - Loss: {avg_train_loss:.4f}, RMSE: {train_rmse:.4f}")
        print(f"  Val   - Loss: {avg_val_loss:.4f}, RMSE: {val_rmse:.4f}")
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), GRU_MODEL_SAVE_PATH)
            print(f"  ✓ Val loss improved, saved model")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered!")
                break
    
    # Save training history
    gru_training_history_path = path_builder("src\\model_files", "gru_training_history.pkl")
    os.makedirs("../models", exist_ok=True)
    with open(gru_training_history_path, "wb") as f:
        pickle.dump(history, f)
    logger.info(f"Training history saved to {gru_training_history_path}")
    
    # Load best model
    model.load_state_dict(torch.load(GRU_MODEL_SAVE_PATH))
    return model

def objective(trial):
    """Optuna objective function for hyperparameter optimization."""
    # Load data once for all trials
    if not os.path.exists(X_TRAIN_PATH):
        X_train, y_train, X_val, y_val, X_test, y_test = generate_gru_window_datasets()
    else:
        X_train = np.load(X_TRAIN_PATH)
        y_train = np.load(Y_TRAIN_PATH)
        X_val = np.load(X_VAL_PATH)
        y_val = np.load(Y_VAL_PATH)
    
    # Suggest hyperparameters
    hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256])
    num_layers = trial.suggest_int('num_layers', 2, 4)
    lr = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
    
    # Create data loaders with trial batch size
    train_ds = OptionSequenceDataset(X_train, y_train)
    val_ds = OptionSequenceDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    
    # Create model with trial parameters
    model = AttentiveGRU(input_dim=X_train.shape[-1], hidden_dim=hidden_dim, num_layers=num_layers)
    
    # Train with reduced epochs for optimization speed
    model = train_gru(model, train_loader, val_loader, epochs=15, lr=lr, patience=5, device="cuda")
    
    # Return final validation RMSE
    from sklearn.metrics import mean_squared_error
    model.eval()
    val_predictions = []
    val_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to("cuda"), y_batch.to("cuda")
            y_pred = model(X_batch)
            val_predictions.extend(y_pred.cpu().numpy().flatten())
            val_targets.extend(y_batch.cpu().numpy().flatten())
    
    return np.sqrt(mean_squared_error(val_targets, val_predictions))

def main():
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device name: {torch.cuda.get_device_name(0)}")
    
    # Run Optuna optimization
    logger.info("Starting Optuna hyperparameter optimization...")
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20)
    
    print("Best trial:")
    print(f"  Value (RMSE): {study.best_value:.4f}")
    print(f"  Params: {study.best_params}")
    
    # Train final model with best parameters
    logger.info("Training final model with best parameters...")
    if not os.path.exists(X_TRAIN_PATH):
        X_train, y_train, X_val, y_val, X_test, y_test = generate_gru_window_datasets()
    else:
        X_train = np.load(X_TRAIN_PATH)
        y_train = np.load(Y_TRAIN_PATH)
        X_val = np.load(X_VAL_PATH)
        y_val = np.load(Y_VAL_PATH)
    
    best_params = study.best_params
    train_ds = OptionSequenceDataset(X_train, y_train)
    val_ds = OptionSequenceDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=best_params['batch_size'], shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=best_params['batch_size'])
    
    model = AttentiveGRU(
        input_dim=X_train.shape[-1], 
        hidden_dim=best_params['hidden_dim'], 
        num_layers=best_params['num_layers']
    )
    model = train_gru(model, train_loader, val_loader, epochs=30, lr=best_params['learning_rate'], patience=10, device="cuda")
    
    # Save study results
    study_path = path_builder("src\\model_files", "gru_optuna_study.pkl")
    with open(study_path, "wb") as f:
        pickle.dump({'best_params': study.best_params, 'best_value': study.best_value, 'trials': [{'params': t.params, 'value': t.value} for t in study.trials]}, f)
    logger.info(f"Study results saved to {study_path}")


if __name__ == "__main__":
    logger.info(f"GRU Model save path: {GRU_MODEL_SAVE_PATH}")
    path_exists = os.path.exists(GRU_MODEL_SAVE_PATH)
    if path_exists:
        logger.info(f" GRU Model path exists: {path_exists}")
        main()
    else:
        logger.info(f"GRU Model path does not exist")
