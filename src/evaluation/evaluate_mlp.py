import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import json
import pickle
from src.utils import path_builder
from src.models.pytorch_mlp import build_mlp
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, confusion_matrix, median_absolute_error, explained_variance_score
from src.models.config import FEATURE_DATA_PATH, FEATURE_COLS, TARGET_COL

def plot_predictions_vs_true(y_true, y_pred, max_points=100000):
    """
    Plot predicted values against true values with a diagonal reference line.
    
    Args:
        y_true: Array of true values
        y_pred: Array of predicted values
        max_points: Maximum number of points to plot to avoid overcrowding
    """
    # Ensure y_true and y_pred have the same length
    if len(y_true) != len(y_pred):
        print(f"Warning: Length mismatch - y_true: {len(y_true)}, y_pred: {len(y_pred)}")
        # Use the smaller size
        min_len = min(len(y_true), len(y_pred))
        if isinstance(y_true, pd.Series):
            y_true = y_true.iloc[:min_len]
        else:
            y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        print(f"Using first {min_len} elements from both arrays")
    
    # Sample points if there are too many
    if len(y_true) > max_points:
        # Use a fixed random seed for reproducibility
        np.random.seed(42)
        idx = np.random.choice(len(y_true), max_points, replace=False)
        y_true_sample = y_true.iloc[idx] if isinstance(y_true, pd.Series) else y_true[idx]
        y_pred_sample = y_pred[idx]
    else:
        y_true_sample = y_true
        y_pred_sample = y_pred
    
    # Create the scatter plot
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true_sample, y_pred_sample, alpha=0.5)
    
    # Add diagonal line (perfect predictions)
    min_val = min(np.min(y_true_sample), np.min(y_pred_sample))
    max_val = max(np.max(y_true_sample), np.max(y_pred_sample))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('MLP Predicted vs True Values')
    plt.axis('equal')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_residuals(y_true, y_pred):
    """Plot histogram and scatter plot of regression residuals."""
    residuals = y_true - y_pred
    plt.figure(figsize=(8,5))
    sns.histplot(residuals, bins=50, kde=True)
    plt.title("Residuals Distribution (Actual - Predicted)")
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.show()

    plt.figure(figsize=(8,5))
    plt.scatter(y_true, residuals, alpha=0.3)
    plt.axhline(0, color='r', linestyle='--')
    plt.title("Residuals vs Actual")
    plt.xlabel("Actual Value")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.tight_layout()
    plt.show()

def regression_confusion_matrix(y_true, y_pred, bins=5):
    """Show a binned confusion matrix for regression."""
    y_true_binned = pd.cut(y_true, bins, labels=False)
    y_pred_binned = pd.cut(y_pred, bins, labels=False)
    cm = confusion_matrix(y_true_binned, y_pred_binned)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted Bin")
    plt.ylabel("Actual Bin")
    plt.title("Binned Regression Confusion Matrix")
    plt.show()

def print_metrics(y_true, y_pred):
    """Print RMSE and MAE."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    ev = explained_variance_score(y_true, y_pred)
    print(f"RMSE: {rmse:.5f}")
    print(f"MAE: {mae:.5f}")
    print(f"MedAE: {medae:.5f}")
    print(f"R2: {r2:.5f}")
    print(f"Explained Variance: {ev:.5f}")

def get_test_data():
    """
    Get test data using the same time-based split as other models.
    
    Returns:
        tuple: (X_test, y_test) DataFrames
    """
    print(f"Loading test data from {FEATURE_DATA_PATH}")
    df = pd.read_parquet(FEATURE_DATA_PATH)
    
    # Use same time-based split as other models (70/15/15)
    unique_dates = df['date'].sort_values().unique()
    n_dates = len(unique_dates)
    
    # Calculate split indices
    train_idx = int(n_dates * 0.7)   # 70% for training
    val_idx = int(n_dates * 0.85)    # 85% cumulative (next 15% for validation)
    
    # Test dates are the last 15%
    test_dates = unique_dates[val_idx:]
    
    print(f"Test dates: {test_dates[0]} to {test_dates[-1]} ({len(test_dates)} dates)")
    
    # Filter data by test dates
    test_data = df[df['date'].isin(test_dates)]
    
    X_test = test_data[FEATURE_COLS]
    y_test = test_data[TARGET_COL]
    
    print(f"Test data shape: {X_test.shape}")
    print(f"Test target range (before filtering): {y_test.min():.2f} to {y_test.max():.2f}")
    
    return X_test, y_test

if __name__ == "__main__":
    # Load scalers and test data
    with open(path_builder("data\mlp", "scaler_X.pkl"), "rb") as f:
        scaler_X = pickle.load(f)
    
    with open(path_builder("data\mlp", "scaler_y.pkl"), "rb") as f:
        scaler_y = pickle.load(f)
    
    # Get raw test data (same as other models)
    X_test, y_test = get_test_data()
    
    print(f"MLP test data shape: {X_test.shape}")
    print(f"MLP test target shape: {y_test.shape}")
    
    # Apply same filtering as other models (remove options with price < 50)
    print(f"Number of samples before filtering: {len(y_test)}")
    mask = y_test >= 50
    X_test_filtered = X_test[mask]
    y_test_filtered = y_test[mask]
    
    print(f"Number of samples after filtering y_test >= 50: {len(y_test_filtered)}")
    print(f"MLP test target range (after filtering): {y_test_filtered.min():.2f} to {y_test_filtered.max():.2f}")
    
    # Scale the filtered data
    X_test_scaled = scaler_X.transform(X_test_filtered)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model config
    with open(path_builder("src\model_files", "pytorch_mlp_config.json"), "r") as f:
        model_config = json.load(f)
    
    # Create model with same architecture
    model = build_mlp(
        input_dim=model_config['input_dim'],
        hidden_units=model_config['hidden_units'],
        dropout_rate=model_config['dropout_rate']
    )
    
    # Load model weights
    model.load_state_dict(torch.load(path_builder("src\model_files", "best_pytorch_mlp.pt"), 
                                     map_location=device))
    model.to(device)
    model.eval()  # Set to evaluation mode
    
    # Convert data to tensor
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
    
    # Make predictions
    with torch.no_grad():
        y_pred_scaled = model(X_test_tensor).cpu().numpy().flatten()
    
    # Convert predictions back to original scale
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    
    print(f"Predictions shape: {y_pred.shape}")
    print(f"Predictions range: {y_pred.min():.2f} to {y_pred.max():.2f}")
    
    # Generate evaluation plots and metrics
    plot_predictions_vs_true(y_test_filtered, y_pred)
    plot_residuals(y_test_filtered, y_pred)
    regression_confusion_matrix(y_test_filtered, y_pred, bins=5)
    print_metrics(y_test_filtered, y_pred)
    
    # Create and print a table with input features, predictions, and true values
    results_df = pd.DataFrame(X_test_filtered, columns=FEATURE_COLS)
    results_df['predicted_price'] = y_pred
    results_df['true_price'] = y_test_filtered.values
    results_df['error'] = results_df['true_price'] - results_df['predicted_price']
    results_df['abs_error'] = results_df['error'].abs()
    
    # Print first 10 rows of the table
    print("\nSample of test data with predictions:")
    pd.set_option('display.max_columns', 10)  # Limit columns shown
    pd.set_option('display.width', 120)       # Set width to fit console
    print(results_df.head(10).to_string())
    
    # Save the complete results table
    results_df.to_csv("mlp_predictions.csv", index=False)
    print(f"Full results saved to mlp_predictions.csv")
    
    # Summary comparison output
    print("\n" + "="*50)
    print("SUMMARY:")
    print(f"MLP test samples: {len(y_test_filtered)}")
    print(f"MLP target range: {y_test_filtered.min():.2f} to {y_test_filtered.max():.2f}")
    print("="*50)
