import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import json
import pickle
from scipy import stats
from src.utils import path_builder
from src.models.pytorch_mlp import build_mlp
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, confusion_matrix, median_absolute_error, explained_variance_score
from src.models.config import FEATURE_DATA_PATH, TARGET_COL

def plot_predictions_vs_true(y_true, y_pred, max_points=100000):
    """Plot predicted values against true values with a diagonal reference line."""
    if len(y_true) != len(y_pred):
        print(f"Warning: Length mismatch - y_true: {len(y_true)}, y_pred: {len(y_pred)}")
        min_len = min(len(y_true), len(y_pred))
        if isinstance(y_true, pd.Series):
            y_true = y_true.iloc[:min_len]
        else:
            y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        print(f"Using first {min_len} elements from both arrays")
    
    if len(y_true) > max_points:
        np.random.seed(42)
        idx = np.random.choice(len(y_true), max_points, replace=False)
        y_true_sample = y_true.iloc[idx] if isinstance(y_true, pd.Series) else y_true[idx]
        y_pred_sample = y_pred[idx]
    else:
        y_true_sample = y_true
        y_pred_sample = y_pred
    
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true_sample, y_pred_sample, alpha=0.5)
    
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

def plot_residual_histogram(y_true, y_pred, model_name="MLP"):
    """Plot histogram of residuals with clipped range for visibility."""
    residuals = y_true - y_pred
    
    # Clip residuals to [-50, 50] range for visualization only
    residuals_clipped = np.clip(residuals, -50, 50)
    
    plt.figure(figsize=(10, 6))
    plt.hist(residuals_clipped, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    plt.xlim(-50, 50)
    plt.xlabel('Residuals (True - Predicted) [Clipped to ±50]')
    plt.ylabel('Frequency')
    plt.title(f'{model_name} Residual Distribution (Clipped to ±50)')
    plt.grid(True, alpha=0.3)
    
    mean_res = np.mean(residuals)
    std_res = np.std(residuals)
    
    plt.axvline(mean_res, color='red', linestyle='--', label=f'Mean: {mean_res:.2f}')
    plt.axvline(mean_res + std_res, color='orange', linestyle='--', alpha=0.7, label=f'±1σ: {std_res:.2f}')
    plt.axvline(mean_res - std_res, color='orange', linestyle='--', alpha=0.7)
    
    outliers = np.sum((residuals < -50) | (residuals > 50))
    plt.text(0.02, 0.98, f'Outliers clipped: {outliers}/{len(residuals)}', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_qq_plot(y_true, y_pred, model_name="MLP"):
    """Plot Q-Q plot to check if residuals follow normal distribution."""
    residuals = y_true - y_pred
    
    plt.figure(figsize=(8, 8))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title(f'{model_name} QQ Plot (Residuals vs Normal Distribution)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def get_test_data():
    """Get test data using the SAME leak-free contract-based split as training."""
    from src.features.build_features import generate_flat_datasets_for_xgboost
    
    print("Loading test data using SAME leak-free method as training...")
    
    X_train, y_train, X_val, y_val, X_test, y_test = generate_flat_datasets_for_xgboost()
    
    print(f"Test data shape: {X_test.shape}")
    print(f"Test target range: {y_test.min():.2f} to {y_test.max():.2f}")
    print(f"Feature columns: {list(X_test.columns)}")
    
    return X_test, y_test

if __name__ == "__main__":
    with open(path_builder("data\mlp", "scaler_X.pkl"), "rb") as f:
        scaler_X = pickle.load(f)
    
    with open(path_builder("data\mlp", "scaler_y.pkl"), "rb") as f:
        scaler_y = pickle.load(f)
    
    X_test, y_test = get_test_data()
    
    print(f"MLP test data shape: {X_test.shape}")
    print(f"MLP test target shape: {y_test.shape}")
    
    print(f"Number of samples before filtering: {len(y_test)}")
    mask = y_test >= 0
    X_test_filtered = X_test[mask]
    y_test_filtered = y_test[mask]
    
    print(f"Number of samples after filtering y_test >= 50: {len(y_test_filtered)}")
    print(f"MLP test target range (after filtering): {y_test_filtered.min():.2f} to {y_test_filtered.max():.2f}")
    
    X_test_scaled = scaler_X.transform(X_test_filtered)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_path = path_builder("src\model_files", "best_pytorch_mlp.pt")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    first_layer_weight = checkpoint['model.0.weight']
    input_dim = first_layer_weight.shape[1]
    first_hidden = first_layer_weight.shape[0]
    
    print(f"Detected input_dim: {input_dim}")
    print(f"Detected first hidden layer size: {first_hidden}")
    
    hidden_units = []
    layer_idx = 0
    
    while f'model.{layer_idx}.weight' in checkpoint:
        weight_shape = checkpoint[f'model.{layer_idx}.weight'].shape
        if layer_idx == 0:
            hidden_units.append(weight_shape[0])
        else:
            if weight_shape[0] == 1:
                break
            else:
                hidden_units.append(weight_shape[0])
        
        layer_idx += 4
    
    print(f"Detected architecture - hidden_units: {hidden_units}")
    
    try:
        with open(path_builder("src\model_files", "pytorch_mlp_config.json"), "r") as f:
            model_config = json.load(f)
        dropout_rate = model_config.get('dropout_rate', 0.2)
    except:
        dropout_rate = 0.2
    
    model = build_mlp(
        input_dim=input_dim,
        hidden_units=hidden_units,
        dropout_rate=dropout_rate
    )
    
    try:
        model.load_state_dict(checkpoint)
        print("Model loaded successfully with detected architecture!")
    except Exception as e:
        print(f"Still having issues loading model: {e}")
        
        print("Trying with inferred architecture from error messages...")
        model = build_mlp(
            input_dim=15,
            hidden_units=[256, 128, 64],
            dropout_rate=dropout_rate
        )
        
        try:
            model.load_state_dict(checkpoint)
            print("Model loaded successfully with inferred architecture!")
        except Exception as e2:
            print(f"Failed to load with inferred architecture: {e2}")
            print("You may need to retrain the model or check the saved checkpoint.")
            exit(1)
    
    model.to(device)
    model.eval()
    
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
    
    with torch.no_grad():
        y_pred_scaled = model(X_test_tensor).cpu().numpy().flatten()
    
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    
    print(f"Predictions shape: {y_pred.shape}")
    print(f"Predictions range: {y_pred.min():.2f} to {y_pred.max():.2f}")
    
    plot_predictions_vs_true(y_test_filtered, y_pred)
    plot_residuals(y_test_filtered, y_pred)
    plot_residual_histogram(y_test_filtered, y_pred, "MLP")
    plot_qq_plot(y_test_filtered, y_pred, "MLP")
    regression_confusion_matrix(y_test_filtered, y_pred, bins=5)
    print_metrics(y_test_filtered, y_pred)
    
    results_df = pd.DataFrame(X_test_filtered, columns=X_test.columns)
    results_df['predicted_price'] = y_pred
    results_df['true_price'] = y_test_filtered.values
    results_df['error'] = results_df['true_price'] - results_df['predicted_price']
    results_df['abs_error'] = results_df['error'].abs()
    
    print("\nSample of test data with predictions:")
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.width', 120)
    print(results_df.head(10).to_string())
    
    results_df.to_csv("mlp_predictions.csv", index=False)
    print(f"Full results saved to mlp_predictions.csv")
    
    print("\n" + "="*50)
    print("SUMMARY:")
    print(f"MLP test samples: {len(y_test_filtered)}")
    print(f"MLP target range: {y_test_filtered.min():.2f} to {y_test_filtered.max():.2f}")
    print("="*50)
