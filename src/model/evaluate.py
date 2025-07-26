import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error, confusion_matrix, median_absolute_error, explained_variance_score
from config import FEATURE_DATA_PATH, FEATURE_COLS, TARGET_COL, HYPERPARAM_SPACE, MODEL_SAVE_PATH

def plot_feature_importance(model, feature_names, max_features=20):
    """Plot XGBoost's built-in and custom bar feature importance."""
    # XGBoost's built-in plot
    xgb.plot_importance(model, importance_type='gain', show_values=False, max_num_features=max_features)
    plt.title("XGBoost Feature Importance (Gain)")
    plt.tight_layout()
    plt.show()
    
    # Custom bar plot
    importances = model.feature_importances_
    indices = importances.argsort()[::-1]
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(feature_names)), importances[indices], align='center')
    plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices], rotation=45)
    plt.title("Feature Importances (XGBoost)")
    plt.tight_layout()
    plt.show()

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
    min_val = min(y_true_sample.min(), y_pred_sample.min())
    max_val = max(y_true_sample.max(), y_pred_sample.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('XGBoost Predicted vs True Values')
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
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
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
    Get test data using the same time-based split as GRU.
    
    Returns:
        tuple: (X_test, y_test) DataFrames
    """
    print(f"Loading test data from {FEATURE_DATA_PATH}")
    df = pd.read_parquet(FEATURE_DATA_PATH)
    
    # Use same time-based split as GRU (70/15/15)
    unique_dates = df['date'].sort_values().unique()
    n_dates = len(unique_dates)
    
    # Calculate split indices (same as GRU)
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

# Example usage after training/evaluation in model_train.py:
if __name__ == "__main__":
    # Load your trained model
    model = xgb.XGBRegressor()
    model.load_model(MODEL_SAVE_PATH)
    
    # Get test data using time-based split (same as GRU)
    X_test, y_test = get_test_data()
    
    print(f"XGBoost test data shape: {X_test.shape}")
    print(f"XGBoost test target shape: {y_test.shape}")
    
    # Apply same filtering as GRU (remove options with price < 50)
    print(f"Number of samples before filtering: {len(y_test)}")
    mask = y_test >= 50
    X_test_filtered = X_test[mask]
    y_test_filtered = y_test[mask]
    
    print(f"Number of samples after filtering y_test >= 50: {len(y_test_filtered)}")
    print(f"XGBoost test target range (after filtering): {y_test_filtered.min():.2f} to {y_test_filtered.max():.2f}")
    
    # Make predictions on filtered data
    preds = model.predict(X_test_filtered)
    print(f"Predictions shape: {preds.shape}")
    print(f"Predictions range: {preds.min():.2f} to {preds.max():.2f}")
    
    # Generate evaluation plots and metrics
    plot_feature_importance(model, FEATURE_COLS)
    plot_predictions_vs_true(y_test_filtered, preds)
    plot_residuals(y_test_filtered, preds)
    regression_confusion_matrix(y_test_filtered, preds, bins=5)
    print_metrics(y_test_filtered, preds)
    
    # Summary comparison output
    print("\n" + "="*50)
    print("SUMMARY:")
    print(f"XGBoost test samples: {len(y_test_filtered)}")
    print(f"XGBoost target range: {y_test_filtered.min():.2f} to {y_test_filtered.max():.2f}")
    print("="*50)

    df = pd.DataFrame(X_test_filtered, columns=FEATURE_COLS)
    df.to_csv("xgboost_test_data.csv", index=False)
