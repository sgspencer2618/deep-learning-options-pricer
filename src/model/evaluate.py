import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error, confusion_matrix
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
    print(f"RMSE: {rmse:.5f}")
    print(f"MAE: {mae:.5f}")

# Example usage after training/evaluation in model_train.py:
if __name__ == "__main__":
    # Load your trained model, data, and feature list here:
    model = xgb.XGBRegressor()
    model.load_model(MODEL_SAVE_PATH)
    
    # Load your validation data:
    df = pd.read_parquet(FEATURE_DATA_PATH)
    split_idx = 1600856
    X_val = df[FEATURE_COLS].iloc[split_idx:]
    y_val = df[TARGET_COL].iloc[split_idx:]
    preds = model.predict(X_val)
    
    # For demo purposes, replace these with your actual variables:
    model, X_val, y_val, preds, FEATURE_COLS

    plot_feature_importance(model, FEATURE_COLS)
    plot_residuals(y_val, preds)
    regression_confusion_matrix(y_val, preds, bins=5)
    print_metrics(y_val, preds)
