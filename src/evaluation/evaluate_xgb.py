import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error, confusion_matrix, median_absolute_error, explained_variance_score
from scipy import stats
from src.models.config import FEATURE_DATA_PATH, FEATURE_COLS, TARGET_COL, HYPERPARAM_SPACE, MODEL_SAVE_PATH

def plot_feature_importance(model, feature_names, max_features=20):
    """Plot XGBoost's built-in and custom bar feature importance."""
    xgb.plot_importance(model, importance_type='gain', show_values=False, max_num_features=max_features)
    plt.title("XGBoost Feature Importance (Gain)")
    plt.tight_layout()
    plt.show()
    
    importances = model.feature_importances_
    indices = importances.argsort()[::-1]
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(feature_names)), importances[indices], align='center')
    plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices], rotation=45)
    plt.title("Feature Importances (XGBoost)")
    plt.tight_layout()
    plt.show()

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

def plot_residual_histogram(y_true, y_pred, model_name="XGBoost"):
    """Plot histogram of residuals with clipped range for visibility."""
    residuals = y_true - y_pred
    
    # Clip residuals to [-50, 50] range for visualization only
    residuals_clipped = np.clip(residuals, -50, 50)
    
    plt.figure(figsize=(10, 6))
    plt.hist(residuals_clipped, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlim(-50, 50)
    plt.xlabel('Residuals (True - Predicted) [Clipped to ±50]')
    plt.ylabel('Frequency')
    plt.title(f'{model_name} Residual Distribution (Clipped to ±50)')
    plt.grid(True, alpha=0.3)
    
    mean_res = np.mean(residuals)
    std_res = np.std(residuals)
    clipped_count = len(residuals) - len(residuals_clipped[residuals_clipped == residuals])
    
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

def plot_qq_plot(y_true, y_pred, model_name="XGBoost"):
    """Plot Q-Q plot to check if residuals follow normal distribution."""
    residuals = y_true - y_pred
    
    plt.figure(figsize=(8, 8))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title(f'{model_name} QQ Plot (Residuals vs Normal Distribution)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
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
    """Get test data using the SAME leak-free contract-based split as training."""
    from src.features.build_features import generate_flat_datasets_for_xgboost
    
    print("Loading test data using SAME leak-free method as training...")
    
    X_train, y_train, X_val, y_val, X_test, y_test = generate_flat_datasets_for_xgboost()
    
    print(f"Test data shape: {X_test.shape}")
    print(f"Test target range: {y_test.min():.2f} to {y_test.max():.2f}")
    print(f"Feature columns: {list(X_test.columns)}")
    
    return X_test, y_test

if __name__ == "__main__":
    model = xgb.XGBRegressor()
    model.load_model(MODEL_SAVE_PATH)
    
    X_test, y_test = get_test_data()
    
    print(f"XGBoost test data shape: {X_test.shape}")
    print(f"XGBoost test target shape: {y_test.shape}")
    
    print(f"Number of samples before filtering: {len(y_test)}")
    mask = y_test >= 0
    X_test_filtered = X_test[mask]
    y_test_filtered = y_test[mask]
    
    print(f"Number of samples after filtering y_test >= 50: {len(y_test_filtered)}")
    print(f"XGBoost test target range (after filtering): {y_test_filtered.min():.2f} to {y_test_filtered.max():.2f}")
    
    preds = model.predict(X_test_filtered)
    print(f"Predictions shape: {preds.shape}")
    print(f"Predictions range: {preds.min():.2f} to {preds.max():.2f}")
    
    plot_feature_importance(model, X_test.columns.tolist())
    plot_predictions_vs_true(y_test_filtered, preds)
    plot_residuals(y_test_filtered, preds)
    plot_residual_histogram(y_test_filtered, preds, "XGBoost")
    plot_qq_plot(y_test_filtered, preds, "XGBoost")
    regression_confusion_matrix(y_test_filtered, preds, bins=5)
    print_metrics(y_test_filtered, preds)
    
    results_df = pd.DataFrame(X_test_filtered, columns=X_test.columns)
    results_df['predicted_price'] = preds
    results_df['true_price'] = y_test_filtered.values
    results_df['error'] = results_df['true_price'] - results_df['predicted_price']
    results_df['abs_error'] = results_df['error'].abs()
    
    print("\nSample of test data with predictions:")
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.width', 120)
    print(results_df.head(10).to_string())
    
    results_df.to_csv("xgboost_predictions.csv", index=False)
    print(f"Full results saved to xgboost_predictions.csv")
    
    print("\n" + "="*50)
    print("SUMMARY:")
    print(f"XGBoost test samples: {len(y_test_filtered)}")
    print(f"XGBoost target range: {y_test_filtered.min():.2f} to {y_test_filtered.max():.2f}")
    print("="*50)

    df = pd.DataFrame(X_test_filtered, columns=X_test.columns)
    df.to_csv("xgboost_test_data.csv", index=False)
