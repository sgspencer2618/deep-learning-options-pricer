import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from src.models.config import GRU_MODEL_SAVE_PATH, SCALER_DATA_PATH, FEATURE_COLS, FEATURE_DATA_PATH, TARGET_COL, X_TRAIN_PATH, Y_TRAIN_PATH, X_VAL_PATH, Y_VAL_PATH, X_TEST_PATH, Y_TEST_PATH
from src.utils import path_builder

from src.models.attentive_gru import AttentiveGRU
from src.datasets.dataset import OptionSequenceDataset
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, median_absolute_error, explained_variance_score, root_mean_squared_error

import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_model(model, data_loader, scaler_y=None, device="cpu"):
    model.eval()
    model.to(device)
    preds, trues = [], []
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_pred = model(X_batch).cpu().numpy()
            preds.append(y_pred)
            trues.append(y_batch.numpy())
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(trues)
    return y_true, y_pred

def regression_metrics(y_true, y_pred):
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    ev = explained_variance_score(y_true, y_pred)
    print(f"RMSE: {rmse:.5f}")
    print(f"MAE: {mae:.5f}")
    print(f"MedAE: {medae:.5f}")
    print(f"R2: {r2:.5f}")
    print(f"Explained Variance: {ev:.5f}")
    return {
        "RMSE": rmse,
        "MAE": mae,
        "MedAE": medae,
        "R2": r2,
        "Explained Variance": ev,
    }

def plot_predictions(y_true, y_pred, title="True vs Predicted"):
    plt.figure(figsize=(7,7))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.4, s=12)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, "--", color="grey")
    plt.xlabel("True Price")
    plt.ylabel("Predicted Price")
    plt.title(title)
    plt.grid(True)
    plt.show()

def plot_residuals(y_true, y_pred, title="Residuals Plot"):
    residuals = y_pred - y_true
    plt.figure(figsize=(8,5))
    plt.scatter(y_true, residuals, alpha=0.3, s=12)
    plt.axhline(0, linestyle='--', color='black')
    plt.xlabel("True Price")
    plt.ylabel("Residual (Pred - True)")
    plt.title(title)
    plt.grid(True)
    plt.show()

def plot_histograms(y_true, y_pred, title="Prediction Distribution"):
    residuals = y_pred - y_true
    plt.figure(figsize=(14,4))
    plt.subplot(1,3,1)
    sns.histplot(y_true, bins=30, kde=True, color="blue")
    plt.title("True Values")
    plt.subplot(1,3,2)
    sns.histplot(y_pred, bins=30, kde=True, color="green")
    plt.title("Predicted Values")
    plt.subplot(1,3,3)
    sns.histplot(residuals, bins=30, kde=True, color="red")
    plt.title("Residuals")
    plt.tight_layout()
    plt.suptitle(title, y=1.05, fontsize=14)
    plt.show()

def plot_qq(residuals, title="QQ Plot of Residuals"):
    import scipy.stats as stats
    plt.figure(figsize=(6,6))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title(title)
    plt.show()

def plot_residual_histogram(y_true, y_pred, model_name="GRU"):
    """Plot histogram of residuals with clipped range for visibility."""
    residuals = y_true - y_pred
    
    # Clip residuals to [-50, 50] range for visualization only
    residuals_clipped = np.clip(residuals, -50, 50)
    
    plt.figure(figsize=(10, 6))
    plt.hist(residuals_clipped, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
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

def plot_qq_plot(y_true, y_pred, model_name="GRU"):
    """Plot Q-Q plot to check if residuals follow normal distribution."""
    residuals = y_true - y_pred
    
    plt.figure(figsize=(8, 8))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title(f'{model_name} QQ Plot (Residuals vs Normal Distribution)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import pickle
    import torch
    from torch.utils.data import DataLoader

    X_test = np.load(X_TEST_PATH)
    y_test = np.load(Y_TEST_PATH)
    y_train = np.load(Y_TRAIN_PATH)

    with open(SCALER_DATA_PATH, "rb") as f:
        scaler_dict = pickle.load(f)
        print(type(scaler_dict))

    if isinstance(scaler_dict, dict):
        scaler_X = scaler_dict.get('scaler_X')
        scaler_y = scaler_dict.get('scaler_y')
        print("Both X and y scalers loaded successfully")
    else:
        scaler_y = scaler_dict
        scaler_X = None
        print("Only y scaler found (old format)")

    if scaler_X is None:
        scaler_x_path = path_builder("data", "scaler_X.pkl")
        try:
            with open(scaler_x_path, "rb") as f:
                scaler_X = pickle.load(f)
            print("X scaler loaded successfully from separate file")
        except FileNotFoundError:
            print("Warning: X scaler file not found")
            scaler_X = None

    print("=== SCALER DEBUG ===")
    print(f"scaler_dict type: {type(scaler_dict)}")
    print(f"scaler_dict keys: {scaler_dict.keys() if isinstance(scaler_dict, dict) else 'Not a dict'}")
    print(f"scaler_X is None: {scaler_X is None}")
    print(f"scaler_y is None: {scaler_y is None}")

    if scaler_X is not None:
        print(f"scaler_X type: {type(scaler_X)}")
        print(f"scaler_X has mean_: {hasattr(scaler_X, 'mean_')}")
        if hasattr(scaler_X, 'mean_'):
            print(f"scaler_X mean shape: {scaler_X.mean_.shape}")
            print(f"scaler_X first 3 means: {scaler_X.mean_[:3]}")
    print("=====================")

    model = AttentiveGRU(
        input_dim=X_test.shape[-1],
        hidden_dim=128,
        num_layers=4,
        use_attention=True
    )
    model.load_state_dict(torch.load(GRU_MODEL_SAVE_PATH, map_location="cuda", weights_only=False))

    test_ds = OptionSequenceDataset(X_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    y_true, y_pred = evaluate_model(model, test_loader, scaler_y=scaler_y, device="cuda")
    metrics = regression_metrics(y_true, y_pred)

    print("Scaler mean:", scaler_y.mean_[0])
    print("Scaler std:", scaler_y.scale_[0])
    print("y_train mean:", y_train.mean())
    print("y_train std:", y_train.std())

    sample = y_train[0]
    scaled = scaler_y.transform([[sample]])[0,0]
    inverted = scaler_y.inverse_transform([[scaled]])[0,0]
    print(f"Original: {sample}, Scaled: {scaled}, Inverted: {inverted}")

    feature_names = FEATURE_COLS.copy()
    print(f"Feature names (without target): {len(feature_names)} features")

    if len(X_test.shape) == 3:
        X_flat = X_test[:, -1, :]
    else:
        X_flat = X_test

    if X_flat.shape[1] != len(feature_names):
        print(f"Adjusting feature names: received {X_flat.shape[1]} features but got {len(feature_names)} names.")
        feature_names = feature_names[:X_flat.shape[1]]

    print("=== X_FLAT DEBUG ===")
    print(f"X_flat shape: {X_flat.shape}")
    print(f"X_flat first sample first 3 features: {X_flat[0][:3]}")
    print("=====================")

    if scaler_X is not None:
        if X_flat.shape[1] != len(scaler_X.mean_):
            print(f"WARNING: Feature dimension mismatch! X_flat has {X_flat.shape[1]} features but scaler has {len(scaler_X.mean_)} features")
            X_flat_matched = X_flat[:, :len(scaler_X.mean_)]
            print(f"Using only the first {len(scaler_X.mean_)} features for unscaling")
            
            print("DEBUG: scaler_X.mean_ =", scaler_X.mean_)
            print("DEBUG: scaler_X.scale_ =", scaler_X.scale_)
            sample_scaled = X_flat_matched[0]
            sample_unscaled_manual = sample_scaled * scaler_X.scale_ + scaler_X.mean_
            print("DEBUG: First X_flat sample (scaled, matched):", sample_scaled)
            print("DEBUG: First X_flat sample (manually unscaled):", sample_unscaled_manual)
            
            X_flat_unscaled = scaler_X.inverse_transform(X_flat_matched)
            df_results = pd.DataFrame(X_flat_unscaled, columns=feature_names[:len(scaler_X.mean_)])
            print("Features unscaled successfully using scaler_X (after dimension matching)")
        else:
            print("DEBUG: scaler_X.mean_ =", scaler_X.mean_)
            print("DEBUG: scaler_X.scale_ =", scaler_X.scale_)
            sample_scaled = X_flat[0]
            sample_unscaled_manual = sample_scaled * scaler_X.scale_ + scaler_X.mean_
            print("DEBUG: First X_flat sample (scaled):", sample_scaled)
            print("DEBUG: First X_flat sample (manually unscaled):", sample_unscaled_manual)
            
            X_flat_unscaled = scaler_X.inverse_transform(X_flat)
            df_results = pd.DataFrame(X_flat_unscaled, columns=feature_names)
            print("Features unscaled successfully using scaler_X")
    else:
        df_results = pd.DataFrame(X_flat, columns=feature_names)
        print("Warning: scaler_X not available, using scaled features")

    df_results['y_true'] = y_true
    df_results['y_pred'] = y_pred
    df_results['abs_diff'] = df_results['y_pred'] - df_results['y_true']

    rmse = np.sqrt(root_mean_squared_error(df_results['y_true'], df_results['y_pred']))
    mae = mean_absolute_error(df_results['y_true'], df_results['y_pred'])
    medae = median_absolute_error(df_results['y_true'], df_results['y_pred'])
    r2 = r2_score(df_results['y_true'], df_results['y_pred'])

    print(f"RMSE: {rmse:.5f}")
    print(f"MAE: {mae:.5f}")
    print(f"MedAE: {medae:.5f}")
    print(f"R2: {r2:.5f}")

    df_results['abs_diff'] = df_results['abs_diff'].abs()
    print("Number of samples in results DataFrame:", len(df_results))
    condition = df_results['y_true'] < 50
    df_results.drop(df_results[condition].index, inplace=True)
    df_results['pct_error'] = (df_results['y_pred'] - df_results['y_true']) / df_results['y_true'] * 100

    print("Number of samples after dropping y_true < 50:", len(df_results))

    print("gru range:", df_results['y_true'].min(), df_results['y_true'].max())
    print(f"gru eval result columbs: {df_results.columns.tolist()}")

    df_results.to_csv("gru_evaluation_results.csv", index=False)
    print(df_results.head(10))


    plot_predictions(y_true, y_pred, title="GRU Test: True vs Predicted")
    plot_residuals(y_true, y_pred, title="GRU Test: Residuals")
    plot_residual_histogram(y_true, y_pred, "GRU")
    plot_qq_plot(y_true, y_pred, "GRU")
    plot_histograms(y_true, y_pred, title="GRU Test: Value and Error Distribution")
    plot_qq(y_pred - y_true, title="GRU Test: QQ Plot of Residuals")
