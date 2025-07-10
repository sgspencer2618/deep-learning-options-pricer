import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from src.model.config import GRU_MODEL_SAVE_PATH, SCALER_DATA_PATH, FEATURE_COLS
from src.features.build_features import X_test_path, y_test_path, y_train_path

from src.neural.attentive_gru import AttentiveGRU
from src.neural.dataset import OptionSequenceDataset
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, median_absolute_error, explained_variance_score, root_mean_squared_error

import pandas as pd

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
    if scaler_y is not None:
        y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        y_true = scaler_y.inverse_transform(y_true.reshape(-1, 1)).flatten()
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


if __name__ == "__main__":
    import pickle
    import torch
    from torch.utils.data import DataLoader

    # -------- Load data --------
    X_test = np.load(X_test_path)
    y_test = np.load(y_test_path)
    y_train = np.load(y_train_path)

    # load the scaler for y
    with open(SCALER_DATA_PATH, "rb") as f:
        scaler_y = pickle.load(f)

    # -------- Load model --------
    model = AttentiveGRU(
        input_dim=X_test.shape[-1],
        hidden_dim=128,  # match your training config
        num_layers=2,
        use_attention=True
    )
    model.load_state_dict(torch.load(GRU_MODEL_SAVE_PATH, map_location="cuda"))

    # -------- Make predictions --------
    test_ds = OptionSequenceDataset(X_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    # -------- Metrics & Plot --------
    y_true, y_pred = evaluate_model(model, 
                                    test_loader, 
                                    scaler_y=scaler_y, 
                                    device="cuda")
    metrics = regression_metrics(y_true, y_pred)

    print("Scaler mean:", scaler_y.mean_[0])
    print("Scaler std:", scaler_y.scale_[0])
    print("y_train mean:", y_train.mean())
    print("y_train std:", y_train.std())

    sample = y_train[0]
    scaled = scaler_y.transform([[sample]])[0,0]
    inverted = scaler_y.inverse_transform([[scaled]])[0,0]
    print(f"Original: {sample}, Scaled: {scaled}, Inverted: {inverted}")


    # Your feature names (ordered as in your data!)
    feature_names = FEATURE_COLS

    # If X_test is 3D [N, window_size, num_features], just take last row for each window:
    if len(X_test.shape) == 3:
        # For sequence models: take only the last time step for each sample
        X_flat = X_test[:, -1, :]
    else:
        X_flat = X_test

    # Create the DataFrame
    df_results = pd.DataFrame(X_flat, columns=feature_names)

    # Add y_test, y_pred, and delta (prediction error)
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
    #df_results.sort_values(by='abs_diff', ascending=True, inplace=True)


    # See the first 10 rows
    print(df_results.head(10))


        # plot_predictions(y_true, y_pred, title="GRU Test: True vs Predicted")
        # plot_residuals(y_true, y_pred, title="GRU Test: Residuals")
        # plot_histograms(y_true, y_pred, title="GRU Test: Value and Error Distribution")
        # plot_qq(y_pred - y_true, title="GRU Test: QQ Plot of Residuals")
