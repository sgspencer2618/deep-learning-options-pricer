import os
import pandas as pd
import xgboost as xgb
import optuna
from sklearn.metrics import root_mean_squared_error
import logging
from config import FEATURE_DATA_PATH, FEATURE_COLS, TARGET_COL, HYPERPARAM_SPACE, MODEL_SAVE_PATH

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data():
    """
    Load the feature data from the specified path.
    
    Returns:
        pd.DataFrame: DataFrame containing the feature data.
    """
    logger.info(f"Loading data from {FEATURE_DATA_PATH}")

    df = pd.read_parquet(FEATURE_DATA_PATH)
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    return X, y

def objective(trial):
    """
    Objective function for Optuna to optimize hyperparameters of the XGBoost model.
    """
    params = {
        'max_depth': trial.suggest_int('max_depth', *HYPERPARAM_SPACE['max_depth']),
        'min_child_weight': trial.suggest_int('min_child_weight', *HYPERPARAM_SPACE['min_child_weight']),
        'gamma': trial.suggest_float('gamma', *HYPERPARAM_SPACE['gamma'], log=True),
        'subsample': trial.suggest_float('subsample', *HYPERPARAM_SPACE['subsample']),
        'colsample_bytree': trial.suggest_float('colsample_bytree', *HYPERPARAM_SPACE['colsample_bytree']),
        'learning_rate': trial.suggest_float('learning_rate', *HYPERPARAM_SPACE['learning_rate'], log=True),
        'n_estimators': trial.suggest_int('n_estimators', *HYPERPARAM_SPACE['n_estimators']),
        'objective': 'reg:squarederror',
        'tree_method': 'hist',  # Fastest on most modern machines,
        'enable_categorical': True,  # Enable categorical feature support
        'early_stopping_rounds': 30,  # Early stopping to prevent overfitting
    }

    X, y = load_data()
    # Time-wise split
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    logger.info(f"Training on {len(X_train)} samples, validating on {len(X_val)} samples")
    model = xgb.XGBRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=True
    )

    logger.info("Model training complete. Evaluating on validation set...")
    preds = model.predict(X_val)
    rmse = root_mean_squared_error(y_val, preds)
    return rmse


def main():
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20)
    print("Best RMSE:", study.best_value)
    print("Best Params:", study.best_params)

    # Retrain on full data, save model
    X, y = load_data()
    model = xgb.XGBRegressor(**study.best_params, objective='reg:squarederror', tree_method='hist')
    model.fit(X, y)
    model.save_model(MODEL_SAVE_PATH)
    print(f"Model saved at: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    logger.info(f"Model save path: {MODEL_SAVE_PATH}")
    path_exists = os.path.exists(MODEL_SAVE_PATH)
    logger.info(f"Model path exists: {path_exists}")

    if path_exists:
        main()
    else:
        logger.error(f"Model save path does not exist: {MODEL_SAVE_PATH}. Please check the configuration.")
        raise FileNotFoundError(f"Model save path does not exist: {MODEL_SAVE_PATH}")