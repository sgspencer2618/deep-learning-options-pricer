import os
import pandas as pd
import xgboost as xgb
import optuna
from sklearn.metrics import root_mean_squared_error
import logging
from src.models.config import FEATURE_DATA_PATH, FEATURE_COLS, TARGET_COL, HYPERPARAM_SPACE, MODEL_SAVE_PATH

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

    unseen_split_idx = int(len(X) * 0.8)
    X, y = X.iloc[:unseen_split_idx], y.iloc[:unseen_split_idx]

    logger.info(f"Training on {len(X)} samples, validating on {len(y)} samples")

    return X, y

def load_data_time_split():
    """
    Load data and split using time-based approach matching GRU.
    
    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    logger.info(f"Loading data from {FEATURE_DATA_PATH}")
    df = pd.read_parquet(FEATURE_DATA_PATH)
    
    # Time-based split exactly like GRU
    unique_dates = df['date'].sort_values().unique()
    n_dates = len(unique_dates)
    
    # Same ratios as GRU: 70/15/15
    train_idx = int(n_dates * 0.7)
    val_idx = int(n_dates * 0.85)
    
    # Split dates chronologically
    train_dates = unique_dates[:train_idx]
    val_dates = unique_dates[train_idx:val_idx]
    test_dates = unique_dates[val_idx:]
    
    logger.info(f"Train dates: {train_dates[0]} to {train_dates[-1]} ({len(train_dates)} dates)")
    logger.info(f"Val dates: {val_dates[0]} to {val_dates[-1]} ({len(val_dates)} dates)")
    logger.info(f"Test dates: {test_dates[0]} to {test_dates[-1]} ({len(test_dates)} dates)")
    
    # Filter data by dates
    train_df = df[df['date'].isin(train_dates)]
    val_df = df[df['date'].isin(val_dates)]
    test_df = df[df['date'].isin(test_dates)]
    
    # Extract features and targets
    X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET_COL]
    X_val, y_val = val_df[FEATURE_COLS], val_df[TARGET_COL]
    X_test, y_test = test_df[FEATURE_COLS], test_df[TARGET_COL]
    
    logger.info(f"Train samples: {len(X_train)}")
    logger.info(f"Val samples: {len(X_val)}")
    logger.info(f"Test samples: {len(X_test)}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test

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
        'tree_method': 'hist',
        'enable_categorical': True,
        'early_stopping_rounds': 30,
    }

    # Use time-based split
    X_train, y_train, X_val, y_val, X_test, y_test = load_data_time_split()

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

    # Get data splits
    X_train, y_train, X_val, y_val, X_test, y_test = load_data_time_split()

    # Combine train and validation for final training (like GRU does)
    X_combined = pd.concat([X_train, X_val], axis=0)
    y_combined = pd.concat([y_train, y_val], axis=0)

    # Train final model on train+val data
    best_params = study.best_params.copy()
    best_params.update({'objective': 'reg:squarederror', 'tree_method': 'hist'})
    
    model = xgb.XGBRegressor(**best_params)
    model.fit(X_combined, y_combined)
    
    # Save model
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    model.save_model(MODEL_SAVE_PATH)
    logger.info(f"Model saved at: {MODEL_SAVE_PATH}")
    
    # Evaluate on test set
    test_preds = model.predict(X_test)
    test_rmse = root_mean_squared_error(y_test, test_preds)
    logger.info(f"Test RMSE: {test_rmse:.5f}")
    logger.info(f"Test target range: {y_test.min():.2f} to {y_test.max():.2f}")

if __name__ == "__main__":
    logger.info(f"Model save path: {MODEL_SAVE_PATH}")
    path_exists = os.path.exists(MODEL_SAVE_PATH)
    logger.info(f"Model path exists: {path_exists}")

    if path_exists:
        main()
    else:
        logger.error(f"Model save path does not exist: {MODEL_SAVE_PATH}. Please check the configuration.")
        raise FileNotFoundError(f"Model save path does not exist: {MODEL_SAVE_PATH}")