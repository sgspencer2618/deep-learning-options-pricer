import os
import pandas as pd
import xgboost as xgb
import optuna
from sklearn.metrics import root_mean_squared_error
import logging
from src.models.config import FEATURE_DATA_PATH, FEATURE_COLS, TARGET_COL, HYPERPARAM_SPACE, MODEL_SAVE_PATH
from src.utils import path_builder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data():
    """Load the feature data from the specified path."""
    logger.info(f"Loading data from {FEATURE_DATA_PATH}")

    df = pd.read_parquet(FEATURE_DATA_PATH)
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    unseen_split_idx = int(len(X) * 0.8)
    X, y = X.iloc[:unseen_split_idx], y.iloc[:unseen_split_idx]

    logger.info(f"Training on {len(X)} samples, validating on {len(y)} samples")

    return X, y

def load_data_time_split():
    """Load data using EXACT SAME leak-free contract-based splitting as GRU."""
    from src.features.build_features import generate_flat_datasets_for_xgboost
    
    logger.info("Loading leak-free datasets using SAME method as GRU...")
    return generate_flat_datasets_for_xgboost()

def objective(trial):
    """Objective function for Optuna to optimize hyperparameters of the XGBoost model."""
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

    # Use leak-free split
    X_train, y_train, X_val, y_val, X_test, y_test = load_data_time_split()

    logger.info(f"Trial {trial.number}: Training on {len(X_train)} samples, validating on {len(X_val)} samples")
    
    # Initialize lists to track training metrics
    training_rmse = []
    validation_rmse = []
    
    # Use early stopping and capture evaluation results
    model = xgb.XGBRegressor(eval_metric='rmse', **params)
    
    # Fit with evaluation set to get training history
    eval_result = {}
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False
    )

    logger.info("Model training complete. Evaluating on validation set...")
    
    # Calculate final RMSE scores
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)
    train_rmse = root_mean_squared_error(y_train, train_preds)
    val_rmse = root_mean_squared_error(y_val, val_preds)
    
    # Try to get training history from model if available
    try:
        if hasattr(model, 'evals_result_'):
            eval_results = model.evals_result_
            if 'validation_0' in eval_results and 'rmse' in eval_results['validation_0']:
                training_rmse = eval_results['validation_0']['rmse']
            if 'validation_1' in eval_results and 'rmse' in eval_results['validation_1']:
                validation_rmse = eval_results['validation_1']['rmse']
        else:
            # Fallback: just use final scores
            training_rmse = [train_rmse]
            validation_rmse = [val_rmse]
    except:
        # Fallback: just use final scores
        training_rmse = [train_rmse]
        validation_rmse = [val_rmse]
    
    logger.info(f"Final Train RMSE: {train_rmse:.4f}, Val RMSE: {val_rmse:.4f}")
    
    preds = model.predict(X_val)
    rmse = root_mean_squared_error(y_val, preds)
    
    # Store training history for this trial
    trial.set_user_attr('training_rmse', training_rmse)
    trial.set_user_attr('validation_rmse', validation_rmse)
    trial.set_user_attr('final_train_rmse', training_rmse[-1] if training_rmse else None)
    trial.set_user_attr('final_val_rmse', rmse)
    
    return rmse


def main():
    import pickle
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20)
    print("Best RMSE:", study.best_value)
    print("Best Params:", study.best_params)

    # Save study results for analysis
    study_results = {
        'best_value': study.best_value,
        'best_params': study.best_params,
        'trials': []
    }
    
    for trial in study.trials:
        trial_data = {
            'number': trial.number,
            'value': trial.value,
            'params': trial.params,
            'training_rmse': trial.user_attrs.get('training_rmse', []),
            'validation_rmse': trial.user_attrs.get('validation_rmse', []),
            'final_train_rmse': trial.user_attrs.get('final_train_rmse'),
            'final_val_rmse': trial.user_attrs.get('final_val_rmse')
        }
        study_results['trials'].append(trial_data)
    
    # Save study results
    xgb_study_results_path = path_builder("src\\model_files", "xgboost_study_results.pkl")
    os.makedirs("../models", exist_ok=True)
    with open(xgb_study_results_path, "wb") as f:
        pickle.dump(study_results, f)
    logger.info(f"Study results saved to {xgb_study_results_path}")

    # Get data splits
    X_train, y_train, X_val, y_val, X_test, y_test = load_data_time_split()

    # Combine train and validation for final training (like GRU does)
    X_combined = pd.concat([X_train, X_val], axis=0)
    y_combined = pd.concat([y_train, y_val], axis=0)

    # Train final model on train+val data with tracking
    best_params = study.best_params.copy()
    best_params.update({'objective': 'reg:squarederror', 'tree_method': 'hist'})
    
    logger.info("Training final model with best parameters...")
    final_training_rmse = []
    final_test_rmse = []
    
    model = xgb.XGBRegressor(eval_metric='rmse', **best_params)
    model.fit(
        X_combined, y_combined,
        eval_set=[(X_combined, y_combined), (X_test, y_test)],
        verbose=False
    )
    
    # Calculate final RMSE scores for the final model
    combined_preds = model.predict(X_combined)
    test_preds = model.predict(X_test)
    combined_rmse = root_mean_squared_error(y_combined, combined_preds)
    test_rmse = root_mean_squared_error(y_test, test_preds)
    
    final_training_rmse.append(combined_rmse)
    final_test_rmse.append(test_rmse)
    
    logger.info(f"Final model Train RMSE: {combined_rmse:.4f}, Test RMSE: {test_rmse:.4f}")
    
    # Save final training history
    final_history = {
        'training_rmse': final_training_rmse,
        'test_rmse': final_test_rmse,
        'best_params': best_params
    }
    
    xgb_final_history_path = path_builder("src\\model_files", "xgboost_final_training_history.pkl")
    with open(xgb_final_history_path, "wb") as f:
        pickle.dump(final_history, f)
    logger.info(f"Final training history saved to {xgb_final_history_path}")
    
    # Save model
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    model.save_model(MODEL_SAVE_PATH)
    logger.info(f"Model saved at: {MODEL_SAVE_PATH}")
    
    # Evaluate on test set
    test_preds = model.predict(X_test)
    test_rmse = root_mean_squared_error(y_test, test_preds)
    logger.info(f"Final Test RMSE: {test_rmse:.5f}")
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