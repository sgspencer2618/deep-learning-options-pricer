from src.utils import path_builder

FEATURE_DATA_PATH = path_builder('data', 'features_data.parquet')
SCALED_FEATURE_DATA_PATH = path_builder('data', 'scaled_features_data.parquet')
SCALER_DATA_PATH = path_builder('data', 'scaler.pkl')

# removed: 'option_volume', 'last'
#                  'expiration', 
#                  'bid', 
                #  'bid_size', 
                #  'ask', 
                #  'ask_size', 
                #  'open_interest',
                #  'open_x', 
                #  'high_x', 
                #  'low_x', 
                #  'close_x', 
                #  'stock_volume', 
                #  'open_y', 
                #  'high_y', 
                #  'low_y', 
                #  'close_y', 
                #  'volume', 
                #  'log_return', 
                #  'vol_5d', 
                #  'ma_short', 
                #  'ma_long', 
                #  'ma_signal', 
                #  'rsi_14', 
                #  'rsi_overbought', 
                #  'rsi_oversold', 
                #  'rsi_14_norm'

FEATURE_COLS = [
    # Theoretical features
                 'strike', 
                 'option_type_encoded', 
                 'date', 
                 'implied_volatility',
                 'iv_change', 
                 'delta', 
                 'gamma', 
                 'theta', 
                 'vega', 
                 'rho', 
                 'log_moneyness',
                #  'intrinsic_value', 
                #  'is_itm', 
                 'time_to_maturity', 
                 'log_moneyness_norm', 
                 # 'intrinsic_value_norm',
                 'delta_x_iv',
                 'vega_x_ttm',
                 'gamma_x_logm',
                 'theta_x_intrinsic',
    # Stock features
                #  'option_volume', 
                # #  'last',
                #  'expiration', 
                #  'bid',
                #  'bid_size', 
                #  'ask', 
                #  'ask_size', 
                #  'open_interest',
                #  'open_x', 
                #  'high_x', 
                #  'low_x', 
                #  'close_x', 
                #  'stock_volume', 
                #  'open_y', 
                #  'high_y', 
                #  'low_y', 
                #  'close_y', 
                #  'volume', 
                #  'log_return', 
                #  'vol_5d', 
                #  'ma_short', 
                #  'ma_long', 
                #  'ma_signal', 
                #  'rsi_14', 
                #  'rsi_overbought', 
                #  'rsi_oversold', 
                #  'rsi_14_norm'
]
    

TARGET_COL = "mark"

SCALING_COLS = [
                 'strike', 
                 'option_type_encoded', 
                 'date', 
                 'implied_volatility',
                 'iv_change',
                 'mark', 
                 'delta', 
                 'gamma', 
                 'theta', 
                 'vega', 
                 'rho', 
                 'log_moneyness',
                 'time_to_maturity', 
                 'log_moneyness_norm', 
                 'intrinsic_value_norm',
                 'delta_x_iv',
                 'vega_x_ttm',
                 'gamma_x_logm',
                 'theta_x_intrinsic',
]

HYPERPARAM_SPACE = {
    'max_depth': (7, 12),
    'min_child_weight': (1, 10),
    'gamma': (1, 5),
    'subsample': (0.7, 0.9),
    'colsample_bytree': (0.6, 1.0),
    'learning_rate': (0.01, 0.05),
    'n_estimators': (500, 1500)
}

MODEL_SAVE_PATH = path_builder('src\model_files', 'xgb_option_pricer_v4_test.json')
GRU_MODEL_SAVE_PATH = path_builder('src\model_files', 'gru_option_pricer_GRU_2.pt')

GROUP_KEY_COL = ["contractID"]

WINDOW_SIZE = 10

X_TRAIN_PATH = path_builder("data", "X_train.npy")
Y_TRAIN_PATH = path_builder("data", "y_train.npy")
X_VAL_PATH = path_builder("data", "X_val.npy")
Y_VAL_PATH = path_builder("data", "y_val.npy")
X_TEST_PATH = path_builder("data", "X_test.npy")
Y_TEST_PATH = path_builder("data", "y_test.npy")