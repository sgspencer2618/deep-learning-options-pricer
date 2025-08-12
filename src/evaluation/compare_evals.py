import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pickle
import os
from src.utils import path_builder

def find_matching_rows():
    """Find rows that can be matched between XGBoost and GRU evaluation datasets."""
    xgboost_df = pd.read_csv('xgboost_predictions.csv')
    gru_df = pd.read_csv('gru_evaluation_results.csv')
    mlp_df = pd.read_csv('mlp_predictions.csv')

    print("XGBoost columns:", xgboost_df.columns.tolist())
    print("GRU columns:", gru_df.columns.tolist())
    print("MLP columns:", mlp_df.columns.tolist())

    if 'true_price' in xgboost_df.columns:
        xgboost_df = xgboost_df.rename(columns={'true_price': 'y_true', 'predicted_price': 'y_pred', 'abs_error': 'abs_error_xgb'})
    
    if 'y_true' not in xgboost_df.columns or 'y_true' not in gru_df.columns:
        raise ValueError("Could not find 'y_true' or 'true_price' column in one or both datasets")

    gru_df = gru_df.rename(columns={'abs_diff': 'abs_error_gru'})
    mlp_df = mlp_df.rename(columns={'true_price': 'y_true', 'predicted_price': 'y_pred', 'abs_error': 'abs_error_mlp'})

    xgb_value_counts = xgboost_df['y_true'].value_counts()
    gru_value_counts = gru_df['y_true'].value_counts()
    mlp_value_counts = mlp_df['y_true'].value_counts()

    xgb_unique_values = xgb_value_counts[xgb_value_counts == 1].index.tolist()
    gru_unique_values = gru_value_counts[gru_value_counts == 1].index.tolist()
    mlp_unique_values = mlp_value_counts[mlp_value_counts == 1].index.tolist()

    common_unique_values = set(xgb_unique_values).intersection(set(gru_unique_values)).intersection(set(mlp_unique_values))

    print(f"Found {len(xgb_unique_values)} unique values in XGBoost dataset")
    print(f"Found {len(gru_unique_values)} unique values in GRU dataset")
    print(f"Found {len(mlp_unique_values)} unique values in MLP dataset")
    print(f"Found {len(common_unique_values)} common unique values between datasets")
    
    xgb_matching_rows = xgboost_df[xgboost_df['y_true'].isin(common_unique_values)]
    gru_matching_rows = gru_df[gru_df['y_true'].isin(common_unique_values)]
    mlp_matching_rows = mlp_df[mlp_df['y_true'].isin(common_unique_values)]
    
    print(f"XGBoost matching rows: {len(xgb_matching_rows)}")
    print(f"GRU matching rows: {len(gru_matching_rows)}")
    print(f"MLP matching rows: {len(mlp_matching_rows)}")

    merged_df = pd.merge(
        xgb_matching_rows[['y_true', 'y_pred', 'abs_error_xgb']], 
        gru_matching_rows[['y_true', 'y_pred', 'abs_error_gru']],
        on='y_true',
        suffixes=('_xgb', '_gru')
    )

    merged_df = pd.merge(
        merged_df,
        mlp_matching_rows[['y_true', 'y_pred', 'abs_error_mlp']],
        on='y_true',
    )

    merged_df = merged_df.rename(columns={'y_pred': 'y_pred_mlp',})

    print(merged_df.columns.tolist())
    
    print(merged_df.columns.tolist())
    print(merged_df.head())
    merged_df['avg_abs_error'] = (merged_df['abs_error_xgb'] + merged_df['abs_error_gru']) + merged_df['abs_error_mlp'] / 3

    merged_df['pct_error_xgb'] = (merged_df['abs_error_xgb'] / merged_df['y_true']) * 100
    merged_df['pct_error_gru'] = (merged_df['abs_error_gru'] / merged_df['y_true']) * 100
    merged_df['pct_error_mlp'] = (merged_df['abs_error_mlp'] / merged_df['y_true']) * 100
    
    merged_df['avg_pct_error'] = (merged_df['pct_error_xgb'] + merged_df['pct_error_gru'] + merged_df['pct_error_mlp']) / 3

    merged_df = merged_df.sort_values('avg_abs_error')
    
    print(f"Merged dataset rows: {len(merged_df)}")
    xgb_matching_rows = xgb_matching_rows.set_index('y_true').loc[merged_df['y_true']].reset_index()
    gru_matching_rows = gru_matching_rows.set_index('y_true').loc[merged_df['y_true']].reset_index()
    mlp_matching_rows = mlp_matching_rows.set_index('y_true').loc[merged_df['y_true']].reset_index()
    
    # Save the results
    xgb_matching_rows.to_csv('src/evaluation/xgboost_matching_rows.csv', index=False)
    gru_matching_rows.to_csv('src/evaluation/gru_matching_rows.csv', index=False)
    mlp_matching_rows.to_csv('src/evaluation/mlp_matching_rows.csv', index=False)
    merged_df.to_csv('src/evaluation/merged_matching_rows.csv', index=False)

    # Return a sample of matching rows for inspection (best and worst predictions)
    print("\nBest predictions (lowest average error):")
    for i in range(min(5, len(merged_df))):
        value = merged_df.iloc[i]['y_true']
        print(f"\ny_true = {value}, avg_abs_error = {merged_df.iloc[i]['avg_abs_error']:.4f}")
        print(f"XGBoost: pred = {merged_df.iloc[i]['y_pred_xgb']:.4f}, error = {merged_df.iloc[i]['abs_error_xgb']:.4f}")
        print(f"GRU: pred = {merged_df.iloc[i]['y_pred_gru']:.4f}, error = {merged_df.iloc[i]['abs_error_gru']:.4f}")
        print(f"MLP: pred = {merged_df.iloc[i]['y_pred_mlp']:.4f}, error = {merged_df.iloc[i]['abs_error_mlp']:.4f}")
    
    print("\nWorst predictions (highest average error):")
    for i in range(1, min(6, len(merged_df) + 1)):
        idx = -i
        value = merged_df.iloc[idx]['y_true']
        print(f"\ny_true = {value}, avg_abs_error = {merged_df.iloc[idx]['avg_abs_error']:.4f}")
        print(f"XGBoost: pred = {merged_df.iloc[idx]['y_pred_xgb']:.4f}, error = {merged_df.iloc[idx]['abs_error_xgb']:.4f}")
        print(f"GRU: pred = {merged_df.iloc[idx]['y_pred_gru']:.4f}, error = {merged_df.iloc[idx]['abs_error_gru']:.4f}")
        print(f"MLP: pred = {merged_df.iloc[idx]['y_pred_mlp']:.4f}, error = {merged_df.iloc[idx]['abs_error_mlp']:.4f}")

    return xgb_matching_rows, gru_matching_rows, mlp_matching_rows, merged_df

def load_training_history(model_name):
    """Load training history for a specific model."""
    history_files = {
        'MLP': path_builder("src\\model_files", "mlp_training_history.pkl"),
        'GRU': path_builder("src\\model_files", "gru_training_history.pkl")
    }
    
    if model_name not in history_files:
        print(f"Unknown model: {model_name}")
        return None
    
    file_path = history_files[model_name]
    if not os.path.exists(file_path):
        print(f"History file not found: {file_path}")
        return None
    
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def plot_training_performance_comparison():
    """
    Plot training performance time series for MLP and GRU models.
    """
    # Load training histories
    mlp_history = load_training_history('MLP')
    gru_history = load_training_history('GRU')
    
    if not mlp_history and not gru_history:
        print("No training history found for MLP or GRU models.")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Performance Comparison: MLP vs GRU', fontsize=16, fontweight='bold')
    
    # Plot 1: Training Loss
    axes[0, 0].set_title('Training Loss Over Time')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    if mlp_history and 'train_loss' in mlp_history:
        epochs = range(1, len(mlp_history['train_loss']) + 1)
        axes[0, 0].plot(epochs, mlp_history['train_loss'], label='MLP Train Loss', 
                       linewidth=2, color='lightcoral')
    
    if gru_history and 'train_loss' in gru_history:
        epochs = range(1, len(gru_history['train_loss']) + 1)
        axes[0, 0].plot(epochs, gru_history['train_loss'], label='GRU Train Loss', 
                       linewidth=2, color='lightgreen')
    
    axes[0, 0].legend()
    axes[0, 0].set_yscale('log')
    
    # Plot 2: Validation Loss
    axes[0, 1].set_title('Validation Loss Over Time')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True, alpha=0.3)
    
    if mlp_history and 'val_loss' in mlp_history:
        epochs = range(1, len(mlp_history['val_loss']) + 1)
        axes[0, 1].plot(epochs, mlp_history['val_loss'], label='MLP Val Loss', 
                       linewidth=2, color='darkred', linestyle='--')
    
    if gru_history and 'val_loss' in gru_history:
        epochs = range(1, len(gru_history['val_loss']) + 1)
        axes[0, 1].plot(epochs, gru_history['val_loss'], label='GRU Val Loss', 
                       linewidth=2, color='darkgreen', linestyle='--')
    
    axes[0, 1].legend()
    axes[0, 1].set_yscale('log')
    
    # Plot 3: Training RMSE
    axes[1, 0].set_title('Training RMSE Over Time')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('RMSE')
    axes[1, 0].grid(True, alpha=0.3)
    
    if mlp_history and 'train_rmse' in mlp_history:
        epochs = range(1, len(mlp_history['train_rmse']) + 1)
        axes[1, 0].plot(epochs, mlp_history['train_rmse'], label='MLP Train RMSE', 
                       linewidth=2, color='lightcoral')
    
    if gru_history and 'train_rmse' in gru_history:
        epochs = range(1, len(gru_history['train_rmse']) + 1)
        axes[1, 0].plot(epochs, gru_history['train_rmse'], label='GRU Train RMSE', 
                       linewidth=2, color='lightgreen')
    
    axes[1, 0].legend()
    
    # Plot 4: Validation RMSE
    axes[1, 1].set_title('Validation RMSE Over Time')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('RMSE')
    axes[1, 1].grid(True, alpha=0.3)
    
    if mlp_history and 'val_rmse' in mlp_history:
        epochs = range(1, len(mlp_history['val_rmse']) + 1)
        axes[1, 1].plot(epochs, mlp_history['val_rmse'], label='MLP Val RMSE', 
                       linewidth=2, color='darkred', linestyle='--')
    
    if gru_history and 'val_rmse' in gru_history:
        epochs = range(1, len(gru_history['val_rmse']) + 1)
        axes[1, 1].plot(epochs, gru_history['val_rmse'], label='GRU Val RMSE', 
                       linewidth=2, color='darkgreen', linestyle='--')
    
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(path_builder('assets', 'training_performance_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_model_comparison_charts(merged_df):
    """
    Create comprehensive comparison charts for the three models.
    """
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Error Distribution Comparison (Side-by-side histograms)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Absolute Error Histograms
    axes[0, 0].hist([merged_df['abs_error_xgb'], merged_df['abs_error_gru'], merged_df['abs_error_mlp']], 
                    bins=30, alpha=0.7, label=['XGBoost', 'GRU', 'MLP'], color=['skyblue', 'lightgreen', 'lightcoral'])
    axes[0, 0].set_xlabel('Absolute Error')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Absolute Error Distribution Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Percentage Error Histograms (clipped to reasonable range)
    pct_errors_clipped = [
        np.clip(merged_df['pct_error_xgb'], 0, 100),
        np.clip(merged_df['pct_error_gru'], 0, 100),
        np.clip(merged_df['pct_error_mlp'], 0, 100)
    ]
    axes[0, 1].hist(pct_errors_clipped, bins=30, alpha=0.7, 
                    label=['XGBoost', 'GRU', 'MLP'], color=['skyblue', 'lightgreen', 'lightcoral'])
    axes[0, 1].set_xlabel('Percentage Error (%)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Percentage Error Distribution (Clipped to 0-100%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Box plots for error comparison
    error_data = [merged_df['abs_error_xgb'], merged_df['abs_error_gru'], merged_df['abs_error_mlp']]
    box_plot = axes[1, 0].boxplot(error_data, labels=['XGBoost', 'GRU', 'MLP'], patch_artist=True)
    colors = ['skyblue', 'lightgreen', 'lightcoral']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
    axes[1, 0].set_ylabel('Absolute Error')
    axes[1, 0].set_title('Error Distribution Box Plots')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Average error by price range
    # Bin the true prices and calculate average errors
    merged_df['price_bin'] = pd.cut(merged_df['y_true'], bins=10, labels=False)
    avg_errors_by_bin = merged_df.groupby('price_bin')[['abs_error_xgb', 'abs_error_gru', 'abs_error_mlp']].mean()
    
    x_pos = range(len(avg_errors_by_bin))
    width = 0.25
    axes[1, 1].bar([x - width for x in x_pos], avg_errors_by_bin['abs_error_xgb'], 
                   width, label='XGBoost', color='skyblue', alpha=0.8)
    axes[1, 1].bar(x_pos, avg_errors_by_bin['abs_error_gru'], 
                   width, label='GRU', color='lightgreen', alpha=0.8)
    axes[1, 1].bar([x + width for x in x_pos], avg_errors_by_bin['abs_error_mlp'], 
                   width, label='MLP', color='lightcoral', alpha=0.8)
    
    axes[1, 1].set_xlabel('Price Bin (Low to High)')
    axes[1, 1].set_ylabel('Average Absolute Error')
    axes[1, 1].set_title('Average Error by Price Range')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(path_builder('assets', 'model_comparison_overview.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_prediction_scatter_matrix(merged_df):
    """
    Create scatter plots comparing predictions between models.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # XGBoost vs GRU
    axes[0, 0].scatter(merged_df['y_pred_xgb'], merged_df['y_pred_gru'], alpha=0.6, s=20)
    min_val = min(merged_df['y_pred_xgb'].min(), merged_df['y_pred_gru'].min())
    max_val = max(merged_df['y_pred_xgb'].max(), merged_df['y_pred_gru'].max())
    axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
    axes[0, 0].set_xlabel('XGBoost Predictions')
    axes[0, 0].set_ylabel('GRU Predictions')
    axes[0, 0].set_title('XGBoost vs GRU Predictions')
    axes[0, 0].grid(True, alpha=0.3)
    
    # XGBoost vs MLP
    axes[0, 1].scatter(merged_df['y_pred_xgb'], merged_df['y_pred_mlp'], alpha=0.6, s=20, color='coral')
    min_val = min(merged_df['y_pred_xgb'].min(), merged_df['y_pred_mlp'].min())
    max_val = max(merged_df['y_pred_xgb'].max(), merged_df['y_pred_mlp'].max())
    axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
    axes[0, 1].set_xlabel('XGBoost Predictions')
    axes[0, 1].set_ylabel('MLP Predictions')
    axes[0, 1].set_title('XGBoost vs MLP Predictions')
    axes[0, 1].grid(True, alpha=0.3)
    
    # GRU vs MLP
    axes[0, 2].scatter(merged_df['y_pred_gru'], merged_df['y_pred_mlp'], alpha=0.6, s=20, color='lightgreen')
    min_val = min(merged_df['y_pred_gru'].min(), merged_df['y_pred_mlp'].min())
    max_val = max(merged_df['y_pred_gru'].max(), merged_df['y_pred_mlp'].max())
    axes[0, 2].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
    axes[0, 2].set_xlabel('GRU Predictions')
    axes[0, 2].set_ylabel('MLP Predictions')
    axes[0, 2].set_title('GRU vs MLP Predictions')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Error vs True Value scatter plots
    axes[1, 0].scatter(merged_df['y_true'], merged_df['abs_error_xgb'], alpha=0.6, s=20, color='skyblue')
    axes[1, 0].set_xlabel('True Values')
    axes[1, 0].set_ylabel('Absolute Error')
    axes[1, 0].set_title('XGBoost: Error vs True Value')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].scatter(merged_df['y_true'], merged_df['abs_error_gru'], alpha=0.6, s=20, color='lightgreen')
    axes[1, 1].set_xlabel('True Values')
    axes[1, 1].set_ylabel('Absolute Error')
    axes[1, 1].set_title('GRU: Error vs True Value')
    axes[1, 1].grid(True, alpha=0.3)
    
    axes[1, 2].scatter(merged_df['y_true'], merged_df['abs_error_mlp'], alpha=0.6, s=20, color='lightcoral')
    axes[1, 2].set_xlabel('True Values')
    axes[1, 2].set_ylabel('Absolute Error')
    axes[1, 2].set_title('MLP: Error vs True Value')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(path_builder('assets', 'prediction_scatter_matrix.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_residual_comparison(merged_df):
    """
    Create residual comparison plots across models with centered axes around mean.
    """
    # Calculate residuals
    merged_df['residual_xgb'] = merged_df['y_true'] - merged_df['y_pred_xgb']
    merged_df['residual_gru'] = merged_df['y_true'] - merged_df['y_pred_gru']
    merged_df['residual_mlp'] = merged_df['y_true'] - merged_df['y_pred_mlp']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    colors = ['skyblue', 'lightgreen', 'lightcoral']
    titles = ['XGBoost Residuals', 'GRU Residuals', 'MLP Residuals']
    model_names = ['xgb', 'gru', 'mlp']
    
    # Residual histograms with centered axes around mean
    for i, (color, title, model) in enumerate(zip(colors, titles, model_names)):
        residuals = merged_df[f'residual_{model}']
        mean_res = np.mean(residuals)
        std_res = np.std(residuals)
        
        # Set axis range centered around mean with 50 unit range on each side
        x_min = mean_res - 50
        x_max = mean_res + 50
        
        # Clip residuals for visualization only (keep original for statistics)
        residuals_clipped = np.clip(residuals, x_min, x_max)
        
        axes[0, i].hist(residuals_clipped, bins=40, alpha=0.7, color=color, edgecolor='black')
        axes[0, i].axvline(mean_res, color='red', linestyle='--', alpha=0.8, linewidth=2, label=f'Mean: {mean_res:.2f}')
        axes[0, i].axvline(0, color='black', linestyle='-', alpha=0.5, linewidth=1, label='Zero')
        
        axes[0, i].set_xlabel(f'Residuals [Range: {x_min:.1f} to {x_max:.1f}]')
        axes[0, i].set_ylabel('Frequency')
        axes[0, i].set_title(f'{title} Distribution (Centered on Mean)')
        axes[0, i].grid(True, alpha=0.3)
        axes[0, i].set_xlim(x_min, x_max)
        
        # Add statistics text
        outliers_below = np.sum(residuals < x_min)
        outliers_above = np.sum(residuals > x_max)
        total_outliers = outliers_below + outliers_above
        
        stats_text = f'Mean: {mean_res:.2f}\nStd: {std_res:.2f}\nOutliers: {total_outliers}/{len(residuals)}'
        axes[0, i].text(0.02, 0.98, stats_text,
                       transform=axes[0, i].transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        axes[0, i].legend(loc='upper right')
    
    # Q-Q plots
    for i, model in enumerate(['xgb', 'gru', 'mlp']):
        residuals = merged_df[f'residual_{model}']
        stats.probplot(residuals, dist="norm", plot=axes[1, i])
        axes[1, i].set_title(f'{titles[i]} Q-Q Plot')
        axes[1, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(path_builder('assets', 'residual_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_model_performance_summary(merged_df):
    """
    Create a summary dashboard of model performance metrics.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Calculate summary statistics
    models = ['XGBoost', 'GRU', 'MLP']
    error_cols = ['abs_error_xgb', 'abs_error_gru', 'abs_error_mlp']
    pct_error_cols = ['pct_error_xgb', 'pct_error_gru', 'pct_error_mlp']
    
    mean_abs_errors = [merged_df[col].mean() for col in error_cols]
    median_abs_errors = [merged_df[col].median() for col in error_cols]
    mean_pct_errors = [merged_df[col].mean() for col in pct_error_cols]
    std_abs_errors = [merged_df[col].std() for col in error_cols]
    
    # 1. Mean Absolute Error comparison
    x_pos = np.arange(len(models))
    axes[0, 0].bar(x_pos, mean_abs_errors, color=['skyblue', 'lightgreen', 'lightcoral'], alpha=0.8)
    axes[0, 0].set_ylabel('Mean Absolute Error')
    axes[0, 0].set_title('Mean Absolute Error by Model')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(models)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(mean_abs_errors):
        axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # 2. Error standard deviation comparison
    axes[0, 1].bar(x_pos, std_abs_errors, color=['skyblue', 'lightgreen', 'lightcoral'], alpha=0.8)
    axes[0, 1].set_ylabel('Standard Deviation of Absolute Error')
    axes[0, 1].set_title('Error Variability by Model')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(models)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(std_abs_errors):
        axes[0, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # 3. Mean vs Median Error comparison
    width = 0.35
    axes[1, 0].bar([x - width/2 for x in x_pos], mean_abs_errors, width, 
                   label='Mean', color=['skyblue', 'lightgreen', 'lightcoral'], alpha=0.8)
    axes[1, 0].bar([x + width/2 for x in x_pos], median_abs_errors, width,
                   label='Median', color=['darkblue', 'darkgreen', 'darkred'], alpha=0.8)
    axes[1, 0].set_ylabel('Absolute Error')
    axes[1, 0].set_title('Mean vs Median Absolute Error')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(models)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Win rate matrix (which model performs better pairwise)
    win_rates = {}
    win_rates['XGB vs GRU'] = (merged_df['abs_error_xgb'] < merged_df['abs_error_gru']).mean() * 100
    win_rates['XGB vs MLP'] = (merged_df['abs_error_xgb'] < merged_df['abs_error_mlp']).mean() * 100
    win_rates['GRU vs XGB'] = (merged_df['abs_error_gru'] < merged_df['abs_error_xgb']).mean() * 100
    win_rates['GRU vs MLP'] = (merged_df['abs_error_gru'] < merged_df['abs_error_mlp']).mean() * 100
    win_rates['MLP vs XGB'] = (merged_df['abs_error_mlp'] < merged_df['abs_error_xgb']).mean() * 100
    win_rates['MLP vs GRU'] = (merged_df['abs_error_mlp'] < merged_df['abs_error_gru']).mean() * 100
    
    comparisons = list(win_rates.keys())
    rates = list(win_rates.values())
    
    axes[1, 1].barh(comparisons, rates, color='lightsteelblue', alpha=0.8)
    axes[1, 1].set_xlabel('Win Rate (%)')
    axes[1, 1].set_title('Pairwise Model Win Rates')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axvline(50, color='red', linestyle='--', alpha=0.8)
    
    # Add value labels
    for i, v in enumerate(rates):
        axes[1, 1].text(v + 1, i, f'{v:.1f}%', ha='left', va='center')
    
    plt.tight_layout()
    plt.savefig(path_builder('assets', 'model_performance_summary.png'), dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    xgb_matches, gru_matches, mlp_matches, merged = find_matching_rows()
    print(f"\nAnalysis complete. Found {len(merged)} matching rows between datasets.")
    
    # Print additional statistics about error distribution
    print("\nError Statistics:")
    print(f"Average XGBoost abs error: {merged['abs_error_xgb'].mean():.4f}")
    print(f"Average GRU abs error: {merged['abs_error_gru'].mean():.4f}")
    print(f"Average MLP abs error: {merged['abs_error_mlp'].mean():.4f}")
    print(f"XGB better prediction count: {(merged['abs_error_xgb'] < merged['abs_error_gru']).sum()}")
    print(f"GRU better prediction count: {(merged['abs_error_gru'] < merged['abs_error_xgb']).sum()}")
    print(f"Tie count: {(merged['abs_error_xgb'] == merged['abs_error_gru']).sum()}")
    
    # Print percentage error statistics
    print("\nPercentage Error Statistics:")
    print(f"Average XGBoost percentage error: {merged['pct_error_xgb'].mean():.2f}%")
    print(f"Average GRU percentage error: {merged['pct_error_gru'].mean():.2f}%")
    print(f"Average MLP percentage error: {merged['pct_error_mlp'].mean():.2f}%")
    
    # Generate comprehensive comparison visualizations
    print("\nGenerating comparison visualizations...")
    
    # 1. Training performance time series comparison
    print("Plotting training performance comparison...")
    plot_training_performance_comparison()
    
    # 2. Overall model comparison charts
    print("Plotting model comparison charts...")
    plot_model_comparison_charts(merged)
    
    # 3. Residual comparison analysis (centered on mean)
    print("Plotting residual comparison...")
    plot_residual_comparison(merged)
    
    print("\nVisualization complete! Check the following files:")
    print("- src/evaluation/training_performance_comparison.png")
    print("- src/evaluation/model_comparison_overview.png")
    print("- src/evaluation/residual_comparison.png")