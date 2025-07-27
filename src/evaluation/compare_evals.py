import pandas as pd
import numpy as np

def find_matching_rows():
    """
    Find rows that can be matched between XGBoost and GRU evaluation datasets
    based on unique values in the true_price/y_true column.
    """
    # Load the datasets
    xgboost_df = pd.read_csv('xgboost_predictions.csv')
    gru_df = pd.read_csv('gru_evaluation_results.csv')
    mlp_df = pd.read_csv('mlp_predictions.csv')

    # Check column names to identify the target column in each dataset
    print("XGBoost columns:", xgboost_df.columns.tolist())
    print("GRU columns:", gru_df.columns.tolist())
    print("MLP columns:", mlp_df.columns.tolist())

    # Rename target column for consistency
    if 'true_price' in xgboost_df.columns:
        xgboost_df = xgboost_df.rename(columns={'true_price': 'y_true', 'predicted_price': 'y_pred', 'abs_error': 'abs_error_xgb'})
    
    if 'y_true' not in xgboost_df.columns or 'y_true' not in gru_df.columns:
        raise ValueError("Could not find 'y_true' or 'true_price' column in one or both datasets")

    gru_df = gru_df.rename(columns={'abs_diff': 'abs_error_gru'})
    mlp_df = mlp_df.rename(columns={'true_price': 'y_true', 'predicted_price': 'y_pred', 'abs_error': 'abs_error_mlp'})

    # Find values that appear exactly once in each dataset
    xgb_value_counts = xgboost_df['y_true'].value_counts()
    gru_value_counts = gru_df['y_true'].value_counts()
    mlp_value_counts = mlp_df['y_true'].value_counts()

    # Get the list of values that appear exactly once in each dataset
    xgb_unique_values = xgb_value_counts[xgb_value_counts == 1].index.tolist()
    gru_unique_values = gru_value_counts[gru_value_counts == 1].index.tolist()
    mlp_unique_values = mlp_value_counts[mlp_value_counts == 1].index.tolist()

    # Find the intersection of these unique values
    common_unique_values = set(xgb_unique_values).intersection(set(gru_unique_values)).intersection(set(mlp_unique_values))

    print(f"Found {len(xgb_unique_values)} unique values in XGBoost dataset")
    print(f"Found {len(gru_unique_values)} unique values in GRU dataset")
    print(f"Found {len(mlp_unique_values)} unique values in MLP dataset")
    print(f"Found {len(common_unique_values)} common unique values between datasets")
    
    # Get the rows with these common unique values
    xgb_matching_rows = xgboost_df[xgboost_df['y_true'].isin(common_unique_values)]
    gru_matching_rows = gru_df[gru_df['y_true'].isin(common_unique_values)]
    mlp_matching_rows = mlp_df[mlp_df['y_true'].isin(common_unique_values)]
    
    # Verify the counts match
    print(f"XGBoost matching rows: {len(xgb_matching_rows)}")
    print(f"GRU matching rows: {len(gru_matching_rows)}")
    print(f"MLP matching rows: {len(mlp_matching_rows)}")

    # Create a merged dataset with matching rows
    merged_df = pd.merge(
        xgb_matching_rows[['y_true', 'y_pred', 'abs_error_xgb']], 
        gru_matching_rows[['y_true', 'y_pred', 'abs_error_gru']],
        on='y_true',
        suffixes=('_xgb', '_gru')
    )

    # Then merge the result with MLP dataframe
    merged_df = pd.merge(
        merged_df,
        mlp_matching_rows[['y_true', 'y_pred', 'abs_error_mlp']],
        on='y_true',
    )

    merged_df = merged_df.rename(columns={'y_pred': 'y_pred_mlp',})

    print(merged_df.columns.tolist())
    
    # Calculate error for GRU model and average error
    print(merged_df.columns.tolist())
    print(merged_df.head())
    merged_df['avg_abs_error'] = (merged_df['abs_error_xgb'] + merged_df['abs_error_gru']) + merged_df['abs_error_mlp'] / 3

    # Sort by average absolute error
    merged_df = merged_df.sort_values('avg_abs_error')
    
    print(f"Merged dataset rows: {len(merged_df)}")
    
    # Re-sort the individual dataframes based on the merged sort order
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

if __name__ == "__main__":
    xgb_matches, gru_matches, mlp_matches, merged = find_matching_rows()
    print(f"\nAnalysis complete. Found {len(merged)} matching rows between datasets.")
    
    # Print additional statistics about error distribution
    print("\nError Statistics:")
    print(f"Average XGBoost abs error: {merged['abs_error_xgb'].mean():.4f}")
    print(f"Average GRU abs error: {merged['abs_error_gru'].mean():.4f}")
    print(f"XGB better prediction count: {(merged['abs_error_xgb'] < merged['abs_error_gru']).sum()}")
    print(f"GRU better prediction count: {(merged['abs_error_gru'] < merged['abs_error_xgb']).sum()}")
    print(f"Tie count: {(merged['abs_error_xgb'] == merged['abs_error_gru']).sum()}")