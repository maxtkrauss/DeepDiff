import pandas as pd
import numpy as np

def find_best_model(csv_path=r"Z:\HSP\loss_optimization\hyperparam_sweep_metrics.csv"):
    """
    Find the single best model using equal weighting for all metrics
    """
    # Load the results
    df = pd.read_csv(csv_path)
    
    print("="*60)
    print("FINDING BEST OVERALL MODEL")
    print("="*60)
    print(f"Total runs analyzed: {len(df)}")
    print()
    
    # Define which metrics are "lower is better" vs "higher is better"
    lower_is_better = ['avg_mse', 'avg_mae', 'avg_rase', 'avg_RSE']
    higher_is_better = ['avg_ssim_3d', 'avg_ssim_2d', 'avg_fidelity']
    
    # Create normalized scores (equal weight for all metrics)
    df_norm = df.copy()
    
    # Normalize all metrics to [0,1] where 1 is best
    for metric in lower_is_better:
        if metric in df.columns:
            max_val = df[metric].max()
            min_val = df[metric].min()
            if max_val != min_val:
                df_norm[f'{metric}_norm'] = (max_val - df[metric]) / (max_val - min_val)
            else:
                df_norm[f'{metric}_norm'] = 1.0
    
    for metric in higher_is_better:
        if metric in df.columns:
            max_val = df[metric].max()
            min_val = df[metric].min()
            if max_val != min_val:
                df_norm[f'{metric}_norm'] = (df[metric] - min_val) / (max_val - min_val)
            else:
                df_norm[f'{metric}_norm'] = 1.0
    
    # Calculate simple average of all normalized metrics (equal weights)
    norm_cols = [col for col in df_norm.columns if col.endswith('_norm')]
    df_norm['overall_score'] = df_norm[norm_cols].mean(axis=1)
    
    # Find the best model
    best_idx = df_norm['overall_score'].idxmax()
    best_model = df.loc[best_idx, 'run_name']
    best_score = df_norm.loc[best_idx, 'overall_score']
    
    print(f"🏆 BEST MODEL: {best_model}")
    print(f"📊 Overall Score: {best_score:.4f}")
    print()
    print("📈 Metrics for Best Model:")
    print("-" * 30)
    
    for col in df.columns[1:]:  # Skip run_name
        if col in df.columns:
            value = df.loc[best_idx, col]
            print(f"{col:15s}: {value:.6f}")
    
    print("="*60)
    return best_model

if __name__ == "__main__":
    best_model = find_best_model()
    print(f"\n✅ Use this model: {best_model}")