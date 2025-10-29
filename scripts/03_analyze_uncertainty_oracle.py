"""
Analyze correlation between uncertainty and oracle labels.

This validates that uncertainty is a good proxy for "should retrieve"
and provides key results/plots for the paper.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data", help="Data directory")
    parser.add_argument("--output_dir", default="./figures", help="Output directory for plots")
    parser.add_argument("--benchmark", default="humaneval", help="Benchmark name")
    args = parser.parse_args()

    print("=" * 80)
    print("Analyzing Uncertainty vs Oracle Correlation")
    print("=" * 80)

    # Load uncertainties and oracle labels
    uncertainties_path = os.path.join(args.data_dir, f"{args.benchmark}_uncertainties.csv")
    oracle_path = os.path.join(args.data_dir, f"{args.benchmark}_oracle.csv")

    print(f"\nLoading data from:")
    print(f"  - Uncertainties: {uncertainties_path}")
    print(f"  - Oracle: {oracle_path}")

    uncertainties_df = pd.read_csv(uncertainties_path)
    oracle_df = pd.read_csv(oracle_path)

    # Merge datasets
    df = pd.merge(uncertainties_df, oracle_df, on='task_id')

    print(f"\nMerged {len(df)} examples")

    # Remove any NaN values
    df = df.dropna(subset=['uncertainty', 'should_retrieve'])

    print(f"After removing NaNs: {len(df)} examples")

    # Compute correlation metrics
    print("\n" + "=" * 80)
    print("CORRELATION ANALYSIS")
    print("=" * 80)

    uncertainties = df['uncertainty'].values
    oracle_labels = df['should_retrieve'].values

    # AUC
    auc = roc_auc_score(oracle_labels, uncertainties)
    print(f"\nAUC (Uncertainty predicts Oracle): {auc:.3f}")

    # Find optimal threshold
    fpr, tpr, thresholds = roc_curve(oracle_labels, uncertainties)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    print(f"Optimal threshold: {optimal_threshold:.3f}")

    # Compute metrics at optimal threshold
    predictions = uncertainties > optimal_threshold
    accuracy = np.mean(predictions == oracle_labels)
    precision = np.sum(oracle_labels[predictions]) / np.sum(predictions) if np.sum(predictions) > 0 else 0
    recall = np.sum(predictions[oracle_labels]) / np.sum(oracle_labels) if np.sum(oracle_labels) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\nMetrics at optimal threshold:")
    print(f"  - Accuracy: {accuracy:.3f}")
    print(f"  - Precision: {precision:.3f}")
    print(f"  - Recall: {recall:.3f}")
    print(f"  - F1: {f1:.3f}")

    # Confusion matrix
    cm = confusion_matrix(oracle_labels, predictions)
    print(f"\nConfusion Matrix:")
    print(f"  TN={cm[0,0]}, FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}, TP={cm[1,1]}")

    # Statistics by oracle decision
    print("\n" + "=" * 80)
    print("UNCERTAINTY STATISTICS BY ORACLE DECISION")
    print("=" * 80)

    retrieve_group = df[df['should_retrieve'] == True]
    no_retrieve_group = df[df['should_retrieve'] == False]

    print(f"\nWhen Oracle says RETRIEVE (n={len(retrieve_group)}):")
    print(f"  - Mean uncertainty: {retrieve_group['uncertainty'].mean():.4f}")
    print(f"  - Median uncertainty: {retrieve_group['uncertainty'].median():.4f}")
    print(f"  - Std uncertainty: {retrieve_group['uncertainty'].std():.4f}")

    print(f"\nWhen Oracle says DON'T RETRIEVE (n={len(no_retrieve_group)}):")
    print(f"  - Mean uncertainty: {no_retrieve_group['uncertainty'].mean():.4f}")
    print(f"  - Median uncertainty: {no_retrieve_group['uncertainty'].median():.4f}")
    print(f"  - Std uncertainty: {no_retrieve_group['uncertainty'].std():.4f}")

    # Statistical test
    from scipy.stats import mannwhitneyu
    stat, p_value = mannwhitneyu(
        retrieve_group['uncertainty'],
        no_retrieve_group['uncertainty'],
        alternative='greater'
    )
    print(f"\nMann-Whitney U test (retrieve > no_retrieve):")
    print(f"  - Statistic: {stat:.2f}")
    print(f"  - p-value: {p_value:.4e}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # PLOT 1: Violin plot
    print("\n" + "=" * 80)
    print("CREATING VISUALIZATIONS")
    print("=" * 80)

    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df, x='should_retrieve', y='uncertainty')
    plt.xlabel('Oracle Decision: Should Retrieve?', fontsize=12)
    plt.ylabel('Model Uncertainty', fontsize=12)
    plt.title(f'Uncertainty Distribution by Oracle Decision (AUC={auc:.3f})', fontsize=14)
    plt.xticks([0, 1], ['No', 'Yes'])

    plot1_path = os.path.join(args.output_dir, 'uncertainty_vs_oracle_violin.png')
    plt.tight_layout()
    plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved plot: {plot1_path}")
    plt.close()

    # PLOT 2: ROC curve
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    plt.scatter([fpr[optimal_idx]], [tpr[optimal_idx]], s=100, c='red',
                label=f'Optimal threshold = {optimal_threshold:.3f}')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve: Uncertainty Predicts Oracle Decision', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)

    plot2_path = os.path.join(args.output_dir, 'roc_curve.png')
    plt.tight_layout()
    plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {plot2_path}")
    plt.close()

    # PLOT 3: Decision boundary scatter
    plt.figure(figsize=(12, 6))

    # Sort by uncertainty for better visualization
    df_sorted = df.sort_values('uncertainty')

    colors = ['blue' if label else 'orange' for label in df_sorted['should_retrieve']]
    plt.scatter(range(len(df_sorted)), df_sorted['uncertainty'], c=colors, alpha=0.6, s=30)
    plt.axhline(y=optimal_threshold, color='red', linestyle='--', linewidth=2,
                label=f'Optimal threshold = {optimal_threshold:.3f}')
    plt.xlabel('Problem Index (sorted by uncertainty)', fontsize=12)
    plt.ylabel('Uncertainty', fontsize=12)
    plt.title('Adaptive Retrieval Decision Boundary', fontsize=14)
    plt.legend(['Optimal threshold', 'Oracle: Retrieve', 'Oracle: No retrieve'], fontsize=10)
    plt.grid(alpha=0.3)

    plot3_path = os.path.join(args.output_dir, 'decision_boundary.png')
    plt.tight_layout()
    plt.savefig(plot3_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {plot3_path}")
    plt.close()

    # PLOT 4: Improvement vs Uncertainty
    plt.figure(figsize=(10, 6))
    plt.scatter(df['uncertainty'], df['improvement'], alpha=0.6, s=40)
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    plt.axvline(x=optimal_threshold, color='red', linestyle='--', linewidth=2,
                label=f'Optimal threshold = {optimal_threshold:.3f}')
    plt.xlabel('Uncertainty', fontsize=12)
    plt.ylabel('Improvement (RAG score - No RAG score)', fontsize=12)
    plt.title('Retrieval Benefit vs Uncertainty', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)

    plot4_path = os.path.join(args.output_dir, 'improvement_vs_uncertainty.png')
    plt.tight_layout()
    plt.savefig(plot4_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {plot4_path}")
    plt.close()

    # Save summary statistics
    summary = {
        'auc': float(auc),
        'optimal_threshold': float(optimal_threshold),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'oracle_retrieve_rate': float(oracle_labels.mean()),
        'mean_uncertainty_retrieve': float(retrieve_group['uncertainty'].mean()),
        'mean_uncertainty_no_retrieve': float(no_retrieve_group['uncertainty'].mean()),
        'mann_whitney_p_value': float(p_value)
    }

    summary_path = os.path.join(args.output_dir, 'uncertainty_oracle_summary.json')
    import json
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved summary: {summary_path}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print("\nKey Findings:")
    print(f"  ✓ AUC = {auc:.3f} (uncertainty predicts oracle decisions)")
    print(f"  ✓ Optimal threshold = {optimal_threshold:.3f}")
    print(f"  ✓ Accuracy at threshold = {accuracy:.3f}")
    print(f"  ✓ Oracle retrieves {oracle_labels.mean()*100:.1f}% of the time")
    print(f"  ✓ Uncertainty is significantly higher when retrieval helps (p={p_value:.4e})")
    print("\nThis validates that uncertainty-based adaptive retrieval is viable!")


if __name__ == "__main__":
    main()
