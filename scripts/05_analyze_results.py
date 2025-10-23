"""
Script 5: Analyze results and generate plots for paper.
"""

import sys
sys.path.append("..")

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from scipy import stats


def load_results(results_path: str) -> dict:
    """Load results from JSON file."""
    with open(results_path, 'r') as f:
        return json.load(f)


def create_results_table(results: dict) -> pd.DataFrame:
    """Create results table for paper."""
    data = []

    for exp_name, exp_results in results.items():
        row = {"Experiment": exp_name}

        for benchmark, metrics in exp_results.items():
            for metric_name, value in metrics.items():
                col_name = f"{benchmark}_{metric_name}"
                row[col_name] = value

        data.append(row)

    df = pd.DataFrame(data)
    return df


def calculate_improvements(df: pd.DataFrame) -> dict:
    """Calculate improvements of DoRA+RAG over baselines."""
    improvements = {}

    # Find DoRA+RAG row
    dora_rag = df[df["Experiment"] == "dora_rag"]
    baseline = df[df["Experiment"] == "baseline"]
    dora_only = df[df["Experiment"] == "dora_only"]
    rag_only = df[df["Experiment"] == "rag_only"]

    if len(dora_rag) == 0:
        print("Warning: dora_rag results not found")
        return improvements

    # Calculate relative improvements
    for col in df.columns:
        if col == "Experiment":
            continue

        if len(baseline) > 0 and col in baseline.columns:
            baseline_val = baseline[col].values[0]
            dora_rag_val = dora_rag[col].values[0]

            if baseline_val > 0:
                improvement = ((dora_rag_val - baseline_val) / baseline_val) * 100
                improvements[f"{col}_vs_baseline"] = improvement

        if len(dora_only) > 0 and col in dora_only.columns:
            dora_val = dora_only[col].values[0]
            dora_rag_val = dora_rag[col].values[0]

            if dora_val > 0:
                improvement = ((dora_rag_val - dora_val) / dora_val) * 100
                improvements[f"{col}_vs_dora_only"] = improvement

    return improvements


def plot_benchmark_comparison(df: pd.DataFrame, output_dir: Path):
    """Create bar plot comparing all experiments across benchmarks."""
    # Extract pass@1 scores for main benchmarks
    benchmarks = ["humaneval", "mbpp", "ds1000", "unseen_api"]
    plot_data = []

    for _, row in df.iterrows():
        for benchmark in benchmarks:
            col_name = f"{benchmark}_pass@1"
            if col_name in df.columns:
                plot_data.append({
                    "Experiment": row["Experiment"],
                    "Benchmark": benchmark.upper().replace("_", "-"),
                    "Pass@1": row[col_name]
                })

    plot_df = pd.DataFrame(plot_data)

    # Create plot
    plt.figure(figsize=(14, 6))
    sns.set_style("whitegrid")

    ax = sns.barplot(
        data=plot_df,
        x="Benchmark",
        y="Pass@1",
        hue="Experiment",
        palette="Set2"
    )

    plt.title("Pass@1 Comparison Across Benchmarks", fontsize=16, fontweight='bold')
    plt.ylabel("Pass@1 Score", fontsize=12)
    plt.xlabel("Benchmark", fontsize=12)
    plt.legend(title="Experiment", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    output_path = output_dir / "benchmark_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    plt.close()


def plot_ablation_study(df: pd.DataFrame, output_dir: Path):
    """Create ablation study visualization."""
    # Focus on key ablations
    ablation_order = ["baseline", "rag_only", "dora_only", "dora_rag"]
    df_ablation = df[df["Experiment"].isin(ablation_order)]

    # Reorder
    df_ablation["Experiment"] = pd.Categorical(
        df_ablation["Experiment"],
        categories=ablation_order,
        ordered=True
    )
    df_ablation = df_ablation.sort_values("Experiment")

    # Extract scores
    benchmarks = ["humaneval", "mbpp", "ds1000", "unseen_api"]
    plot_data = []

    for _, row in df_ablation.iterrows():
        for benchmark in benchmarks:
            col_name = f"{benchmark}_pass@1"
            if col_name in df_ablation.columns:
                plot_data.append({
                    "Experiment": row["Experiment"],
                    "Benchmark": benchmark.upper().replace("_", "-"),
                    "Pass@1": row[col_name]
                })

    plot_df = pd.DataFrame(plot_data)

    # Create plot
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle("Ablation Study: Effect of DoRA and RAG", fontsize=16, fontweight='bold')

    for i, benchmark in enumerate(["HumanEval", "MBPP", "DS-1000", "UNSEEN-API"]):
        benchmark_data = plot_df[plot_df["Benchmark"] == benchmark]

        ax = axes[i]
        sns.barplot(
            data=benchmark_data,
            x="Experiment",
            y="Pass@1",
            ax=ax,
            palette="viridis"
        )

        ax.set_title(benchmark, fontsize=14, fontweight='bold')
        ax.set_ylabel("Pass@1" if i == 0 else "")
        ax.set_xlabel("")
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()

    output_path = output_dir / "ablation_study.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    plt.close()


def generate_latex_table(df: pd.DataFrame, output_dir: Path):
    """Generate LaTeX table for paper."""
    # Select key metrics
    cols_to_include = ["Experiment"]
    for benchmark in ["humaneval", "mbpp", "ds1000", "unseen_api"]:
        col = f"{benchmark}_pass@1"
        if col in df.columns:
            cols_to_include.append(col)

    table_df = df[cols_to_include].copy()

    # Rename columns for LaTeX
    table_df.columns = [
        "Experiment",
        "HumanEval",
        "MBPP",
        "DS-1000",
        "Unseen-API"
    ]

    # Format numbers
    for col in table_df.columns[1:]:
        table_df[col] = table_df[col].apply(lambda x: f"{x:.4f}")

    # Convert to LaTeX
    latex_str = table_df.to_latex(index=False, escape=False)

    # Save
    output_path = output_dir / "results_table.tex"
    with open(output_path, 'w') as f:
        f.write(latex_str)

    print(f"Saved LaTeX table to {output_path}")


def main():
    """Main analysis function."""
    print("="*80)
    print("STEP 5: Analyzing Results")
    print("="*80)

    # Load results
    results_path = Path("outputs/ablation_results/all_results.json")
    if not results_path.exists():
        print(f"Error: Results not found at {results_path}")
        print("Please run 04_run_ablations.py first")
        return

    print(f"Loading results from {results_path}...")
    results = load_results(results_path)

    # Create results table
    print("\nCreating results table...")
    df = create_results_table(results)

    # Print table
    print("\n--- Results Table ---")
    print(df.to_string())

    # Calculate improvements
    print("\n--- Calculating Improvements ---")
    improvements = calculate_improvements(df)

    print("\nDoRA+RAG improvements:")
    for metric, improvement in improvements.items():
        print(f"  {metric}: {improvement:+.2f}%")

    # Create output directory for plots
    output_dir = Path("outputs/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save results table
    df.to_csv(output_dir / "results_table.csv", index=False)
    print(f"\nSaved results table to {output_dir / 'results_table.csv'}")

    # Generate plots
    print("\n--- Generating Plots ---")
    plot_benchmark_comparison(df, output_dir)
    plot_ablation_study(df, output_dir)

    # Generate LaTeX table
    print("\n--- Generating LaTeX Table ---")
    generate_latex_table(df, output_dir)

    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()
