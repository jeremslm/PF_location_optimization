"""
Analyze Bayesian Optimization Acceleration for Paper

Generates key metrics and plots demonstrating how Bayesian optimization
accelerates convergence compared to multi-start L-BFGS:

1. Sample efficiency: Error vs function evaluations
2. Speedup metrics: Evaluations to reach target error
3. Hyperparameter analysis: Bayesian/refinement ratio impact
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
from pathlib import Path
import pandas as pd
import datetime 

# Set publication-quality plot style
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['lines.linewidth'] = 2

# Constants for projected time analysis
TARGET_TOLS = [0.10, 0.01, 0.001]  # 10%, 1%, 0.1% relative to best overall
TAUS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]  # seconds per evaluation


def load_results(json_path):
    """Load optimization results from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def calculate_target_metrics(bayesian_results, multistart_results, target_tols=None, taus=None):
    """
    Calculate projected time-to-target metrics for multiple tolerances and τ values.

    IMPORTANT: Does NOT use actual timing data. Computes hypothetical wall-clock time
    assuming τ seconds per objective evaluation (time = evals * τ).

    Parameters
    ----------
    bayesian_results : dict
        Results dict with 'best_cost' and 'convergence_history'
    multistart_results : dict
        Results dict with 'best_cost' and 'convergence_history'
    target_tols : list of float
        Relative tolerances (e.g., [0.10, 0.01, 0.001] for 10%, 1%, 0.1%)
    taus : list of float
        Hypothetical seconds per evaluation for projected time calculations

    Returns
    -------
    dict with structure:
        {
            'bayesian_final': float,
            'multistart_final': float,
            'best_overall': float,
            'by_tol': {
                eps: {
                    'target_cost': float,
                    'bayesian_evals': int or None,
                    'multistart_evals': int or None,
                    'eval_speedup': float or None,
                    'time_by_tau': {
                        tau: {
                            'bayesian_time': float or None,
                            'multistart_time': float or None,
                            'time_speedup': float or None
                        }, ...
                    }
                }, ...
            }
        }
    """
    if target_tols is None:
        target_tols = TARGET_TOLS
    if taus is None:
        taus = TAUS

    # Get final costs
    bay_final = float(bayesian_results['best_cost'])
    multi_final = float(multistart_results['best_cost'])
    best_overall = min(bay_final, multi_final)

    # Load and enforce monotonicity in convergence histories
    bay_conv = np.minimum.accumulate(np.array(bayesian_results['convergence_history'], dtype=float))
    multi_conv = np.minimum.accumulate(np.array(multistart_results['convergence_history'], dtype=float))

    # Calculate metrics for each tolerance
    by_tol = {}

    for eps in target_tols:
        target_cost = best_overall * (1.0 + eps)

        # Find evaluations to reach target for each method
        bay_idx = np.where(bay_conv <= target_cost)[0]
        multi_idx = np.where(multi_conv <= target_cost)[0]

        bay_evals = int(bay_idx[0] + 1) if bay_idx.size > 0 else None
        multi_evals = int(multi_idx[0] + 1) if multi_idx.size > 0 else None

        # Compute eval speedup
        if bay_evals is not None and multi_evals is not None:
            eval_speedup = multi_evals / bay_evals
        else:
            eval_speedup = None

        # Compute projected time-to-target for each τ
        time_by_tau = {}
        for tau in taus:
            bay_time = bay_evals * tau if bay_evals is not None else None
            multi_time = multi_evals * tau if multi_evals is not None else None

            if bay_time is not None and multi_time is not None:
                time_speedup = multi_time / bay_time
            else:
                time_speedup = None

            time_by_tau[tau] = {
                'bayesian_time': bay_time,
                'multistart_time': multi_time,
                'time_speedup': time_speedup
            }

        by_tol[eps] = {
            'target_cost': target_cost,
            'bayesian_evals': bay_evals,
            'multistart_evals': multi_evals,
            'eval_speedup': eval_speedup,
            'time_by_tau': time_by_tau
        }

    return {
        'bayesian_final': bay_final,
        'multistart_final': multi_final,
        'best_overall': best_overall,
        'by_tol': by_tol
    }


def create_target_table(all_configs_metrics, output_dir, taus=None, target_tols=None):
    """
    Create flat presentation-ready CSV table showing projected time-to-target.

    Parameters
    ----------
    all_configs_metrics : dict
        Mapping of config_name -> output of calculate_target_metrics()
    output_dir : str
        Directory to save CSV
    taus : list of float
        Tau values to include as columns
    target_tols : list of float
        Target tolerances

    Returns
    -------
    pd.DataFrame
    """
    if taus is None:
        taus = TAUS
    if target_tols is None:
        target_tols = TARGET_TOLS

    rows = []

    for config_name in sorted(all_configs_metrics.keys()):
        metrics = all_configs_metrics[config_name]

        bay_final = metrics['bayesian_final']
        multi_final = metrics['multistart_final']

        for eps in target_tols:
            tol_data = metrics['by_tol'][eps]

            # Format tolerance as percentage string
            tol_pct = f"{eps*100:.1f}%" if eps >= 0.01 else f"{eps*100:.2f}%"

            row = {
                'Configuration': config_name,
                'Tolerance': tol_pct,
                'TargetCost': f"{tol_data['target_cost']:.3e}",
                'BayesianFinalCost': f"{bay_final:.3e}",
                'MultiFinalCost': f"{multi_final:.3e}",
                'BayesianEvalsToTarget': tol_data['bayesian_evals'] if tol_data['bayesian_evals'] is not None else 'N/A',
                'MultiEvalsToTarget': tol_data['multistart_evals'] if tol_data['multistart_evals'] is not None else 'N/A',
                'EvalSpeedup_MultiOverBay': f"{tol_data['eval_speedup']:.2f}" if tol_data['eval_speedup'] is not None else 'N/A'
            }

            # Add time columns for each tau
            for tau in taus:
                tau_data = tol_data['time_by_tau'][tau]

                # Format times nicely
                def format_time(t):
                    if t is None:
                        return 'N/A'
                    if tau >= 1.0:
                        if t < 1e4:
                            return f"{int(t)}"
                        else:
                            return f"{t:.3e}"
                    else:
                        return f"{t:.3f}"

                tau_label = f"tau={tau}s" if tau >= 1.0 else f"tau={tau:.2f}s"

                row[f'BayesTime@{tau_label}'] = format_time(tau_data['bayesian_time'])
                row[f'MultiTime@{tau_label}'] = format_time(tau_data['multistart_time'])
                row[f'TimeSpeedup@{tau_label}'] = f"{tau_data['time_speedup']:.2f}" if tau_data['time_speedup'] is not None else 'N/A'

            rows.append(row)

    df = pd.DataFrame(rows)

    # Save CSV
    csv_path = os.path.join(output_dir, 'target_table.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    return df


def plot_convergence_comparison(results_dict, output_dir, title_suffix=""):
    """
    Plot convergence: cost vs evaluations for all configurations.

    Parameters
    ----------
    results_dict : dict
        {lambda_val: {'methods': {'Bayesian': ..., 'Multi-start L-BFGS': ...}}}
    """
    n_configs = len(results_dict)

    # Calculate grid size
    ncols = min(3, n_configs)
    nrows = int(np.ceil(n_configs / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows))
    if n_configs == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()

    # Sort lambda values properly (convert to float for sorting)
    configs = sorted(results_dict.keys(), key=lambda x: float(x))

    for idx, config in enumerate(configs):
        if idx >= len(axes):
            break

        ax = axes[idx]
        data = results_dict[config]

        # Plot Bayesian
        if 'Bayesian' in data['methods']:
            bay = data['methods']['Bayesian']
            conv = bay['convergence_history']
            ax.plot(range(1, len(conv)+1), conv, 'b-', label='Bayesian', linewidth=2)

            # Mark where Bayesian exploration ends
            if 'n_bayesian_evals' in bay:
                n_bay = bay['n_bayesian_evals']
                ax.axvline(n_bay, color='b', linestyle='--', alpha=0.3, linewidth=1.5,
                          label=f'Refinement starts')

        # Plot Multi-start
        if 'Multi-start L-BFGS' in data['methods']:
            multi = data['methods']['Multi-start L-BFGS']
            conv = multi['convergence_history']
            ax.plot(range(1, len(conv)+1), conv, 'r--', label='Multi-start L-BFGS', linewidth=2, alpha=0.8)

        ax.set_xlabel('Function Evaluations', fontsize=11)
        ax.set_ylabel('Cost', fontsize=11)
        ax.set_title(f'λ = {config}', fontsize=12, weight='bold')
        ax.set_yscale('log')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(len(configs), len(axes)):
        axes[idx].axis('off')

    fig.suptitle(f'Convergence Comparison{title_suffix}',
                 fontsize=15, weight='bold', y=0.995)
    plt.tight_layout()

    output_path = os.path.join(output_dir, 'convergence_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    return fig


def plot_sample_efficiency(results_dict, output_dir, max_evals=None):
    """
    Create single plot showing sample efficiency across all lambda values for this coil count.
    Key plot for demonstrating acceleration!
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Sort lambda values properly
    configs = sorted(results_dict.keys(), key=lambda x: float(x))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(configs)))

    for idx, lambda_val in enumerate(configs):
        data = results_dict[lambda_val]

        if 'Bayesian' in data['methods'] and 'Multi-start L-BFGS' in data['methods']:
            bay = data['methods']['Bayesian']
            multi = data['methods']['Multi-start L-BFGS']

            bay_conv = np.array(bay['convergence_history'])
            multi_conv = np.array(multi['convergence_history'])

            # Truncate to max_evals if specified
            if max_evals:
                bay_conv = bay_conv[:max_evals]
                multi_conv = multi_conv[:max_evals]

            # Plot both
            ax.plot(range(1, len(bay_conv)+1), bay_conv,
                   color=colors[idx], linestyle='-', linewidth=2.5,
                   label=f'λ={lambda_val} (Bayesian)')
            ax.plot(range(1, len(multi_conv)+1), multi_conv,
                   color=colors[idx], linestyle='--', linewidth=2, alpha=0.7,
                   label=f'λ={lambda_val} (Multi-start)')

    ax.set_xlabel('Function Evaluations', fontsize=13)
    ax.set_ylabel('Optimization Cost', fontsize=13)
    ax.set_title('Sample Efficiency: Bayesian vs Multi-start L-BFGS', fontsize=14, weight='bold')
    ax.set_yscale('log')
    ax.legend(fontsize=9, loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'sample_efficiency.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    return fig


def plot_hypothetical_time_tau(per_config_metrics, output_dir, eps=0.001, tau=1.0):
    """
    Create horizontal bar chart showing projected time-to-target at specified tolerance and τ.

    Parameters
    ----------
    per_config_metrics : dict
        Config name -> calculate_target_metrics output
    output_dir : str
        Output directory
    eps : float
        Tolerance level to visualize (default: 0.001 = 0.1%)
    tau : float
        Seconds per evaluation (default: 1.0)
    """
    configs = sorted(per_config_metrics.keys())

    bay_times = []
    multi_times = []
    labels = []

    for config in configs:
        metrics = per_config_metrics[config]
        tol_data = metrics['by_tol'][eps]
        tau_data = tol_data['time_by_tau'][tau]

        bay_time = tau_data['bayesian_time']
        multi_time = tau_data['multistart_time']

        bay_times.append(bay_time if bay_time is not None else 0)
        multi_times.append(multi_time if multi_time is not None else 0)
        labels.append(config)

    fig, ax = plt.subplots(1, 1, figsize=(10, max(6, len(configs)*0.5)))

    y_pos = np.arange(len(labels))
    width = 0.35

    # Plot bars
    bars1 = ax.barh(y_pos - width/2, bay_times, width, label='Bayesian', color='steelblue')
    bars2 = ax.barh(y_pos + width/2, multi_times, width, label='Multi-start', color='coral')

    # Annotate bars that didn't reach target
    for i, (bay_t, multi_t) in enumerate(zip(bay_times, multi_times)):
        if bay_t == 0:
            ax.text(0.02, i - width/2, 'NR', va='center', ha='left', fontsize=9, color='red', weight='bold')
        if multi_t == 0:
            ax.text(0.02, i + width/2, 'NR', va='center', ha='left', fontsize=9, color='red', weight='bold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel(f'Projected Time to Target (seconds)', fontsize=12)
    ax.set_title(f'Projected Time-to-Target\n(Tolerance={eps*100:.1f}%, τ={tau}s per evaluation)',
                fontsize=13, weight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    # Format filename
    tol_str = f"{eps*100:.1f}pct" if eps >= 0.01 else f"{eps*100:.2f}pct"
    tau_str = f"{tau}s" if tau >= 1.0 else f"{tau:.2f}s"
    output_path = os.path.join(output_dir, f'hyp_time_tau{tau_str}_tol{tol_str}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    return fig


def plot_speedup_summary(speedup_data, output_dir):
    """
    Bar chart showing speedup factors across configurations.
    """
    configs = list(speedup_data.keys())
    speedups = [speedup_data[c]['speedup_factor'] for c in configs]

    # Cap infinite speedups for visualization
    speedups_capped = [min(s, 50) if s != float('inf') else 50 for s in speedups]

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    bars = ax.barh(configs, speedups_capped, color='steelblue', edgecolor='black')

    # Add value labels
    for i, (bar, speedup) in enumerate(zip(bars, speedups)):
        if speedup == float('inf'):
            label = '∞'
        else:
            label = f'{speedup:.1f}×'
        ax.text(speedups_capped[i], i, f'  {label}',
               va='center', ha='left', fontsize=11, weight='bold')

    ax.set_xlabel('Speedup Factor (×)', fontsize=13)
    ax.set_title('Bayesian Speedup: Function Evaluations to Reach Target',
                fontsize=14, weight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'speedup_summary.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    return fig


def plot_early_convergence(results_dict, output_dir, n_evals=500):
    """
    Focus on early convergence to show Bayesian exploration advantage.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Sort lambda values properly
    configs = sorted(results_dict.keys(), key=lambda x: float(x))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(configs)))

    for idx, lambda_val in enumerate(configs):
        data = results_dict[lambda_val]

        if 'Bayesian' in data['methods'] and 'Multi-start L-BFGS' in data['methods']:
            bay = data['methods']['Bayesian']
            multi = data['methods']['Multi-start L-BFGS']

            bay_conv = np.array(bay['convergence_history'][:n_evals])
            multi_conv = np.array(multi['convergence_history'][:n_evals])

            ax.plot(range(1, len(bay_conv)+1), bay_conv,
                   color=colors[idx], linestyle='-', linewidth=2.5, marker='o',
                   markersize=3, markevery=50,
                   label=f'λ={lambda_val} (Bayesian)')
            ax.plot(range(1, len(multi_conv)+1), multi_conv,
                   color=colors[idx], linestyle='--', linewidth=2, alpha=0.7,
                   marker='s', markersize=3, markevery=50,
                   label=f'λ={lambda_val} (Multi-start)')

    ax.set_xlabel('Function Evaluations', fontsize=13)
    ax.set_ylabel('Best Cost Found', fontsize=13)
    ax.set_title(f'Early Convergence (First {n_evals} Evaluations)', fontsize=14, weight='bold')
    ax.set_yscale('log')
    ax.legend(fontsize=9, loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'early_convergence.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    return fig


def create_summary_table(speedup_data, output_dir):
    """
    Create summary table with key metrics.
    """
    rows = []

    for config, metrics in speedup_data.items():
        row = {
            'Configuration': config,
            'Bayesian Final Cost': f"{metrics['bayesian_final']:.3e}",
            'Multi-start Final Cost': f"{metrics['multistart_final']:.3e}",
            'Cost Ratio': f"{metrics['final_cost_ratio']:.2f}",
            'Bayesian Evals': metrics['bayesian_evals_to_target'],
            'Multi-start Evals': metrics['multistart_evals_to_target'] if metrics['multistart_evals_to_target'] else 'N/A',
            'Speedup': f"{metrics['speedup_factor']:.1f}×" if metrics['speedup_factor'] != float('inf') else '∞',
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Save to CSV
    csv_path = os.path.join(output_dir, 'speedup_summary.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Print to console
    print("\n" + "="*80)
    print("BAYESIAN ACCELERATION SUMMARY")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80 + "\n")

    return df


def analyze_bayesian_ratio(ratio_results_dict, output_dir):
    """
    Analyze impact of Bayesian/refinement time ratio.

    Parameters
    ----------
    ratio_results_dict : dict
        {ratio_value: results_data}
    """
    if not ratio_results_dict or len(ratio_results_dict) < 2:
        print("Not enough ratio data for analysis")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ratios = sorted(ratio_results_dict.keys())
    final_costs = []
    bayesian_evals = []
    total_evals = []

    for ratio in ratios:
        data = ratio_results_dict[ratio]
        if 'Bayesian' in data['methods']:
            bay = data['methods']['Bayesian']
            final_costs.append(bay['best_cost'])
            bayesian_evals.append(bay.get('n_bayesian_evals', 0))
            total_evals.append(bay['n_evals'])

    # Plot 1: Final cost vs ratio
    ax1.plot(ratios, final_costs, 'o-', markersize=8, linewidth=2, color='steelblue')
    ax1.set_xlabel('Bayesian Exploration Ratio', fontsize=12)
    ax1.set_ylabel('Final Cost', fontsize=12)
    ax1.set_title('Final Cost vs Bayesian/Refinement Ratio', fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Plot 2: Evaluation breakdown
    refinement_evals = np.array(total_evals) - np.array(bayesian_evals)

    x = np.arange(len(ratios))
    width = 0.35

    ax2.bar(x, bayesian_evals, width, label='Bayesian Exploration', color='steelblue')
    ax2.bar(x, refinement_evals, width, bottom=bayesian_evals,
           label='L-BFGS Refinement', color='coral')

    ax2.set_xlabel('Bayesian Exploration Ratio', fontsize=12)
    ax2.set_ylabel('Function Evaluations', fontsize=12)
    ax2.set_title('Evaluation Budget Allocation', fontsize=13)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{r:.1f}' for r in ratios])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'ratio_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    return fig


def main():
    """Main analysis pipeline."""

    # Configuration
    base_dir = Path('examples/comparisons/closed_boundary_DIIID')
    date = datetime.datetime.now().strftime("%m_%d_%H:%M")
    output_dir = f'bayesian_analysis_output/res_{date}'
    os.makedirs(output_dir, exist_ok=True)

    print("="*80)
    print("BAYESIAN OPTIMIZATION ACCELERATION ANALYSIS")
    print("="*80)

    # 1. Load all comparison results from test_general ONLY

    # test_folder = 'test_general'
    test_folder = 'for_presentation'
    test_dir = base_dir / test_folder

    if not test_dir.exists():
        print(f"Error: {test_dir} does not exist!")
        return

    # Find all results.json files in test_general
    result_files = list(test_dir.glob('**/results.json'))
    print(f"Found {len(result_files)} result files")

    # Organize by number of coils
    results_by_coils = {}

    for result_file in result_files:
        # Get configuration (lambda, coils)
        config_folder = result_file.parent.name

        # Parse config: lambda:1e-05,coils:3
        try:
            parts = config_folder.split(',')
            lambda_part = parts[0]  # lambda:1e-05
            coils_part = parts[1]    # coils:3

            lambda_val = lambda_part.split(':')[1]
            n_coils = int(coils_part.split(':')[1])

            # Load data
            data = load_results(result_file)

            # Check if has both methods
            if 'Bayesian' in data['methods'] and 'Multi-start L-BFGS' in data['methods']:
                if n_coils not in results_by_coils:
                    results_by_coils[n_coils] = {}
                results_by_coils[n_coils][lambda_val] = data
        except:
            print(f"  Skipping {config_folder} - could not parse")
            continue

    print(f"\nOrganized by number of coils:")
    for n_coils in sorted(results_by_coils.keys()):
        print(f"  {n_coils} coils: {len(results_by_coils[n_coils])} lambda values")

    # 2. Analyze each coil count separately
    print("\n2. Generating analysis plots grouped by coil count...")

    all_metrics = {}  # For overall summary

    for n_coils in sorted(results_by_coils.keys()):
        results_dict = results_by_coils[n_coils]

        if len(results_dict) == 0:
            continue

        print(f"\n  Analyzing: {n_coils} coils ({len(results_dict)} lambda values)")

        # Create subdirectory for this coil count
        coil_output_dir = os.path.join(output_dir, f'{n_coils}_coils')
        os.makedirs(coil_output_dir, exist_ok=True)

        # Calculate target metrics for each lambda value
        per_config_metrics = {}
        speedup_data = {}  # For backward compatibility with old plots

        for lambda_val, data in results_dict.items():
            bay = data['methods']['Bayesian']
            multi = data['methods']['Multi-start L-BFGS']

            config_name = f'λ={lambda_val}'

            # New: Calculate projected time-to-target metrics
            metrics = calculate_target_metrics(bay, multi)
            per_config_metrics[config_name] = metrics

            # Store for overall summary
            overall_config_name = f'{n_coils} coils, λ={lambda_val}'
            all_metrics[overall_config_name] = metrics

            # Old: Keep backward compatibility for existing plots
            # Use 10% tolerance for speedup summary
            tol_10pct = metrics['by_tol'][0.10]
            speedup_data[config_name] = {
                'bayesian_final': metrics['bayesian_final'],
                'multistart_final': metrics['multistart_final'],
                'target_cost': tol_10pct['target_cost'],
                'bayesian_evals_to_target': tol_10pct['bayesian_evals'],
                'multistart_evals_to_target': tol_10pct['multistart_evals'],
                'speedup_factor': tol_10pct['eval_speedup'] if tol_10pct['eval_speedup'] is not None else float('inf'),
                'final_cost_ratio': metrics['multistart_final'] / metrics['bayesian_final']
            }

        # Generate NEW projected time table
        create_target_table(per_config_metrics, coil_output_dir)

        # Generate optional projected time plot
        try:
            plot_hypothetical_time_tau(per_config_metrics, coil_output_dir, eps=0.001, tau=1.0)
        except Exception as e:
            print(f"  Warning: Could not generate hypothetical time plot: {e}")

        # Generate OLD plots (each plot shows different lambda values for this coil count)
        plot_convergence_comparison(results_dict, coil_output_dir, f" ({n_coils} coils)")
        plot_sample_efficiency(results_dict, coil_output_dir, max_evals=10000)
        plot_early_convergence(results_dict, coil_output_dir, n_evals=500)
        plot_speedup_summary(speedup_data, coil_output_dir)

        # Create OLD summary table (backward compatibility)
        create_summary_table(speedup_data, coil_output_dir)

    # 3. Create overall summary
    print("\n3. Creating overall summary...")

    # NEW: Overall projected time table
    create_target_table(all_metrics, output_dir)

    # OLD: Backward compatibility summary table
    all_speedup_data = {}
    for config_name, metrics in all_metrics.items():
        tol_10pct = metrics['by_tol'][0.10]
        all_speedup_data[config_name] = {
            'bayesian_final': metrics['bayesian_final'],
            'multistart_final': metrics['multistart_final'],
            'target_cost': tol_10pct['target_cost'],
            'bayesian_evals_to_target': tol_10pct['bayesian_evals'],
            'multistart_evals_to_target': tol_10pct['multistart_evals'],
            'speedup_factor': tol_10pct['eval_speedup'] if tol_10pct['eval_speedup'] is not None else float('inf'),
            'final_cost_ratio': metrics['multistart_final'] / metrics['bayesian_final']
        }

    create_summary_table(all_speedup_data, output_dir)

    print(f"\n✓ All analysis complete! Results saved to: {output_dir}/")
    print(f"✓ New projected time-to-target tables: target_table.csv in each folder")


if __name__ == "__main__":
    main()
