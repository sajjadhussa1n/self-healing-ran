# src/agents/evaluation.py
"""
Evaluation pipeline for comparing trained RL agents against
heuristic compensation strategies.

evaluate_heuristic() and evaluate_rl_agent() are the exact
functions validated in the project notebooks and used to
produce the paper's Table III results.

Plotting functions (_plot_coverage_solve_rate,
_plot_energy_steps, _plot_action_distribution,
_plot_bs_heatmap) are ported verbatim from the validated
notebook plotting cells.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from src.environment.gym_env import SelfHealingNetworkEnv

IEEE_DPI = 600


# ============================================================
# Heuristic evaluator (verbatim from validated notebook code)
# ============================================================
def evaluate_heuristic(strategy_name, action_id,
                       config, n_episodes=500,
                       seed_offset=9999):
    results = {
        'coverage_pct'       : [],
        'mean_sinr_db'       : [],
        'ues_in_outage'      : [],
        'steps_to_solve'     : [],
        'solved'             : [],
        'failed_bs_id'       : [],
        'mean_throughput'    : [],
        'total_power_boost'  : [],
        'cumulative_energy'  : [],
        'avg_boost_per_step' : [],
        'power_efficiency'   : [],
        'tilt_only_steps'    : [],
        'zero_boost_fraction': [],
        'n_steps_taken'      : [],
    }

    env = SelfHealingNetworkEnv(
        config         = config,
        max_steps      = 10,
        n_normal_steps = 5,
        ue_step_size_m = 10.0,
        failed_bs_id   = None,
        use_curriculum = False,
        suppress_output= True,
        verbose        = False)

    for ep in range(n_episodes):
        env._episode_count = seed_offset + ep
        obs, info          = env.reset()

        baseline_outage_ues = env._baseline_outage_ues

        cumulative_energy = 0.0
        tilt_only_steps   = 0
        zero_boost_steps  = 0
        n_steps           = 0

        obs, reward, term, trunc, info = env.step(action_id)
        n_steps += 1

        step_boost = sum(
            max(bs.tx_power_dbm - bs.nominal_power_dbm, 0.0)
            for bs in env.network.base_stations
            if bs.is_active)
        cumulative_energy += step_boost
        if step_boost < 0.01:
            zero_boost_steps += 1
            if action_id in {5, 6}:
                tilt_only_steps += 1

        done = term or trunc

        while not done:
            obs, reward, term, trunc, info = env.step(0)
            n_steps += 1

            step_boost = sum(
                max(bs.tx_power_dbm - bs.nominal_power_dbm, 0.0)
                for bs in env.network.base_stations
                if bs.is_active)
            cumulative_energy += step_boost
            if step_boost < 0.01:
                zero_boost_steps += 1
            done = term or trunc

        stats  = info['stats_after']
        solved = (stats['ues_in_outage'] == 0)

        n_rescued = max(
            baseline_outage_ues - stats['ues_in_outage'], 0)

        if cumulative_energy > 0.01:
            pwr_efficiency = n_rescued / cumulative_energy
        else:
            pwr_efficiency = (float(n_rescued) * 10.0
                              if n_rescued > 0 else 0.0)

        avg_boost = (cumulative_energy / n_steps
                    if n_steps > 0 else 0.0)
        zero_frac = (zero_boost_steps / n_steps
                    if n_steps > 0 else 0.0)

        final_boost = sum(
            max(bs.tx_power_dbm - bs.nominal_power_dbm, 0)
            for bs in env.network.base_stations
            if bs.is_active)

        results['coverage_pct'].append(stats['coverage_pct'])
        results['mean_sinr_db'].append(stats['mean_sinr_db'])
        results['ues_in_outage'].append(stats['ues_in_outage'])
        results['solved'].append(int(solved))
        results['failed_bs_id'].append(info['failed_bs_id'])
        results['mean_throughput'].append(
            stats.get('mean_throughput', 0))
        results['steps_to_solve'].append(1 if solved else 10)
        results['total_power_boost'].append(final_boost)
        results['cumulative_energy'].append(cumulative_energy)
        results['avg_boost_per_step'].append(avg_boost)
        results['power_efficiency'].append(pwr_efficiency)
        results['tilt_only_steps'].append(tilt_only_steps)
        results['zero_boost_fraction'].append(zero_frac)
        results['n_steps_taken'].append(n_steps)

    return {k: np.array(v) for k, v in results.items()}


# ============================================================
# RL agent evaluator (verbatim from validated notebook code)
# ============================================================
def evaluate_rl_agent(model, model_name, config,
                      n_episodes=500, seed_offset=99999):
    """
    Evaluate a trained RL agent over n_episodes.
    Agent selects actions greedily (deterministic).
    """
    results = {
        'coverage_pct'     : [],
        'mean_sinr_db'     : [],
        'ues_in_outage'    : [],
        'steps_to_solve'   : [],
        'solved'           : [],
        'failed_bs_id'     : [],
        'mean_throughput'  : [],
        'total_power_boost': [],
        'actions_taken'    : [],
        'cumulative_energy'  : [],
        'avg_boost_per_step' : [],
        'power_efficiency'   : [],
        'tilt_only_steps'    : [],
        'zero_boost_fraction': [],
        'n_steps_taken'      : [],
    }

    env = SelfHealingNetworkEnv(
        config         = config,
        max_steps      = 10,
        n_normal_steps = 5,
        ue_step_size_m = 10.0,
        failed_bs_id   = None,
        use_curriculum = False,
        verbose        = False)

    for ep in range(n_episodes):
        env._episode_count = seed_offset + ep
        obs, info          = env.reset()
        baseline_outage_ues = env._baseline_outage_ues

        done        = False
        ep_actions  = []
        step_solved = None
        cumulative_energy = 0.0
        tilt_only_steps  = 0
        zero_boost_steps = 0
        n_steps          = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(int(action))
            ep_actions.append(int(action))
            n_steps += 1
            done = term or trunc

            if term and step_solved is None:
                step_solved = env._step_count

            step_boost = sum(
                max(bs.tx_power_dbm - bs.nominal_power_dbm, 0.0)
                for bs in env.network.base_stations
                if bs.is_active)
            cumulative_energy += step_boost

            if step_boost < 0.01:
                zero_boost_steps += 1
                if int(action) in {5, 6}:
                    tilt_only_steps += 1

        stats  = info['stats_after']
        solved = (stats['ues_in_outage'] == 0)

        n_rescued = max(
            baseline_outage_ues - stats['ues_in_outage'], 0)

        if cumulative_energy > 0.01:
            pwr_efficiency = n_rescued / cumulative_energy
        else:
            pwr_efficiency = (float(n_rescued) * 10.0
                              if n_rescued > 0 else 0.0)

        avg_boost = (cumulative_energy / n_steps
                    if n_steps > 0 else 0.0)
        zero_boost_frac = (zero_boost_steps / n_steps
                           if n_steps > 0 else 0.0)

        final_boost = sum(
            max(bs.tx_power_dbm - bs.nominal_power_dbm, 0)
            for bs in env.network.base_stations
            if bs.is_active)

        results['coverage_pct'].append(stats['coverage_pct'])
        results['mean_sinr_db'].append(stats['mean_sinr_db'])
        results['ues_in_outage'].append(stats['ues_in_outage'])
        results['solved'].append(int(solved))
        results['failed_bs_id'].append(info['failed_bs_id'])
        results['mean_throughput'].append(
            stats.get('mean_throughput', 0))
        results['steps_to_solve'].append(
            step_solved if step_solved else env._step_count)
        results['actions_taken'].append(ep_actions)
        results['total_power_boost'].append(final_boost)
        results['cumulative_energy'].append(cumulative_energy)
        results['avg_boost_per_step'].append(avg_boost)
        results['power_efficiency'].append(pwr_efficiency)
        results['tilt_only_steps'].append(tilt_only_steps)
        results['zero_boost_fraction'].append(zero_boost_frac)
        results['n_steps_taken'].append(n_steps)

    return {k: (np.array(v) if k != 'actions_taken' else v)
            for k, v in results.items()}


# ============================================================
# Plot 1: Coverage + Solve Rate (dual axis)
# ============================================================
def plot_coverage_comparison(all_results, save_dir="docs/figures"):
    plt.rcParams['font.family'] = 'Serif'
    plt.rcParams['font.size'] = 12

    names = list(all_results.keys())
    short_labels = [
        'No Action', 'S1: Fixed', 'S2: Proportional',
        'S3: Best Nbr', 'S4: Simultaneous', 'S5: Tilt Only',
        'S6: Joint', 'PPO', 'DQN'
    ]

    means = [np.mean(r['coverage_pct']) for r in all_results.values()]
    p5s   = [np.percentile(r['coverage_pct'], 5)
            for r in all_results.values()]
    p95s  = [np.percentile(r['coverage_pct'], 95)
            for r in all_results.values()]
    err_lo = [m - p for m, p in zip(means, p5s)]
    err_hi = [p - m for m, p in zip(means, p95s)]

    # Extract solve rates from results (not hardcoded)
    solve_rates = [np.mean(r['solved']) * 100
                  for r in all_results.values()]

    # Swap last two elements (PPO/DQN ordering)
    means[-2], means[-1] = means[-1], means[-2]
    p5s[-2], p5s[-1] = p5s[-1], p5s[-2]
    p95s[-2], p95s[-1] = p95s[-1], p95s[-2]
    err_lo[-2], err_lo[-1] = err_lo[-1], err_lo[-2]
    err_hi[-2], err_hi[-1] = err_hi[-1], err_hi[-2]
    solve_rates[-2], solve_rates[-1] = solve_rates[-1], solve_rates[-2]

    colors = []
    for n in names:
        if 'DQN' in n or 'PPO' in n:
            colors.append('#e74c3c')
        elif n == 'No Healing':
            colors.append('#95a5a6')
        else:
            colors.append('#3498db')

    fig, ax = plt.subplots(figsize=(4.50, 3.0))
    x = np.arange(len(names))

    bars = ax.bar(
        x, means, color=colors, edgecolor='black',
        linewidth=0.5, width=0.62, yerr=[err_lo, err_hi],
        capsize=2.5,
        error_kw={'elinewidth': 0.7, 'ecolor': 'black'})

    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=20, ha='right', fontsize=7)
    ax.set_ylabel("Mean Coverage (%)", fontsize=7)
    ax.set_title(
        "Solve Rate and Coverage Comparison: RL Agents vs "
        "Heuristic Strategies\n"
        "(error bars = Coverage 5th\u201395th percentile)",
        fontsize=7)
    ax.tick_params(axis='both', labelsize=7)
    ax.set_ylim(85, 101)
    ax.axhline(y=100, color='green', linestyle='--', lw=1,
              label='100% target')
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.set_facecolor('white')

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + 0.12, bar.get_height() + 0.12,
               f'{mean:.1f}%', ha='center', va='bottom', fontsize=7)

    ax2 = ax.twinx()
    ax2.plot(x, solve_rates, color='black', marker='o',
            markersize=4.5, linewidth=1.2, linestyle='--',
            label='Solve Rate')
    ax2.set_ylabel("Solve Rate (%)", fontsize=7)
    ax2.tick_params(axis='y', labelsize=7)
    ax2.set_ylim(0, 80)

    for xi, sr in zip(x, solve_rates):
        ax2.text(xi + 0.12, sr + 0.7, f'{sr:.1f}%', ha='center',
                va='bottom', fontsize=7, color='black')

    legend_els = [
        mpatches.Patch(facecolor='#e74c3c', label='RL Agent'),
        mpatches.Patch(facecolor='#3498db', label='Heuristic'),
        mpatches.Patch(facecolor='#95a5a6', label='No Healing'),
        Line2D([0], [0], color='black', marker='o', markersize=4.5,
              linestyle='--', linewidth=1.2, label='Solve Rate')
    ]
    ax.legend(handles=legend_els, fontsize=7, loc='upper left')

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "coverage_comparison.png")
    plt.savefig(save_path, dpi=IEEE_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ============================================================
# Plot 2: Cumulative Energy + Steps (dual axis)
# ============================================================
def plot_cumulative_energy(all_results, save_dir="docs/figures"):
    strategy_order = [
        'No Healing', 'S1: Fixed', 'S2: Proportional',
        'S3: Best Nbr', 'S4: Simultaneous', 'S5: Tilt Only',
        'S6: Joint', 'PPO Agent', 'DQN Agent'
    ]
    short_labels = [
        'No Action', 'S1: Fixed', 'S2: Proportional',
        'S3: Best Nbr', 'S4: Simultaneous', 'S5: Tilt Only',
        'S6: Joint', 'PPO', 'DQN'
    ]

    present = [s for s in strategy_order if s in all_results]
    labels = [short_labels[strategy_order.index(s)] for s in present]
    x = np.arange(len(present))

    bar_colors = []
    for s in present:
        if s == 'No Healing':
            bar_colors.append('#636363')
        elif s in ('DQN Agent', 'PPO Agent'):
            bar_colors.append('#e74c3c')
        else:
            bar_colors.append('#3498db')

    mean_energy = np.array([
        np.mean(all_results[s]['cumulative_energy']) for s in present])
    std_energy = np.array([
        np.std(all_results[s]['cumulative_energy']) for s in present])

    # Mean steps: derive from actions_taken for RL agents,
    # n_steps_taken/steps_to_solve for heuristics
    mean_steps, std_steps = [], []
    for s in present:
        if ('DQN' in s) or ('PPO' in s):
            step_counts = np.array([
                len(actions)
                for actions in all_results[s]['actions_taken']])
            mean_steps.append(np.mean(step_counts))
            std_steps.append(np.std(step_counts))
        else:
            steps = all_results[s].get(
                'n_steps_taken',
                all_results[s].get('steps_to_solve', [10]))
            mean_steps.append(np.mean(steps))
            std_steps.append(np.std(steps))

    mean_steps = np.array(mean_steps)
    std_steps  = np.array(std_steps)

    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    ax_steps = ax.twinx()

    bars = ax.bar(
        x, mean_energy, yerr=std_energy, color=bar_colors,
        edgecolor='black', linewidth=0.5, width=0.62, capsize=2.5,
        error_kw={'elinewidth': 0.7, 'ecolor': '#333333'}, zorder=3)

    for xi, e in zip(x, mean_energy):
        ax.text(xi - 0.34, e + 3, f'{e:.1f}', ha='center',
               va='bottom', fontsize=8)

    ax_steps.errorbar(
        x, mean_steps, yerr=std_steps, color='#1B7837', marker='D',
        markersize=4.5, markerfacecolor='white',
        markeredgecolor='#1B7837', markeredgewidth=1.0,
        linestyle='-', linewidth=1.1, capsize=2.0, elinewidth=0.7,
        ecolor='#1B7837', alpha=0.85, zorder=5)

    for xi, s in zip(x, mean_steps):
        ax_steps.text(xi + 0.34, s + 0.15, f'{s:.1f}', ha='center',
                     va='bottom', fontsize=7, color='#1B7837')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha='right')
    ax.set_ylabel(r'Cumulative Energy (dB$\cdot$steps)')
    ax_steps.set_ylabel('Mean steps to resolution', color='#1B7837')
    ax_steps.tick_params(axis='y', labelcolor='#1B7837')
    ax.set_title(
        'Mean cumulative compensation energy and steps per episode\n'
        r'(error bars = $\pm1\sigma$)')
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 310)
    ax_steps.set_ylim(0, 12.5)

    if 'S3: Best Nbr' in present:
        idx = present.index('S3: Best Nbr')
        ax.axhline(mean_energy[idx], color='#4393C3', lw=0.7,
                  linestyle=':')
        ax.text(len(present) - 2.5, mean_energy[idx] + 5.5,
               'Best heuristic', fontsize=7, color='#2166AC', ha='right')

    legend_handles = [
        mpatches.Patch(facecolor='#3498db', edgecolor='black',
                      label='Cumulative energy (bars, left axis)'),
        Line2D([0], [0], color='#1B7837', marker='D', markersize=4.5,
              markerfacecolor='white', linewidth=1.1,
              label='Mean steps (line, right axis)'),
    ]
    ax.legend(handles=legend_handles, loc='upper right',
             fontsize=5.5, framealpha=0.9)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "fig_cumulative_energy.png")
    plt.savefig(save_path, dpi=IEEE_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ============================================================
# Plot 3: Per-BS Coverage Heatmap
# ============================================================
def plot_per_bs_heatmap(all_results, num_bs=7, save_dir="docs/figures"):
    plt.rcParams['font.family'] = 'Serif'
    plt.rcParams['font.size'] = 12

    name_map = {
        'No Healing'    : 'No Healing',
        'S1: Fixed'     : 'S1: Power Boost',
        'S2: Proportional': 'S2: Proportional Power Boost',
        'S3: Best Nbr'  : 'S3: Best Single Neighbor',
        'S4: Simultaneous': 'S4: Simultaneous Power Boost',
        'S5: Tilt Only' : 'S5: Tilt Only',
        'S6: Joint'     : 'S6: Joint Power + Tilt',
        'DQN Agent'     : 'DQN Agent',
        'PPO Agent'     : 'PPO Agent',
    }
    selected = {name_map[k]: v for k, v in all_results.items()
               if k in name_map}

    bs_ids = list(range(num_bs))
    strat_names = list(selected.keys())
    matrix = np.zeros((len(strat_names), len(bs_ids)))

    for si, (name, res) in enumerate(selected.items()):
        for bi, bs_id in enumerate(bs_ids):
            mask = res['failed_bs_id'] == bs_id
            if mask.sum() > 0:
                matrix[si, bi] = np.mean(res['coverage_pct'][mask])

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(matrix, cmap='RdYlGn', vmin=85, vmax=100,
                  aspect='auto')
    plt.colorbar(im, ax=ax, label='Mean Coverage (%)', shrink=0.8)

    ax.set_xticks(range(len(bs_ids)))
    ax.set_xticklabels(
        [f'BS-{i}\n({"centre" if i==0 else "edge"})' for i in bs_ids],
        fontsize=11)
    ax.set_yticks(range(len(strat_names)))
    ax.set_yticklabels(strat_names, fontsize=11)
    ax.set_title(
        "Coverage Heatmap: Strategy \u00d7 Failed BS\n"
        "(green=high coverage, red=low coverage)",
        fontsize=12, fontweight='bold')

    for i in range(len(strat_names)):
        for j in range(len(bs_ids)):
            ax.text(j, i, f'{matrix[i, j]:.0f}%', ha='center',
                   va='center', fontsize=10, fontweight='bold',
                   color='black')

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "per_bs_heatmap.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ============================================================
# Plot 4: DQN Action Distribution (edge vs centre BS)
# ============================================================
def plot_action_distribution(dqn_results, num_bs=7,
                             save_dir="docs/figures"):
    actions = ['No Action', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6']
    colors = [
        '#95a5a6', '#3498db', '#2ecc71', '#e74c3c',
        '#9b59b6', '#f39c12', '#1abc9c'
    ]

    def compute_pct(bs_ids_sel):
        mask = np.isin(dqn_results['failed_bs_id'], bs_ids_sel)
        if mask.sum() == 0:
            return np.zeros(len(actions))
        all_actions = []
        for i in range(len(dqn_results['actions_taken'])):
            if mask[i]:
                all_actions.extend(dqn_results['actions_taken'][i])
        counts = np.zeros(len(actions))
        for a in all_actions:
            counts[a] += 1
        return counts / max(counts.sum(), 1) * 100

    edge_pct   = compute_pct(list(range(1, num_bs)))
    centre_pct = compute_pct([0])

    fig, ax = plt.subplots(figsize=(4.5, 3.0))

    x_left = np.arange(7)
    gap = 2
    x_right = np.arange(7) + 7 + gap

    bars_left = ax.bar(x_left, edge_pct, color=colors,
                       edgecolor='black', linewidth=0.5, width=0.75)
    bars_right = ax.bar(x_right, centre_pct, color=colors,
                        edgecolor='black', linewidth=0.5, width=0.75)

    for bar in list(bars_left) + list(bars_right):
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1.0,
                   f'{h:.1f}%', ha='center', va='bottom', fontsize=6)

    ax.set_xticks(list(x_left) + list(x_right))
    ax.set_xticklabels(actions + actions, rotation=20, ha='right')

    separator_x = 7 + gap / 2 - 0.5
    ax.axvline(separator_x, color='black', linestyle='--', linewidth=0.8)

    ax.text(np.mean(x_left), -10, 'Edge BS Failure', ha='center',
           va='top', fontsize=9)
    ax.text(np.mean(x_right), -10, 'Centre BS Failure', ha='center',
           va='top', fontsize=10)

    ax.set_ylabel('Action Usage (%)')
    ax.set_ylim(0, 65)
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)
    ax.set_title('DQN Agent Action Distribution')

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "action_distribution.png")
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ============================================================
# Orchestrator
# ============================================================
STRATEGY_ACTION_IDS = {
    'No Healing'       : 0,
    'S1: Fixed'        : 1,
    'S2: Proportional' : 2,
    'S3: Best Nbr'     : 3,
    'S4: Simultaneous' : 4,
    'S5: Tilt Only'    : 5,
    'S6: Joint'        : 6,
}


def evaluate_models(agents, config, n_episodes=500,
                    num_bs=7, save_dir="results",
                    fig_dir="docs/figures", verbose=True):
    """
    Full paper-results evaluation: compares trained RL
    agents against all six heuristic strategies (plus a
    no-action baseline) over n_episodes test episodes,
    and reproduces all four validated paper figures.
    """
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    raw_results = {}

    for name, action_id in STRATEGY_ACTION_IDS.items():
        if verbose:
            print(f"[evaluate_models] Evaluating '{name}'...", end=" ")
        raw_results[name] = evaluate_heuristic(
            name, action_id, config,
            n_episodes=n_episodes, seed_offset=9999)
        if verbose:
            cov = np.mean(raw_results[name]['coverage_pct'])
            sol = np.mean(raw_results[name]['solved']) * 100
            print(f"Cov={cov:.1f}% | Solved={sol:.1f}%")

    dqn_results = None
    for name, model in agents.items():
        if verbose:
            print(f"[evaluate_models] Evaluating '{name}'...", end=" ")
        raw_results[name] = evaluate_rl_agent(
            model, name, config,
            n_episodes=n_episodes, seed_offset=99999)
        if 'DQN' in name:
            dqn_results = raw_results[name]
        if verbose:
            cov = np.mean(raw_results[name]['coverage_pct'])
            sol = np.mean(raw_results[name]['solved']) * 100
            print(f"Cov={cov:.1f}% | Solved={sol:.1f}%")

    # Summary table
    rows = []
    for method, res in raw_results.items():
        rows.append({
            "method": method,
            "mean_coverage_pct": np.mean(res['coverage_pct']),
            "solve_rate_pct": np.mean(res['solved']) * 100,
            "mean_energy_dB_steps": np.mean(res['cumulative_energy']),
            "mean_steps": np.mean(res['n_steps_taken']),
        })
    summary_df = pd.DataFrame(rows).sort_values(
        "mean_coverage_pct", ascending=False)
    summary_df.to_csv(
        os.path.join(save_dir, "evaluation_summary.csv"), index=False)

    for method, res in raw_results.items():
        flat = {k: v for k, v in res.items() if k != 'actions_taken'}
        safe_name = method.replace(' ', '_').replace(':', '')
        pd.DataFrame(flat).to_csv(
            os.path.join(save_dir, f"raw_{safe_name}.csv"), index=False)

    if verbose:
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(summary_df.to_string(index=False))

    # Figures (all four validated plots)
    plot_coverage_comparison(raw_results, fig_dir)
    plot_cumulative_energy(raw_results, fig_dir)
    plot_per_bs_heatmap(raw_results, num_bs, fig_dir)
    if dqn_results is not None:
        plot_action_distribution(dqn_results, num_bs, fig_dir)

    return summary_df, raw_results
