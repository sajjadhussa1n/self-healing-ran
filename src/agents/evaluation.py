# src/agents/evaluation.py
"""
Evaluation pipeline for comparing trained RL agents against
heuristic compensation strategies.

evaluate_heuristic() and evaluate_rl_agent() are the exact
functions validated in the project notebooks and used to
produce the paper's Table III results — do not alter their
internal logic (seed_offset scheme, step-then-hold heuristic
procedure, etc.) without re-validating against the paper.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.environment.gym_env import SelfHealingNetworkEnv


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

        # Apply heuristic once on step 1
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

        # Hold (no-op) for remaining steps
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
# Orchestrator (replaces previous evaluate_models internals)
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
    using the exact validated evaluate_heuristic /
    evaluate_rl_agent procedures.

    Parameters
    ----------
    agents : dict[str, SB3 model]
        e.g. {"DQN": dqn_model, "PPO": ppo_model}.
    config : SimConfig
    n_episodes : int
    num_bs : int
    save_dir, fig_dir : str
    verbose : bool

    Returns
    -------
    summary_df : pd.DataFrame
    raw_results : dict[str, dict]
        Method name -> full results dict
        (as returned by evaluate_heuristic / evaluate_rl_agent).
    """
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    raw_results = {}

    # --- Heuristic strategies (incl. No Healing) ---
    for name, action_id in STRATEGY_ACTION_IDS.items():
        if verbose:
            print(f"[evaluate_models] Evaluating "
                 f"'{name}'...", end=" ")
        raw_results[name] = evaluate_heuristic(
            name, action_id, config,
            n_episodes=n_episodes, seed_offset=9999)
        if verbose:
            cov = np.mean(raw_results[name]['coverage_pct'])
            sol = np.mean(raw_results[name]['solved']) * 100
            print(f"Cov={cov:.1f}% | Solved={sol:.1f}%")

    # --- RL agents ---
    for name, model in agents.items():
        if verbose:
            print(f"[evaluate_models] Evaluating "
                 f"'{name}'...", end=" ")
        raw_results[name] = evaluate_rl_agent(
            model, name, config,
            n_episodes=n_episodes, seed_offset=99999)
        if verbose:
            cov = np.mean(raw_results[name]['coverage_pct'])
            sol = np.mean(raw_results[name]['solved']) * 100
            print(f"Cov={cov:.1f}% | Solved={sol:.1f}%")

    # --- Summary table ---
    rows = []
    for method, res in raw_results.items():
        rows.append({
            "method": method,
            "mean_coverage_pct": np.mean(res['coverage_pct']),
            "solve_rate_pct": np.mean(res['solved']) * 100,
            "mean_energy_dB_steps": np.mean(
                res['cumulative_energy']),
            "mean_steps": np.mean(res['n_steps_taken']),
        })
    summary_df = pd.DataFrame(rows).sort_values(
        "mean_coverage_pct", ascending=False)
    summary_df.to_csv(
        os.path.join(save_dir, "evaluation_summary.csv"),
        index=False)

    for method, res in raw_results.items():
        # actions_taken is a list-of-lists for RL agents;
        # drop it before building a flat per-episode CSV
        flat = {k: v for k, v in res.items()
               if k != 'actions_taken'}
        pd.DataFrame(flat).to_csv(
            os.path.join(save_dir,
                        f"raw_{method.replace(' ', '_').replace(':','')}.csv"),
            index=False)

    if verbose:
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(summary_df.to_string(index=False))

    # --- Figures ---
    _plot_coverage_solve_rate(summary_df, fig_dir)
    _plot_energy_steps(summary_df, fig_dir)
    _plot_action_distribution(raw_results, agents.keys(), fig_dir)
    _plot_bs_heatmap(raw_results, agents.keys(), num_bs, fig_dir)

    return summary_df, raw_results


def _plot_coverage_solve_rate(summary_df, fig_dir):
    fig, ax1 = plt.subplots(figsize=(10, 5))
    x = np.arange(len(summary_df))
    ax1.bar(x - 0.2, summary_df["mean_coverage_pct"],
           width=0.4, label="Coverage (%)", color="steelblue")
    ax1.set_ylabel("Mean Coverage (%)")
    ax2 = ax1.twinx()
    ax2.bar(x + 0.2, summary_df["solve_rate_pct"],
           width=0.4, label="Solve Rate (%)", color="indianred")
    ax2.set_ylabel("Solve Rate (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(summary_df["method"], rotation=45, ha="right")
    fig.legend(loc="upper right")
    plt.title("Coverage & Solve Rate by Method")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "coverage_solve_rate.png"), dpi=300)
    plt.close(fig)


def _plot_energy_steps(summary_df, fig_dir):
    fig, ax1 = plt.subplots(figsize=(10, 5))
    x = np.arange(len(summary_df))
    ax1.bar(x - 0.2, summary_df["mean_energy_dB_steps"],
           width=0.4, label="Energy (dB·steps)", color="darkorange")
    ax1.set_ylabel("Mean Cumulative Energy (dB·steps)")
    ax2 = ax1.twinx()
    ax2.plot(x, summary_df["mean_steps"], "o-",
           color="seagreen", label="Mean Steps")
    ax2.set_ylabel("Mean Steps to Resolution")
    ax1.set_xticks(x)
    ax1.set_xticklabels(summary_df["method"], rotation=45, ha="right")
    fig.legend(loc="upper right")
    plt.title("Compensation Energy & Steps by Method")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "energy_steps.png"), dpi=300)
    plt.close(fig)


def _plot_action_distribution(raw_results, agent_names, fig_dir):
    for name in agent_names:
        res = raw_results.get(name)
        if res is None or "actions_taken" not in res:
            continue
        all_actions = [a for lst in res["actions_taken"] for a in lst]
        if not all_actions:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(all_actions, bins=np.arange(max(all_actions) + 2) - 0.5,
               rwidth=0.8, color="slateblue")
        ax.set_xlabel("Action ID")
        ax.set_ylabel("Frequency")
        ax.set_title(f"{name} Action Distribution")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f"action_dist_{name.lower()}.png"),
                   dpi=300)
        plt.close(fig)


def _plot_bs_heatmap(raw_results, agent_names, num_bs, fig_dir):
    for name in agent_names:
        res = raw_results.get(name)
        if res is None or "failed_bs_id" not in res:
            continue
        df = pd.DataFrame({
            "failed_bs": res["failed_bs_id"],
            "coverage_pct": res["coverage_pct"],
        })
        pivot = df.groupby("failed_bs")["coverage_pct"].mean().reindex(
            range(num_bs))
        fig, ax = plt.subplots(figsize=(8, 1.5))
        sns.heatmap(pivot.to_frame().T, annot=True, fmt=".1f",
                   cmap="RdYlGn", cbar_kws={"label": "Coverage (%)"}, ax=ax)
        ax.set_yticks([])
        ax.set_xlabel("Failed BS ID")
        ax.set_title(f"{name}: Per-BS Coverage Heatmap")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f"bs_heatmap_{name.lower()}.png"),
                   dpi=300)
        plt.close(fig)
