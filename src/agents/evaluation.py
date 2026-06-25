"""
Single-call evaluation function: runs a head-to-head
comparison of trained RL agents against the six heuristic
strategies (plus no-action baseline), producing the
paper-ready metrics: coverage, solve rate, cumulative
compensation energy, mean steps to resolution, action
distribution, and per-BS coverage heatmap.
"""
import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.network.factory import create_network, simulate_outage
from src.compensation.pipeline import STRATEGY_FUNCS


def _rollout_agent(model, env, n_episodes, deterministic=True):
    """Run a trained SB3 model for n_episodes on env,
    collecting per-episode metrics."""
    records = []
    for ep in range(n_episodes):
        obs, info = env.reset()
        done, truncated = False, False
        steps = 0
        energy = 0.0
        actions_taken = []

        while not (done or truncated):
            action, _ = model.predict(
                obs, deterministic=deterministic)
            obs, reward, done, truncated, info = env.step(
                int(action))
            steps += 1
            energy += info.get("step_energy", 0.0)
            actions_taken.append(int(action))

        records.append({
            "episode": ep,
            "failed_bs": info.get("failed_bs_id"),
            "coverage_pct": info.get("coverage_pct"),
            "solved": info.get("coverage_pct", 0) >= 99.999,
            "steps": steps,
            "energy": energy,
            "actions": actions_taken,
        })
    return pd.DataFrame(records)


def _rollout_heuristic(strategy_name, config, n_episodes,
                       num_bs):
    """Apply a fixed heuristic strategy across n_episodes,
    one random BS failure per episode, single-shot
    compensation (no multi-step iteration)."""
    records = []
    for ep in range(n_episodes):
        failed_bs = ep % num_bs
        net = create_network(config, seed=1000 + ep,
                            verbose=False)
        _, net_after = simulate_outage(
            net, failed_bs, verbose=False)

        net_compensated = copy.deepcopy(net_after)
        if strategy_name != "No Action":
            method = STRATEGY_FUNCS[strategy_name]
            getattr(net_compensated, method)(failed_bs)
            net_compensated.compute_association_and_sinr()

        stats = net_compensated.get_stats()
        boost = getattr(net_compensated,
                       "last_energy_used", 0.0)

        records.append({
            "episode": ep,
            "failed_bs": failed_bs,
            "coverage_pct": stats["coverage_pct"],
            "solved": stats["coverage_pct"] >= 99.999,
            "steps": 1,
            "energy": boost,
        })
    return pd.DataFrame(records)


def evaluate_models(agents, env_factory, config,
                    n_episodes=500, num_bs=7,
                    save_dir="results",
                    fig_dir="docs/figures",
                    strategies=None, verbose=True):
    """
    Full paper-results evaluation: compares trained RL
    agents against all six heuristic compensation
    strategies (plus a no-action baseline) over
    n_episodes test episodes.

    Parameters
    ----------
    agents : dict[str, SB3 model]
        e.g. {"DQN": dqn_model, "PPO": ppo_model}.
    env_factory : callable
        Zero-arg callable returning a fresh SelfHealingEnv
        instance for agent rollouts
        (e.g. lambda: SelfHealingEnv(config=config)).
    config : SimConfig
        Used for heuristic-strategy rollouts.
    n_episodes : int
        Number of test episodes per method.
    num_bs : int
        Number of BSs (for round-robin failure selection
        and the per-BS heatmap).
    save_dir : str
        Directory for result CSVs.
    fig_dir : str
        Directory for result figures.
    strategies : list[str] or None
        Heuristic strategies to include
        (default: all S1-S6 + No Action).
    verbose : bool

    Returns
    -------
    summary_df : pd.DataFrame
        One row per method with mean coverage, solve rate,
        mean energy, mean steps.
    raw_results : dict[str, pd.DataFrame]
        Full per-episode results for every method.
    """
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    strategies = strategies or (
        ["No Action"] + list(STRATEGY_FUNCS.keys()))

    raw_results = {}

    # --- Heuristic strategies ---
    for strat in strategies:
        if verbose:
            print(f"[evaluate_models] Running heuristic "
                 f"'{strat}' for {n_episodes} episodes...")
        raw_results[strat] = _rollout_heuristic(
            strat, config, n_episodes, num_bs)

    # --- RL agents ---
    for name, model in agents.items():
        if verbose:
            print(f"[evaluate_models] Running agent "
                 f"'{name}' for {n_episodes} episodes...")
        env = env_factory()
        raw_results[name] = _rollout_agent(
            model, env, n_episodes)

    # --- Summary table ---
    rows = []
    for method, df in raw_results.items():
        rows.append({
            "method": method,
            "mean_coverage_pct": df["coverage_pct"].mean(),
            "solve_rate_pct": 100 * df["solved"].mean(),
            "mean_energy_dB_steps": df["energy"].mean(),
            "mean_steps": df["steps"].mean(),
        })
    summary_df = pd.DataFrame(rows).sort_values(
        "mean_coverage_pct", ascending=False)
    summary_df.to_csv(
        os.path.join(save_dir, "evaluation_summary.csv"),
        index=False)

    for method, df in raw_results.items():
        df.to_csv(os.path.join(
            save_dir, f"raw_{method.replace(' ', '_')}.csv"),
            index=False)

    if verbose:
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(summary_df.to_string(index=False))

    # --- Figures ---
    _plot_coverage_solve_rate(summary_df, fig_dir)
    _plot_energy_steps(summary_df, fig_dir)
    _plot_action_distribution(raw_results, agents.keys(),
                              fig_dir)
    _plot_bs_heatmap(raw_results, agents.keys(), num_bs,
                     fig_dir)

    return summary_df, raw_results


def _plot_coverage_solve_rate(summary_df, fig_dir):
    fig, ax1 = plt.subplots(figsize=(10, 5))
    x = np.arange(len(summary_df))
    ax1.bar(x - 0.2, summary_df["mean_coverage_pct"],
           width=0.4, label="Coverage (%)", color="steelblue")
    ax1.set_ylabel("Mean Coverage (%)")
    ax2 = ax1.twinx()
    ax2.bar(x + 0.2, summary_df["solve_rate_pct"],
           width=0.4, label="Solve Rate (%)",
           color="indianred")
    ax2.set_ylabel("Solve Rate (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(summary_df["method"], rotation=45,
                        ha="right")
    fig.legend(loc="upper right")
    plt.title("Coverage & Solve Rate by Method")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir,
              "coverage_solve_rate.png"), dpi=300)
    plt.close(fig)


def _plot_energy_steps(summary_df, fig_dir):
    fig, ax1 = plt.subplots(figsize=(10, 5))
    x = np.arange(len(summary_df))
    ax1.bar(x - 0.2, summary_df["mean_energy_dB_steps"],
           width=0.4, label="Energy (dB·steps)",
           color="darkorange")
    ax1.set_ylabel("Mean Cumulative Energy (dB·steps)")
    ax2 = ax1.twinx()
    ax2.plot(x, summary_df["mean_steps"], "o-",
           color="seagreen", label="Mean Steps")
    ax2.set_ylabel("Mean Steps to Resolution")
    ax1.set_xticks(x)
    ax1.set_xticklabels(summary_df["method"], rotation=45,
                        ha="right")
    fig.legend(loc="upper right")
    plt.title("Compensation Energy & Steps by Method")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir,
              "energy_steps.png"), dpi=300)
    plt.close(fig)


def _plot_action_distribution(raw_results, agent_names,
                              fig_dir):
    for name in agent_names:
        df = raw_results.get(name)
        if df is None or "actions" not in df.columns:
            continue
        all_actions = [a for lst in df["actions"]
                      for a in lst]
        if not all_actions:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(all_actions,
               bins=np.arange(max(all_actions) + 2) - 0.5,
               rwidth=0.8, color="slateblue")
        ax.set_xlabel("Action ID")
        ax.set_ylabel("Frequency")
        ax.set_title(f"{name} Action Distribution")
        plt.tight_layout()
        plt.savefig(os.path.join(
            fig_dir, f"action_dist_{name.lower()}.png"),
            dpi=300)
        plt.close(fig)


def _plot_bs_heatmap(raw_results, agent_names, num_bs,
                     fig_dir):
    for name in agent_names:
        df = raw_results.get(name)
        if df is None or "failed_bs" not in df.columns:
            continue
        pivot = df.groupby("failed_bs")[
            "coverage_pct"].mean().reindex(
            range(num_bs))
        fig, ax = plt.subplots(figsize=(8, 1.5))
        sns.heatmap(pivot.to_frame().T, annot=True,
                   fmt=".1f", cmap="RdYlGn",
                   cbar_kws={"label": "Coverage (%)"},
                   ax=ax)
        ax.set_yticks([])
        ax.set_xlabel("Failed BS ID")
        ax.set_title(f"{name}: Per-BS Coverage Heatmap")
        plt.tight_layout()
        plt.savefig(os.path.join(
            fig_dir, f"bs_heatmap_{name.lower()}.png"),
            dpi=300)
        plt.close(fig)
