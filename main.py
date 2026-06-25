"""
main.py
=======
End-to-end pipeline for the Self-Healing 5G RAN project.

Runs, in order:
  1. Create a network and visualise it.
  2. Simulate a BS outage and visualise before/after.
  3. Train and evaluate COD models
     (Threshold, Logistic Regression, Random Forest).
  4. Apply all six heuristic COC strategies to the
     outage scenario and visualise each result.
  5. Create DQN and PPO agents.
  6. Train both agents.
  7. Evaluate all agents against all heuristics
     and produce the paper-ready result figures/tables.

Every step is a single function call -- see src/ for the
underlying implementation of each.
"""

from src.config import SimConfig
from src.network.factory import create_network, simulate_outage
from src.network.visualization import plot_network, plot_before_after
from src.detection.pipeline import train_cod_model
from src.compensation.pipeline import simulate_coc_strategies
from src.agents.factory import create_DQN_agent, create_PPO_agent
from src.agents.training import train_DRL_agents, load_trained_agents
from src.agents.evaluation import evaluate_models
from src.environment.gym_env import SelfHealingEnv


def main():
    config = SimConfig()
    FAILED_BS_ID = 0  # BS to fail for the illustrative demo

    # ------------------------------------------------------------------
    # 1. Create network and visualise normal operation
    # ------------------------------------------------------------------
    print("\n### STEP 1: Create network ###")
    network = create_network(config)
    plot_network(network, title="Network: Normal Operation",
                 save_path="docs/figures/01_network_normal.png",
                 show=False)

    # ------------------------------------------------------------------
    # 2. Simulate outage, visualise before/after
    # ------------------------------------------------------------------
    print("\n### STEP 2: Simulate outage ###")
    network_before, network_after = simulate_outage(
        network, failed_bs_id=FAILED_BS_ID, severity="full")
    plot_before_after(
        network_before, network_after,
        save_path="docs/figures/02_before_after_outage.png",
        show=False)

    # ------------------------------------------------------------------
    # 3. Train and evaluate COD models
    # ------------------------------------------------------------------
    print("\n### STEP 3: Train COD models ###")
    cod_results = train_cod_model(
        config=config, n_episodes=60,
        n_normal_steps=10, n_outage_steps=10,
        save_dir="models")

    # ------------------------------------------------------------------
    # 4. Simulate all heuristic COC strategies on the outage
    # ------------------------------------------------------------------
    print("\n### STEP 4: Simulate COC heuristic strategies ###")
    coc_results, coc_summary = simulate_coc_strategies(
        network_after, failed_bs_id=FAILED_BS_ID,
        save_dir="docs/figures")
    print("\nHeuristic strategy summary:")
    print(coc_summary.to_string(index=False))
    coc_summary.to_csv(
        "results/coc_heuristic_summary.csv", index=False)

    # ------------------------------------------------------------------
    # 5. Create DQN and PPO agents
    # ------------------------------------------------------------------
    print("\n### STEP 5: Create DQN and PPO agents ###")
    dqn_model, dqn_env = create_DQN_agent(config=config)
    ppo_model, ppo_env = create_PPO_agent(config=config)
    agents = {"DQN": dqn_model, "PPO": ppo_model}

    # ------------------------------------------------------------------
    # 6. Train OR load pre-trained DQN and PPO agents
    # ------------------------------------------------------------------
    USE_PRETRAINED = True  # ← flip to False to retrain from scratch
    
    print("\n### STEP 6: Train or load DQN and PPO agents ###")
    if USE_PRETRAINED:
        agents = load_trained_agents(
            agents, load_dir="models/pretrained")
    else:
        agents, training_times = train_DRL_agents(
            agents, total_timesteps=20_000, save_dir="models")
        print(f"Training times (s): {training_times}")

    # ------------------------------------------------------------------
    # 7. Evaluate all agents vs all heuristics, produce results
    # ------------------------------------------------------------------
    print("\n### STEP 7: Evaluate models and produce results ###")
    summary_df, raw_results = evaluate_models(
        agents=agents,
        env_factory=lambda: SelfHealingEnv(config=config),
        config=config,
        n_episodes=500,
        num_bs=config.NUM_BS,
        save_dir="results",
        fig_dir="docs/figures")

    print("\n### PIPELINE COMPLETE ###")
    print(f"Final summary saved to "
         f"results/evaluation_summary.csv")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
