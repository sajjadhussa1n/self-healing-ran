"""
main.py
=======
End-to-end pipeline for the Self-Healing 5G RAN project.
All parameters are controlled via pipeline_config.yaml —
edit that file instead of this one.
"""
import os
from src.utils.config_loader import load_pipeline_config
from src.network.factory import create_network, simulate_outage
from src.network.visualization import plot_network
from src.detection.pipeline import train_cod_model
from src.compensation.pipeline import simulate_coc_strategies
from src.agents.factory import create_DQN_agent, create_PPO_agent
from src.agents.training import train_DRL_agents, load_trained_agents
from src.agents.evaluation import evaluate_models
from src.environment.gym_env import SelfHealingNetworkEnv


def main():
    config, pcfg = load_pipeline_config("pipeline_config.yaml")

    paths = pcfg["paths"]
    for d in [paths["models_dir"], paths["results_dir"],
              paths["figures_dir"]]:
        os.makedirs(d, exist_ok=True)

    FAILED_BS_ID = pcfg["demo"]["failed_bs_id"]
    SEVERITY = pcfg["demo"]["outage_severity"]

    # ------------------------------------------------------------------
    # 1. Create network and visualise normal operation
    # ------------------------------------------------------------------
    print("\n### STEP 1: Create network ###")
    network = create_network(config)
    plot_network(network, title="Before Outage (Normal Operation)",
                save_path=os.path.join(paths["figures_dir"],
                                       "01_before_outage.png"),
                show=True)

    # ------------------------------------------------------------------
    # 2. Simulate outage, visualise before/after
    # ------------------------------------------------------------------
    print("\n### STEP 2: Simulate outage ###")
    network_before, network_after = simulate_outage(
        network, failed_bs_id=FAILED_BS_ID, severity=SEVERITY)
    plot_network(network_after,
                title=f"After Outage (BS-{FAILED_BS_ID} Failed)",
                save_path=os.path.join(paths["figures_dir"],
                                       "02_after_outage.png"),
                show=True)

    # ------------------------------------------------------------------
    # 3. Train and evaluate COD models
    # ------------------------------------------------------------------
    print("\n### STEP 3: Train COD models ###")
    cod_cfg = pcfg["cod"]
    cod_results = train_cod_model(
        config=config,
        n_episodes=cod_cfg["n_episodes"],
        n_normal_steps=cod_cfg["n_normal_steps"],
        n_outage_steps=cod_cfg["n_outage_steps"],
        save_dir=paths["models_dir"])

    # ------------------------------------------------------------------
    # 4. Simulate all heuristic COC strategies on the outage
    # ------------------------------------------------------------------
    print("\n### STEP 4: Simulate COC heuristic strategies ###")
    coc_results, coc_summary = simulate_coc_strategies(
        network_after, failed_bs_id=FAILED_BS_ID,
        save_dir=paths["figures_dir"])
    print("\nHeuristic strategy summary:")
    print(coc_summary.to_string(index=False))
    coc_summary.to_csv(
        os.path.join(paths["results_dir"],
                     "coc_heuristic_summary.csv"), index=False)

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
    print("\n### STEP 6: Train or load DQN and PPO agents ###")
    agent_cfg = pcfg["agents"]
    if agent_cfg["use_pretrained"]:
        agents = load_trained_agents(
            agents,
            load_dir=agent_cfg["pretrained_dir"],
            model_names=agent_cfg["model_names"])
    else:
        agents, training_times = train_DRL_agents(
            agents,
            total_timesteps=agent_cfg["total_timesteps"],
            save_dir=agent_cfg["trained_save_dir"],
            model_names=agent_cfg["model_names"])
        print(f"Training times (s): {training_times}")

    # ------------------------------------------------------------------
    # 7. Evaluate all agents vs all heuristics, produce results
    # ------------------------------------------------------------------
    print("\n### STEP 7: Evaluate models and produce results ###")
    eval_cfg = pcfg["evaluation"]
    summary_df, raw_results = evaluate_models(
        agents=agents,
        config=config,
        n_episodes=eval_cfg["n_episodes"],
        num_bs=config.NUM_BS,
        save_dir=paths["results_dir"],
        fig_dir=paths["figures_dir"])


if __name__ == "__main__":
    main()
