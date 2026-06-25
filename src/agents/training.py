"""
Single-call training function for DQN/PPO agents,
with optional curriculum scheduling and progress logging.
"""
import os
import time


def train_DRL_agents(agents, total_timesteps=20_000,
                     save_dir="models",
                     model_names=None,
                     log_progress=True,
                     callback=None):
    """
    Train one or more agents (e.g. DQN and PPO) for a
    fixed number of timesteps each, and save the trained
    weights to disk.

    Parameters
    ----------
    agents : dict[str, stable_baselines3 model]
        e.g. {"DQN": dqn_model, "PPO": ppo_model}
        (as returned by create_DQN_agent /
        create_PPO_agent).
    total_timesteps : int or dict[str, int]
        Number of training steps. If a dict, allows
        per-agent timestep budgets.
    save_dir : str
        Directory to save trained model .zip files.
    model_names : dict[str, str] or None
        Optional override for save filenames
        (default: lowercase agent key).
    log_progress : bool
    callback : stable_baselines3 callback or None
        Forwarded to model.learn(callback=...).

    Returns
    -------
    trained_agents : dict[str, model]
        Same dict, agents now trained in-place
        (and also returned for convenience).
    training_times : dict[str, float]
        Wall-clock seconds spent training each agent.
    """
    os.makedirs(save_dir, exist_ok=True)
    training_times = {}

    for name, model in agents.items():
        steps = (total_timesteps[name]
                if isinstance(total_timesteps, dict)
                else total_timesteps)

        if log_progress:
            print(f"\n{'=' * 60}")
            print(f"Training {name} for {steps:,} "
                 f"timesteps")
            print(f"{'=' * 60}")

        t0 = time.time()
        model.learn(total_timesteps=steps,
                   callback=callback,
                   progress_bar=log_progress)
        elapsed = time.time() - t0
        training_times[name] = elapsed

        fname = (model_names.get(name, name.lower())
                if model_names else name.lower())
        save_path = os.path.join(save_dir, fname)
        model.save(save_path)

        if log_progress:
            print(f"{name} trained in {elapsed:.1f}s. "
                 f"Saved -> {save_path}.zip")

    return agents, training_times
