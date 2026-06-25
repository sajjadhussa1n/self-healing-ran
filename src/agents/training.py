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

def load_trained_agents(agents, load_dir="models/pretrained",
                        model_names=None, log_progress=True):
    """
    Load pre-trained weights into existing agent objects,
    skipping training entirely. Drop-in replacement for
    train_DRL_agents() when you already have trained
    model .zip files (e.g. the ones used in the paper).

    Parameters
    ----------
    agents : dict[str, stable_baselines3 model]
        e.g. {"DQN": dqn_model, "PPO": ppo_model}
        — these must already be created with the SAME
        env/architecture as create_DQN_agent /
        create_PPO_agent, since only the weights/policy
        are loaded, not the environment.
    load_dir : str
        Directory containing the saved .zip files.
    model_names : dict[str, str] or None
        Optional override for filenames
        (default: lowercase agent key, e.g. "dqn.zip").
    log_progress : bool

    Returns
    -------
    trained_agents : dict[str, model]
        Same dict, with each model's policy loaded from disk
        (loaded in-place via .set_parameters, and also
        returned for convenience).
    """
    for name, model in agents.items():
        fname = (model_names.get(name, name.lower())
                if model_names else name.lower())
        path = os.path.join(load_dir, f"{fname}.zip")

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Pre-trained model not found: {path}")

        model.set_parameters(path)

        if log_progress:
            print(f" Loaded pre-trained weights for "
                 f"{name} <- {path}")

    return agents
