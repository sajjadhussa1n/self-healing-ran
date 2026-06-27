"""
Single-call factory functions for creating DQN and PPO
agents bound to the SelfHealingEnv.
"""
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_util import make_vec_env

from src.environment.gym_env import SelfHealingNetworkEnv


def create_DQN_agent(config=None, env_kwargs=None,
                     policy_kwargs=None, **dqn_kwargs):
    """
    Create a DQN agent on a single SelfHealingEnv instance.

    Parameters
    ----------
    config : SimConfig or None
        Forwarded to SelfHealingEnv.
    env_kwargs : dict or None
        Extra kwargs forwarded to SelfHealingEnv(...).
    policy_kwargs : dict or None
        Forwarded to DQN(policy_kwargs=...).
    **dqn_kwargs :
        Any other DQN hyperparameters
        (learning_rate, buffer_size, gamma, etc.).

    Returns
    -------
    model : stable_baselines3.DQN
    env : SelfHealingEnv
    """
    env_kwargs = env_kwargs or {}
    env = SelfHealingNetworkEnv(config=config, **env_kwargs)
    if policy_kwargs is None:
        policy_kwargs = dict(net_arch=[256, 256, 128])

    default_kwargs = dict(
        learning_rate=3e-4,
        buffer_size=50000,
        learning_starts=500,
        batch_size=128,
        gamma=0.95,
        train_freq=1,
        target_update_interval=500,
        exploration_fraction=0.5,
        exploration_final_eps=0.05,
        verbose=0,
    )
    default_kwargs.update(dqn_kwargs)

    model = DQN("MlpPolicy", env,
               policy_kwargs=policy_kwargs,
               **default_kwargs)
    return model, env


def create_PPO_agent(config=None, env_kwargs=None,
                     n_envs=1, policy_kwargs=None,
                     **ppo_kwargs):
    """
    Create a PPO agent. Uses a (optionally vectorised)
    SelfHealingEnv.

    Parameters
    ----------
    config : SimConfig or None
    env_kwargs : dict or None
    n_envs : int
        Number of parallel environments (1 = single env,
        no vectorisation wrapper overhead).
    policy_kwargs : dict or None
    **ppo_kwargs :
        Any other PPO hyperparameters.

    Returns
    -------
    model : stable_baselines3.PPO
    env : VecEnv or SelfHealingEnv
    """
    env_kwargs = env_kwargs or {}

    if n_envs > 1:
        env = make_vec_env(
            lambda: SelfHealingNetworkEnv(config=config,
                                  **env_kwargs),
            n_envs=n_envs)
    else:
        env = SelfHealingNetworkEnv(config=config, **env_kwargs)

    if policy_kwargs is None:
        policy_kwargs = dict(net_arch=[256, 256, 128])

    default_kwargs = dict(
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64,
        gamma=0.95,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=0,
    )
    default_kwargs.update(ppo_kwargs)

    model = PPO("MlpPolicy", env,
               policy_kwargs=policy_kwargs,
               **default_kwargs)
    return model, env
