# src/agents/train.py
"""
RL Agent Training and Evaluation
==================================
DQN and PPO training with curriculum learning.
Evaluation against all 6 heuristic strategies.
"""

import numpy as np
import time
import os
import warnings
warnings.filterwarnings('ignore')

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from src.config import CFG
from src.environment.gym_env import SelfHealingNetworkEnv

# ── Training configurations ───────────────────────────────────────────

TRAIN_CONFIG = {
    'total_timesteps'   : 20_000,
    'learning_rate'     : 3e-4,
    'batch_size'        : 64,
    'buffer_size'       : 10_000,
    'learning_starts'   : 500,
    'gamma'             : 0.95,
    'tau'               : 0.005,
    'train_freq'        : 1,
    'target_update'     : 500,
    'exploration_init'  : 1.0,
    'exploration_final' : 0.05,
    'exploration_frac'  : 0.4,
    'log_dir'           : './results/training_logs/',
    'model_dir'         : './models/',
    'model_name'        : 'dqn_self_healing',
}

PPO_CONFIG = {
    'total_timesteps': 20_000,
    'learning_rate'  : 3e-4,
    'n_steps'        : 256,
    'batch_size'     : 64,
    'n_epochs'       : 10,
    'gamma'          : 0.95,
    'gae_lambda'     : 0.95,
    'clip_range'     : 0.2,
    'ent_coef'       : 0.01,
    'vf_coef'        : 0.5,
    'max_grad_norm'  : 0.5,
    'model_name'     : 'ppo_self_healing',
}


# ── Training callbacks ────────────────────────────────────────────────

class TrainingCallback(BaseCallback):
    """Logs per-episode metrics during DQN training."""

    def __init__(self, log_interval: int = 200,
                 verbose: int = 0):
        super().__init__(verbose)
        self.log_interval      = log_interval
        self.episode_rewards   = []
        self.episode_coverages = []
        self.episode_solved    = []
        self.episode_steps     = []
        self.episode_failed_bs = []
        self.action_counts     = np.zeros(
            SelfHealingNetworkEnv.N_ACTIONS)
        self._ep_reward  = 0.0
        self._ep_steps   = 0
        self._n_episodes = 0

    def _on_step(self) -> bool:
        self._ep_reward += self.locals['rewards'][0]
        self._ep_steps  += 1
        self.action_counts[self.locals['actions'][0]] += 1

        done = (self.locals['dones'][0] or
                self.locals.get('truncateds',
                                 [False])[0])
        if done:
            self._n_episodes += 1
            info    = self.locals['infos'][0]
            stats   = info.get('stats_after', {})
            cov     = stats.get('coverage_pct',
                      info.get('coverage_pct', 0))
            solved  = (stats.get('ues_in_outage',
                       info.get('ues_in_outage', 1))
                       == 0)
            fail_id = info.get('failed_bs_id', -1)

            self.episode_rewards.append(self._ep_reward)
            self.episode_coverages.append(cov)
            self.episode_solved.append(int(solved))
            self.episode_steps.append(self._ep_steps)
            self.episode_failed_bs.append(fail_id)

            if (self._n_episodes %
                    self.log_interval == 0):
                n        = self.log_interval
                mean_r   = np.mean(
                    self.episode_rewards[-n:])
                mean_cov = np.mean(
                    self.episode_coverages[-n:])
                solve_r  = np.mean(
                    self.episode_solved[-n:]) * 100
                mean_s   = np.mean(
                    self.episode_steps[-n:])
                print(f"   Ep {self._n_episodes:5d} | "
                      f"Steps={self.num_timesteps:7d}"
                      f" | MeanR={mean_r:7.2f} | "
                      f"Cov={mean_cov:5.1f}% | "
                      f"Solved={solve_r:5.1f}% | "
                      f"AvgSteps={mean_s:.1f}")

            self._ep_reward = 0.0
            self._ep_steps  = 0
        return True

    def get_summary(self) -> dict:
        return {
            'episode_rewards'  : self.episode_rewards,
            'episode_coverages': self.episode_coverages,
            'episode_solved'   : self.episode_solved,
            'episode_steps'    : self.episode_steps,
            'episode_failed_bs': self.episode_failed_bs,
            'action_counts'    : self.action_counts,
        }


class PPOTrainingCallback(TrainingCallback):
    """Same as TrainingCallback, adapted for PPO."""

    def _on_step(self) -> bool:
        self._ep_reward += self.locals['rewards'][0]
        self._ep_steps  += 1
        self.action_counts[self.locals['actions'][0]] += 1

        done = self.locals['dones'][0]
        if done:
            self._n_episodes += 1
            info    = self.locals['infos'][0]
            stats   = info.get('stats_after', {})
            cov     = stats.get('coverage_pct',
                      info.get('coverage_pct', 0))
            solved  = (stats.get('ues_in_outage',
                       info.get('ues_in_outage', 1))
                       == 0)
            fail_id = info.get('failed_bs_id', -1)

            self.episode_rewards.append(self._ep_reward)
            self.episode_coverages.append(cov)
            self.episode_solved.append(int(solved))
            self.episode_steps.append(self._ep_steps)
            self.episode_failed_bs.append(fail_id)

            if (self._n_episodes %
                    self.log_interval == 0):
                n        = self.log_interval
                mean_r   = np.mean(
                    self.episode_rewards[-n:])
                mean_cov = np.mean(
                    self.episode_coverages[-n:])
                solve_r  = np.mean(
                    self.episode_solved[-n:]) * 100
                mean_s   = np.mean(
                    self.episode_steps[-n:])
                print(f"   Ep {self._n_episodes:5d} | "
                      f"Steps={self.num_timesteps:7d}"
                      f" | MeanR={mean_r:7.2f} | "
                      f"Cov={mean_cov:5.1f}% | "
                      f"Solved={solve_r:5.1f}% | "
                      f"AvgSteps={mean_s:.1f}")

            self._ep_reward = 0.0
            self._ep_steps  = 0
        return True


# ── Training functions ────────────────────────────────────────────────

def train_dqn(config=None, log_interval=200,
               save=True):
    """Train DQN agent and return model + callback."""
    if config is None:
        config = TRAIN_CONFIG

    os.makedirs(config['log_dir'],  exist_ok=True)
    os.makedirs(config['model_dir'], exist_ok=True)

    train_env = Monitor(
        SelfHealingNetworkEnv(
            config          = CFG,
            max_steps       = 10,
            n_normal_steps  = 5,
            use_curriculum  = True,
            suppress_output = True,
            verbose         = False),
        filename=config['log_dir'])

    model = DQN(
        policy               = 'MlpPolicy',
        env                  = train_env,
        learning_rate        = config['learning_rate'],
        batch_size           = config['batch_size'],
        buffer_size          = config['buffer_size'],
        learning_starts      = config['learning_starts'],
        gamma                = config['gamma'],
        tau                  = config['tau'],
        train_freq           = config['train_freq'],
        target_update_interval=config['target_update'],
        exploration_initial_eps =
            config['exploration_init'],
        exploration_final_eps   =
            config['exploration_final'],
        exploration_fraction    =
            config['exploration_frac'],
        policy_kwargs        = dict(
            net_arch=[256, 256, 128]),
        verbose              = 0,
        seed                 = CFG.RANDOM_SEED,
        device               = 'cpu')

    callback = TrainingCallback(
        log_interval=log_interval)
    t0       = time.time()

    print(f"\n{'─'*55}")
    print(f" DQN Training — "
          f"{config['total_timesteps']:,} timesteps")
    print(f"{'─'*55}\n")

    model.learn(
        total_timesteps = config['total_timesteps'],
        callback        = callback,
        progress_bar    = True)

    elapsed = time.time() - t0
    print(f"\n✅ DQN complete in "
          f"{elapsed/60:.1f} minutes.")

    if save:
        path = os.path.join(config['model_dir'],
                            config['model_name'])
        model.save(path)
        print(f"   Saved: {path}.zip")

    return model, callback


def train_ppo(config=None, log_interval=200,
               save=True):
    """Train PPO agent and return model + callback."""
    if config is None:
        config = PPO_CONFIG

    os.makedirs(TRAIN_CONFIG['log_dir'], exist_ok=True)
    os.makedirs(TRAIN_CONFIG['model_dir'], exist_ok=True)

    train_env = Monitor(
        SelfHealingNetworkEnv(
            config          = CFG,
            max_steps       = 10,
            n_normal_steps  = 5,
            use_curriculum  = True,
            suppress_output = True,
            verbose         = False))

    model = PPO(
        policy        = 'MlpPolicy',
        env           = train_env,
        learning_rate = config['learning_rate'],
        n_steps       = config['n_steps'],
        batch_size    = config['batch_size'],
        n_epochs      = config['n_epochs'],
        gamma         = config['gamma'],
        gae_lambda    = config['gae_lambda'],
        clip_range    = config['clip_range'],
        ent_coef      = config['ent_coef'],
        vf_coef       = config['vf_coef'],
        max_grad_norm = config['max_grad_norm'],
        policy_kwargs = dict(net_arch=[256, 256, 128]),
        verbose       = 0,
        seed          = CFG.RANDOM_SEED,
        device        = 'cpu')

    callback = PPOTrainingCallback(
        log_interval=log_interval)
    t0       = time.time()

    print(f"\n{'─'*55}")
    print(f" PPO Training — "
          f"{config['total_timesteps']:,} timesteps")
    print(f"{'─'*55}\n")

    model.learn(
        total_timesteps = config['total_timesteps'],
        callback        = callback,
        progress_bar    = True)

    elapsed = time.time() - t0
    print(f"\n✅ PPO complete in "
          f"{elapsed/60:.1f} minutes.")

    if save:
        path = os.path.join(
            TRAIN_CONFIG['model_dir'],
            config['model_name'])
        model.save(path)
        print(f"   Saved: {path}.zip")

    return model, callback


# ── Evaluation functions ──────────────────────────────────────────────

def evaluate_heuristic(strategy_name: str,
                        action_id: int,
                        n_episodes: int = 200,
                        seed_offset: int = 9999) -> dict:
    """
    Evaluate a fixed heuristic strategy.
    Applies the strategy on step 1, then does nothing.
    """
    results = {
        'coverage_pct'    : [],
        'mean_sinr_db'    : [],
        'ues_in_outage'   : [],
        'steps_to_solve'  : [],
        'solved'          : [],
        'failed_bs_id'    : [],
        'mean_throughput' : [],
        'total_power_boost': [],
    }

    env = SelfHealingNetworkEnv(
        config          = CFG,
        max_steps       = 10,
        n_normal_steps  = 5,
        use_curriculum  = False,
        suppress_output = True,
        verbose         = False)

    for ep in range(n_episodes):
        env._episode_count = seed_offset + ep
        obs, info          = env.reset()

        obs, reward, term, trunc, info = env.step(
            action_id)
        done = term or trunc
        while not done:
            obs, reward, term, trunc, info = env.step(0)
            done = term or trunc

        stats  = info['stats_after']
        solved = (stats['ues_in_outage'] == 0)
        total_boost = sum(
            max(bs.tx_power_dbm -
                bs.nominal_power_dbm, 0)
            for bs in env.network.base_stations
            if bs.is_active)

        results['coverage_pct'].append(
            stats['coverage_pct'])
        results['mean_sinr_db'].append(
            stats['mean_sinr_db'])
        results['ues_in_outage'].append(
            stats['ues_in_outage'])
        results['solved'].append(int(solved))
        results['failed_bs_id'].append(
            info['failed_bs_id'])
        results['mean_throughput'].append(
            stats.get('mean_throughput', 0))
        results['total_power_boost'].append(total_boost)
        results['steps_to_solve'].append(
            1 if solved else 10)

    return {k: np.array(v)
            for k, v in results.items()}


def evaluate_rl_agent(model, n_episodes: int = 200,
                       seed_offset: int = 99999) -> dict:
    """
    Evaluate a trained RL agent.
    Uses deterministic (greedy) action selection.
    """
    results = {
        'coverage_pct'    : [],
        'mean_sinr_db'    : [],
        'ues_in_outage'   : [],
        'steps_to_solve'  : [],
        'solved'          : [],
        'failed_bs_id'    : [],
        'mean_throughput' : [],
        'total_power_boost': [],
        'actions_taken'   : [],
    }

    env = SelfHealingNetworkEnv(
        config          = CFG,
        max_steps       = 10,
        n_normal_steps  = 5,
        use_curriculum  = False,
        suppress_output = True,
        verbose         = False)

    for ep in range(n_episodes):
        env._episode_count = seed_offset + ep
        obs, info          = env.reset()

        done       = False
        ep_actions = []
        step_solved = None

        while not done:
            action, _ = model.predict(
                obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(
                int(action))
            ep_actions.append(int(action))
            done = term or trunc
            if term and step_solved is None:
                step_solved = env._step_count

        stats  = info['stats_after']
        solved = (stats['ues_in_outage'] == 0)
        total_boost = sum(
            max(bs.tx_power_dbm -
                bs.nominal_power_dbm, 0)
            for bs in env.network.base_stations
            if bs.is_active)

        results['coverage_pct'].append(
            stats['coverage_pct'])
        results['mean_sinr_db'].append(
            stats['mean_sinr_db'])
        results['ues_in_outage'].append(
            stats['ues_in_outage'])
        results['solved'].append(int(solved))
        results['failed_bs_id'].append(
            info['failed_bs_id'])
        results['mean_throughput'].append(
            stats.get('mean_throughput', 0))
        results['total_power_boost'].append(
            total_boost)
        results['steps_to_solve'].append(
            step_solved if step_solved
            else env._step_count)
        results['actions_taken'].append(ep_actions)

    return {k: (np.array(v)
                if k != 'actions_taken' else v)
            for k, v in results.items()}
