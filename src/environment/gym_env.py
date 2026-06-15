# src/environment/gym_env.py
"""
Gymnasium Environment — Self-Healing Radio Network
===================================================
Wraps RadioNetwork as an OpenAI Gymnasium environment
for DRL-based cell outage compensation.

State  : 52-dimensional normalised observation vector
Actions: 7 discrete compensation strategies (S1-S6 + do nothing)
Reward : UE-rescue based with collateral penalty
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import sys
import os

from src.config import SimConfig, CFG
from src.network.radio_network import RadioNetwork
from src.detection.simulator import move_ues


class SelfHealingNetworkEnv(gym.Env):
    """
    Gymnasium environment for self-healing RAN.

    Observation (52 features):
      Per BS (7 × 6 = 42):
        ue_count_norm  : [0, 2]   >1 means overloaded
        sinr_norm      : [0, 1]
        delta_ue_norm  : [-1, 1]  signed
        tilt_norm      : [0, 1]
        power_norm     : [-1, 1]  -1 = failed BS sentinel
        is_active      : {0, 1}
      Global (10):
        failed_bs_onehot : 7 binary
        coverage_pct     : [0, 1]
        outage_ues_norm  : [0, 1]
        timestep_norm    : [0, 1]

    Actions (7 discrete):
      0: Do Nothing
      1: S1 Fixed Power Boost
      2: S2 Proportional Power Boost
      3: S3 Best Single Neighbour
      4: S4 Simultaneous Power Boost
      5: S5 Tilt Reduction Only
      6: S6 Joint Power + Tilt

    Reward:
      +2.0 per rescued UE           (primary)
      -1.5 per collateral UE        (safety)
      +5.0 × Δcoverage              (continuity)
      +1.0 × above_baseline         (maintenance)
      +10.0 success bonus
      -5.0  timeout penalty
    """

    metadata     = {'render_modes': []}
    ACTION_NAMES = {
        0: 'Do Nothing',
        1: 'S1: Fixed Power Boost',
        2: 'S2: Proportional Power Boost',
        3: 'S3: Best Single Neighbour',
        4: 'S4: Simultaneous Power Boost',
        5: 'S5: Tilt Reduction Only',
        6: 'S6: Joint Power + Tilt',
    }
    N_ACTIONS    = 7
    SINR_OBS_MIN = -10.0
    SINR_OBS_MAX =  35.0

    def __init__(self,
                 config          : SimConfig = None,
                 max_steps       : int   = 10,
                 n_normal_steps  : int   = 5,
                 ue_step_size_m  : float = 10.0,
                 failed_bs_id    : int   = None,
                 use_curriculum  : bool  = True,
                 suppress_output : bool  = True,
                 verbose         : bool  = False):
        super().__init__()
        self.cfg             = config if config else CFG
        self.base_max_steps  = max_steps
        self.max_steps       = max_steps
        self.n_normal_steps  = n_normal_steps
        self.ue_step_size    = ue_step_size_m
        self.fixed_fail_id   = failed_bs_id
        self.use_curriculum  = use_curriculum
        self.suppress_output = suppress_output
        self.verbose         = verbose
        self._max_ues        = (self.cfg.NUM_UE /
                                self.cfg.NUM_BS)
        self._episode_count  = 0
        self._old_stdout     = sys.stdout

        n_obs = self.cfg.NUM_BS * 6 + self.cfg.NUM_BS + 3
        self.observation_space = spaces.Box(
            low=-1.0, high=2.0,
            shape=(n_obs,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.N_ACTIONS)

        self.network         = None
        self.failed_bs_id    = None
        self._step_count     = 0
        self._prev_ue_counts = {}
        self._rng            = None

    # ── stdout suppression ────────────────────────────────────────────

    def _suppress(self) -> None:
        if self.suppress_output:
            self._old_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')

    def _restore(self) -> None:
        if self.suppress_output:
            try:
                sys.stdout.close()
            except Exception:
                pass
            sys.stdout = self._old_stdout

    # ── Gymnasium API ─────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._episode_count += 1
        self._step_count     = 0

        use_edge_only = (
            self.use_curriculum and
            self.fixed_fail_id is None and
            self._episode_count 
            self.cfg.RL_CURRICULUM_EPISODES)

        ep_seed            = (self.cfg.RANDOM_SEED +
                               self._episode_count * 100)
        self._rng          = np.random.RandomState(ep_seed)
        ep_cfg             = SimConfig()
        ep_cfg.RANDOM_SEED = ep_seed
        self.network       = RadioNetwork(ep_cfg)

        for t in range(self.n_normal_steps):
            if t > 0:
                move_ues(self.network,
                          self.ue_step_size, self._rng)
            self.network.compute_association_and_sinr()

        self._prev_ue_counts = {
            bs.bs_id: len([
                ue for ue in self.network.ues
                if ue.serving_bs_id == bs.bs_id])
            for bs in self.network.base_stations}

        if self.fixed_fail_id is not None:
            self.failed_bs_id = self.fixed_fail_id
        elif use_edge_only:
            self.failed_bs_id = int(
                self._rng.randint(1, self.cfg.NUM_BS))
        else:
            self.failed_bs_id = int(
                self._rng.randint(0, self.cfg.NUM_BS))

        self._suppress()
        self.network.trigger_outage(
            self.failed_bs_id, severity='full')
        self.network.compute_association_and_sinr()
        self._restore()

        stats = self.network.get_stats()
        self._baseline_coverage   = (
            stats['coverage_pct'] / 100.0)
        self._baseline_outage_ues = (
            stats['ues_in_outage'])
        self.max_steps = max(
            5, min(
                int(self._baseline_outage_ues * 0.8) + 3,
                self.base_max_steps))
        self._prev_coverage = self._baseline_coverage
        self._prev_sinr     = stats['mean_sinr_db']

        if self.verbose:
            phase = ("CURRICULUM" if use_edge_only
                     else "FULL")
            print(f"\n[{phase}] Ep "
                  f"{self._episode_count}: "
                  f"BS-{self.failed_bs_id} | "
                  f"Outage={self._baseline_outage_ues}"
                  f" | MaxSteps={self.max_steps}")

        return self._get_observation(), self._get_info()

    def step(self, action: int):
        assert self.network is not None
        self._step_count += 1

        ues_outage_before = {
            ue.ue_id: ue.in_outage
            for ue in self.network.ues}
        stats_before = self.network.get_stats()

        self._suppress()
        self._apply_action(action)
        self.network.compute_association_and_sinr()
        self._restore()

        stats_after = self.network.get_stats()
        reward      = self._compute_reward(
            stats_before, stats_after,
            ues_outage_before)

        self._prev_coverage = (
            stats_after['coverage_pct'] / 100.0)
        self._prev_sinr  = stats_after['mean_sinr_db']
        self._prev_ue_counts = {
            bs.bs_id: len([
                ue for ue in self.network.ues
                if ue.serving_bs_id == bs.bs_id])
            for bs in self.network.base_stations}

        terminated = (stats_after['ues_in_outage'] == 0)
        truncated  = (self._step_count >= self.max_steps)

        if terminated:
            reward += 10.0
        elif truncated:
            reward -= 5.0

        if self.verbose:
            print(f"   Step {self._step_count}: "
                  f"{self.ACTION_NAMES[action]} | "
                  f"Cov="
                  f"{stats_after['coverage_pct']:.1f}%"
                  f" | Out="
                  f"{stats_after['ues_in_outage']}"
                  f" | R={reward:.2f}")

        obs  = self._get_observation()
        info = self._get_info()
        info['action_name'] = self.ACTION_NAMES[action]
        info['stats_after'] = stats_after

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.network is None:
            return
        stats = self.network.get_stats()
        print(f"BS-{self.failed_bs_id} | "
              f"Step {self._step_count} | "
              f"Cov={stats['coverage_pct']:.1f}%")

    def close(self):
        self._restore()

    # ── Observation ───────────────────────────────────────────────────

    def _get_observation(self) -> np.ndarray:
        cfg = self.cfg
        obs = []
        for bs in self.network.base_stations:
            served = [ue for ue in self.network.ues
                      if ue.serving_bs_id == bs.bs_id]
            sinrs  = [ue.sinr_db for ue in served
                      if (ue.sinr_db is not None and
                          np.isfinite(ue.sinr_db))]
            ue_norm    = min(
                len(served) / max(self._max_ues, 1), 2.0)
            mean_sinr  = np.mean(sinrs) if sinrs else 0.0
            sinr_norm  = float(np.clip(
                (mean_sinr - self.SINR_OBS_MIN) /
                (self.SINR_OBS_MAX - self.SINR_OBS_MIN),
                0.0, 1.0))
            prev_ue    = self._prev_ue_counts.get(
                bs.bs_id, len(served))
            delta_norm = float(np.clip(
                (len(served) - prev_ue) /
                max(self._max_ues, 1), -1.0, 1.0))
            tilt_norm  = float(np.clip(
                (bs.tilt_deg - cfg.BS_TILT_MIN_DEG) /
                (cfg.BS_TILT_MAX_DEG - cfg.BS_TILT_MIN_DEG),
                0.0, 1.0))
            power_norm = (
                -1.0 if not bs.is_active
                else float(np.clip(
                    (bs.tx_power_dbm -
                     cfg.BS_TX_POWER_DBM) /
                    cfg.MAX_POWER_BOOST_CENTRE_DB,
                    0.0, 1.0)))
            obs.extend([ue_norm, sinr_norm, delta_norm,
                        tilt_norm, power_norm,
                        float(bs.is_active)])

        one_hot = [0.0] * cfg.NUM_BS
        if self.failed_bs_id is not None:
            one_hot[self.failed_bs_id] = 1.0
        obs.extend(one_hot)

        stats = self.network.get_stats()
        obs.append(stats['coverage_pct'] / 100.0)
        obs.append(stats['ues_in_outage'] / cfg.NUM_UE)
        obs.append(self._step_count /
                   max(self.max_steps, 1))

        return np.array(obs, dtype=np.float32)

    def _get_info(self) -> dict:
        stats = self.network.get_stats()
        return {
            'failed_bs_id': self.failed_bs_id,
            'step'        : self._step_count,
            'coverage_pct': stats['coverage_pct'],
            'ues_in_outage': stats['ues_in_outage'],
            'mean_sinr_db': stats['mean_sinr_db'],
            'ep_count'    : self._episode_count,
        }

    # ── Reward ────────────────────────────────────────────────────────

    def _compute_reward(self, stats_before: dict,
                         stats_after: dict,
                         ues_outage_before: dict) -> float:
        """
        UE-rescue based reward.
        +2.0 per rescued UE — primary learning signal.
        -1.5 per collateral UE — safety constraint.
        Maintenance reward when holding rescued state.
        """
        n_rescued = sum(
            1 for ue in self.network.ues
            if (ues_outage_before.get(ue.ue_id, False)
                and not ue.in_outage))
        n_coll = sum(
            1 for ue in self.network.ues
            if (not ues_outage_before.get(
                    ue.ue_id, True)
                and ue.in_outage))
        cov_delta = (
            (stats_after['coverage_pct'] -
             stats_before['coverage_pct']) /
            100 * 5.0)

        if n_rescued == 0 and n_coll == 0:
            above_baseline = (
                stats_after['coverage_pct'] / 100 -
                self._baseline_coverage)
            r_maintenance = above_baseline * 1.0
        else:
            r_maintenance = 0.0

        return float(n_rescued * 2.0 +
                     n_coll * (-1.5) +
                     cov_delta + r_maintenance)

    # ── Action execution ──────────────────────────────────────────────

    def _apply_action(self, action: int) -> None:
        """
        Maps action to RadioNetwork compensation strategy.
        Uses selective reset — only resets the parameter
        type being changed to allow multi-step strategies.
        """
        fid          = self.failed_bs_id
        power_actions = {1, 2, 3, 4}
        tilt_actions  = {5}
        joint_actions = {6}

        if action in power_actions:
            for bs in self.network.base_stations:
                if bs.is_active:
                    bs.tx_power_dbm = bs.nominal_power_dbm
        elif action in tilt_actions:
            for bs in self.network.base_stations:
                if bs.is_active:
                    bs.tilt_deg = bs.nominal_tilt_deg
        elif action in joint_actions:
            self.network.reset_all_powers()

        if   action == 0: pass
        elif action == 1:
            self.network.apply_power_compensation(fid)
        elif action == 2:
            self.network.apply_proportional_compensation(
                fid)
        elif action == 3:
            self.network.apply_targeted_compensation(fid)
        elif action == 4:
            self.network.apply_simultaneous_compensation(
                fid)
        elif action == 5:
            self.network.apply_tilt_compensation(fid)
        elif action == 6:
            self.network.apply_joint_compensation(fid)

    def get_action_name(self, action: int) -> str:
        return self.ACTION_NAMES.get(action, 'Unknown')
