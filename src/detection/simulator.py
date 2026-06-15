# src/detection/simulator.py
"""
Episode Simulator
=================
Generates labelled KPI datasets for COD training.
Each episode: normal operation → random BS outage.
UEs move between timesteps (random walk).
"""

import numpy as np
import pandas as pd

from src.config import SimConfig, CFG
from src.network.radio_network import RadioNetwork
from src.detection.kpi_logger import KPILogger


def move_ues(network: RadioNetwork,
              step_size_m: float = 10.0,
              rng: np.random.RandomState = None) -> None:
    """
    Apply small random walk to all UEs.
    Models pedestrian mobility (~1 m/s at 10s intervals).
    """
    if rng is None:
        rng = np.random.RandomState()
    cfg = network.cfg
    for ue in network.ues:
        ue.x = np.clip(ue.x + rng.normal(0, step_size_m),
                        0, cfg.AREA_SIZE)
        ue.y = np.clip(ue.y + rng.normal(0, step_size_m),
                        0, cfg.AREA_SIZE)


def simulate_episodes(
        config          = None,
        n_episodes      : int   = 300,
        n_normal_steps  : int   = 15,
        n_outage_steps  : int   = 10,
        ue_step_size_m  : float = 10.0,
        failed_bs_ids           = None,
        verbose         : bool  = True
) -> tuple:
    """
    Generate labelled KPI dataset.

    Episode timeline:
      Phase 1 (n_normal_steps): Normal operation.
        All BSs labelled 0.
      Phase 2 (n_outage_steps): One BS fails.
        Failed BS = label 1. Degraded neighbours = label 2.

    Returns:
        df     : labelled KPI DataFrame
        logger : KPILogger instance
    """
    if config is None:
        config = CFG

    logger   = KPILogger(None)
    n_bs     = config.NUM_BS
    total_ts = n_normal_steps + n_outage_steps

    if verbose:
        print(f"🔄 Simulating {n_episodes} episodes...")
        print(f"   Normal steps  : {n_normal_steps}")
        print(f"   Outage steps  : {n_outage_steps}")
        print(f"   Est. snapshots: "
              f"~{n_episodes * total_ts * n_bs:,}")

    for ep in range(n_episodes):
        ep_seed            = config.RANDOM_SEED + ep * 100
        rng                = np.random.RandomState(ep_seed)
        ep_cfg             = SimConfig()
        ep_cfg.RANDOM_SEED = ep_seed
        network            = RadioNetwork(ep_cfg)
        logger.network     = network
        logger.reset()

        if failed_bs_ids is not None:
            fail_id = failed_bs_ids[ep %
                                     len(failed_bs_ids)]
        else:
            fail_id = int(rng.randint(0, n_bs))

        # Phase 1: Normal operation
        for t in range(n_normal_steps):
            if t > 0:
                move_ues(network, ue_step_size_m, rng)
            network.compute_association_and_sinr()
            logger.log(episode      = ep,
                       timestep     = t,
                       failed_bs_id = None)

        # Phase 2: Outage
        import sys, os
        sys.stdout = open(os.devnull, 'w')
        network.trigger_outage(fail_id, severity='full')
        sys.stdout = sys.__stdout__

        for t in range(n_normal_steps,
                       n_normal_steps + n_outage_steps):
            if t > n_normal_steps:
                move_ues(network, ue_step_size_m, rng)
            network.compute_association_and_sinr()
            logger.log(episode      = ep,
                       timestep     = t,
                       failed_bs_id = fail_id)

        if (verbose and
                (ep + 1) % 50 == 0):
            snaps = len(logger.snapshots)
            n_out = sum(1 for s in logger.snapshots
                        if s.label == 1)
            n_deg = sum(1 for s in logger.snapshots
                        if s.label == 2)
            print(f"   Ep {ep+1:4d}/{n_episodes}: "
                  f"{snaps:,} snapshots | "
                  f"outage={n_out:,} | "
                  f"degraded={n_deg:,}")

    df = logger.get_dataframe()

    if verbose:
        n_total = len(df)
        n_norm  = (df['label'] == 0).sum()
        n_out   = (df['label'] == 1).sum()
        n_deg   = (df['label'] == 2).sum()
        print(f"\n✅ Dataset generated: {n_total:,} rows")
        print(f"   Normal={n_norm:,} "
              f"({100*n_norm/n_total:.1f}%) | "
              f"Outage={n_out:,} "
              f"({100*n_out/n_total:.1f}%) | "
              f"Degraded={n_deg:,} "
              f"({100*n_deg/n_total:.1f}%)")

    return df, logger
