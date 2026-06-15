# src/config.py
"""
Simulation Configuration
========================
Central configuration for the self-healing RAN simulation.
All parameters are defined here — change nothing elsewhere.
"""

import numpy as np


class SimConfig:
    """
    Radio network simulation configuration.

    Channel model : Log-distance path loss with 3GPP TR 36.942
                    vertical antenna pattern model.
    Deployment    : Urban macro-cell inspired parameters.
                    7-BS hexagonal layout, ISD=350m.
    """

    # ── Deployment geometry ──────────────────────────────────────────
    AREA_SIZE        = 1000        # metres
    NUM_BS           = 7           # 1 centre + 6 hex vertices
    HEX_RADIUS       = 350         # metres — ISD = 350m
    BS_HEIGHT_M      = 25.0        # metres
    UE_HEIGHT_M      = 1.5         # metres

    # ── UE placement ─────────────────────────────────────────────────
    NUM_UE           = 100
    UE_CLUSTER_STD   = 60          # metres — Gaussian cluster std

    # ── Carrier & bandwidth ──────────────────────────────────────────
    FREQUENCY_GHZ    = 2.1         # GHz
    BANDWIDTH_MHZ    = 10.0        # MHz

    # ── Transmit power ───────────────────────────────────────────────
    BS_TX_POWER_DBM  = 43.0        # dBm

    # ── Path loss — log-distance model ───────────────────────────────
    PATH_LOSS_EXPONENT = 3.5       # urban macro
    SHADOW_FADING_STD  = 0.0       # dB (0 = disabled)

    # ── Noise ────────────────────────────────────────────────────────
    NOISE_POWER_DBM  = -104.0      # dBm (10 MHz BW)

    # ── Antenna tilt — 3GPP TR 36.942 vertical pattern ───────────────
    BS_TILT_DEFAULT_DEG            = 15.0  # degrees nominal
    BS_TILT_MIN_DEG                = 6.0   # minimum (wider coverage)
    BS_TILT_MAX_DEG                = 25.0  # maximum (tightest)
    ANTENNA_VERTICAL_BEAMWIDTH_DEG = 15.0  # theta_3dB
    ANTENNA_SLA_DB                 = 20.0  # side-lobe attenuation
    TILT_REDUCTION_STEP_DEG        = 3.0   # degrees per step

    # ── Outage thresholds ─────────────────────────────────────────────
    SINR_OUTAGE_THRESHOLD_DB        = 0.0    # dB
    SINR_COMPENSATION_THRESHOLD_DB  = -4.0   # dB
    MIN_RX_POWER_DBM                = -105.0 # dBm

    # ── Throughput ────────────────────────────────────────────────────
    MIN_THROUGHPUT_MBPS = 0.5      # Mbps

    # ── Cell outage severity ──────────────────────────────────────────
    OUTAGE_POWER_REDUCTION_DB = 20.0  # dB for partial outage

    # ── Self-healing power caps ───────────────────────────────────────
    MAX_POWER_BOOST_DB         = 6.0   # dB — edge BS
    MAX_POWER_BOOST_CENTRE_DB  = 14.0  # dB — centre BS

    # ── KPI monitoring thresholds (COD — independent of outage def) ──
    KPI_LOW_SINR_THRESHOLD_DB     = -5.0    # dB
    KPI_WEAK_SIGNAL_THRESHOLD_DBM = -100.0  # dBm

    # ── RL environment ────────────────────────────────────────────────
    RL_CURRICULUM_EPISODES = 1500  # edge BSs only for first N eps
    MAX_STEPS_PER_EPISODE  = 10

    # ── Reproducibility ───────────────────────────────────────────────
    RANDOM_SEED = 42


# Global config instance
CFG = SimConfig()
