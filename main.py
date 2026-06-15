#!/usr/bin/env python3
"""
main.py — Self-Healing RAN: End-to-End Pipeline
================================================
Runs the complete self-healing pipeline in sequence:
  1. Network initialisation and normal operation
  2. Cell outage simulation and heuristic compensation
  3. Cell outage detection (COD) pipeline
  4. DQN and PPO RL agent training
  5. Evaluation: RL agents vs heuristics
  6. Publication-quality figures saved to docs/figures/

Usage:
    python main.py                    # full pipeline
    python main.py --skip-training    # skip RL training
    python main.py --quick            # fast demo mode
"""

import argparse
import os
import sys
import time
import copy
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize, ListedColormap
from scipy.spatial import Voronoi, voronoi_plot_2d
warnings.filterwarnings('ignore')

# ── Project imports ───────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from src.config import SimConfig, CFG
from src.network import (BaseStation,
                          UserEquipment,
                          RadioNetwork)
from src.detection.kpi_logger import KPILogger
from src.detection.simulator import (move_ues,
                                      simulate_episodes)
from src.detection.cod_classifier import (ThresholdCOD,
                                           MLCOD,
                                           COD_FEATURES,
                                           CLASS_NAMES)
from src.environment.gym_env import SelfHealingNetworkEnv
from src.agents.train import (train_dqn, train_ppo,
                               evaluate_heuristic,
                               evaluate_rl_agent,
                               TRAIN_CONFIG, PPO_CONFIG,
                               TrainingCallback,
                               PPOTrainingCallback)

# ── Directory setup ───────────────────────────────────────────────────
FIGURES_DIR = os.path.join('docs', 'figures')
MODELS_DIR  = 'models'
RESULTS_DIR = 'results'

for d in [FIGURES_DIR, MODELS_DIR, RESULTS_DIR,
          os.path.join(RESULTS_DIR, 'training_logs')]:
    os.makedirs(d, exist_ok=True)

# ── Argument parser ───────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description='Self-Healing RAN — End-to-End Pipeline')
parser.add_argument('--skip-training',
                    action='store_true',
                    help='Skip RL training (load models)')
parser.add_argument('--quick',
                    action='store_true',
                    help='Fast demo: fewer episodes/steps')
parser.add_argument('--failed-bs',
                    type=int, default=3,
                    help='BS ID to fail (0-6, default=3)')
args = parser.parse_args()

FAILED_BS_ID = args.failed_bs
QUICK_MODE   = args.quick

# Adjust counts for quick vs full run
N_COD_EPISODES  = 50  if QUICK_MODE else 300
N_EVAL_EPISODES = 50  if QUICK_MODE else 200
TS_TRAINING     = 2_000 if QUICK_MODE else 20_000

# ─────────────────────────────────────────────────────────────────────
# ── SECTION HEADER HELPER ─────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────

def section(title: str) -> None:
    width = 60
    print(f"\n{'═'*width}")
    print(f"  {title}")
    print(f"{'═'*width}")


def subsection(title: str) -> None:
    print(f"\n{'─'*50}")
    print(f"  {title}")
    print(f"{'─'*50}")


def save_fig(filename: str) -> None:
    path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  💾 Saved: {path}")


# ─────────────────────────────────────────────────────────────────────
# ══ SECTION 1: NETWORK SETUP AND NORMAL OPERATION ════════════════════
# ─────────────────────────────────────────────────────────────────────

section("1. NETWORK INITIALISATION AND NORMAL OPERATION")

print(f"\n  Configuration:")
print(f"  BSs          : {CFG.NUM_BS} "
      f"(hexagonal, ISD={CFG.HEX_RADIUS}m)")
print(f"  UEs          : {CFG.NUM_UE} "
      f"(clustered, σ={CFG.UE_CLUSTER_STD}m)")
print(f"  TX Power     : {CFG.BS_TX_POWER_DBM}dBm")
print(f"  Frequency    : {CFG.FREQUENCY_GHZ}GHz")
print(f"  PL Exponent  : {CFG.PATH_LOSS_EXPONENT}")
print(f"  Noise floor  : {CFG.NOISE_POWER_DBM}dBm")
print(f"  Antenna tilt : {CFG.BS_TILT_DEFAULT_DEG}° "
      f"(range {CFG.BS_TILT_MIN_DEG}°–"
      f"{CFG.BS_TILT_MAX_DEG}°)")

# Initialise network
net = RadioNetwork(CFG)
net.compute_association_and_sinr()
stats_normal = net.get_stats()

print(f"\n  Normal operation stats:")
print(f"  Mean SINR    : {stats_normal['mean_sinr_db']:.2f}dB")
print(f"  Coverage     : {stats_normal['coverage_pct']:.1f}%")
print(f"  Outage UEs   : {stats_normal['ues_in_outage']}")


# ── Plot utilities ────────────────────────────────────────────────────

def compute_sinr_grid(network, resolution=120):
    cfg          = network.cfg
    xs           = np.linspace(0, cfg.AREA_SIZE, resolution)
    ys           = np.linspace(0, cfg.AREA_SIZE, resolution)
    X, Y         = np.meshgrid(xs, ys)
    noise_linear = 10**(cfg.NOISE_POWER_DBM / 10)
    sinr_grid    = np.zeros_like(X)
    f            = cfg.FREQUENCY_GHZ * 1e9

    for i in range(resolution):
        for j in range(resolution):
            rx_dbms = []
            rx_lins = []
            for bs in network.base_stations:
                if not bs.is_active:
                    rx_dbms.append(-np.inf)
                    rx_lins.append(0.0)
                    continue
                d   = max(np.sqrt((X[i,j]-bs.x)**2 +
                                   (Y[i,j]-bs.y)**2), 1.0)
                pl  = (20*np.log10(4*np.pi/3e8) +
                       20*np.log10(f) +
                       10*cfg.PATH_LOSS_EXPONENT*
                       np.log10(d))
                ag  = network._antenna_gain_db(
                    d, bs.tilt_deg)
                rx  = bs.tx_power_dbm - pl + ag
                rx_dbms.append(rx)
                rx_lins.append(10**(rx/10)
                               if np.isfinite(rx)
                               else 0.0)

            active_rx = [p for p in rx_dbms
                         if np.isfinite(p)]
            best_pwr  = (max(active_rx)
                         if active_rx else -np.inf)
            if best_pwr < cfg.MIN_RX_POWER_DBM:
                sinr_grid[i, j] = -20.0
                continue

            best_idx     = int(np.argmax(rx_lins))
            signal       = rx_lins[best_idx]
            interference = sum(rx_lins) - signal
            sinr         = signal / (interference +
                                      noise_linear)
            sinr_grid[i, j] = 10*np.log10(
                max(sinr, 1e-15))
    return X, Y, sinr_grid


def _style_ax(ax, cfg):
    ax.set_xlim(0, cfg.AREA_SIZE)
    ax.set_ylim(0, cfg.AREA_SIZE)
    ax.set_aspect('equal')
    ax.set_facecolor('white')
    ax.grid(True, color='grey', linestyle='--',
            linewidth=0.4, alpha=0.6, zorder=0)
    ax.set_xlabel("x (m)", fontsize=9)
    ax.set_ylabel("y (m)", fontsize=9)
    ax.tick_params(labelsize=8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def _draw_bs(ax, network, fontsize=7):
    for bs in network.base_stations:
        marker = '^' if bs.is_active else 'X'
        edge   = 'black' if bs.is_active else 'red'
        face   = bs.color if bs.is_active else 'red'
        ax.scatter(bs.x, bs.y, color=face, s=140,
                   marker=marker, edgecolors=edge,
                   linewidths=1.3, zorder=6)
        ax.annotate(
            f"BS{bs.bs_id}\n"
            f"{bs.tx_power_dbm:.0f}dBm",
            (bs.x, bs.y),
            textcoords="offset points",
            xytext=(6, 6), fontsize=fontsize,
            color='black' if bs.is_active else 'red',
            fontweight='bold')


def plot_network_state(network, title,
                        filename,
                        grid_resolution=120):
    """Three-panel network state plot."""
    cfg = network.cfg
    print(f"  Computing SINR grid...", end="", flush=True)
    X, Y, sinr_grid = compute_sinr_grid(
        network, grid_resolution)
    cov_grid = (
        sinr_grid >= cfg.SINR_OUTAGE_THRESHOLD_DB
    ).astype(float)
    print(" done.")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(title, fontsize=13,
                 fontweight='bold', y=1.01)

    # Panel 1: Association
    ax1 = axes[0]
    _style_ax(ax1, cfg)
    ax1.set_title("UE Association",
                  fontsize=10, fontweight='bold')
    bs_pts  = np.array([[bs.x, bs.y]
                         for bs in network.base_stations])
    pad_pts = []
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            for bp in bs_pts:
                pad_pts.append(
                    bp + np.array([sx*4*cfg.AREA_SIZE,
                                    sy*4*cfg.AREA_SIZE]))
    try:
        vor = Voronoi(np.vstack([bs_pts, pad_pts]))
        voronoi_plot_2d(vor, ax=ax1,
                        show_vertices=False,
                        line_colors='#888888',
                        line_width=1.0,
                        line_alpha=0.5, point_size=0)
    except Exception:
        pass
    ax1.set_xlim(0, cfg.AREA_SIZE)
    ax1.set_ylim(0, cfg.AREA_SIZE)
    for ue in network.ues:
        if ue.serving_bs_id is not None:
            c = network.base_stations[
                ue.serving_bs_id].color
            ax1.scatter(ue.x, ue.y, color=c, s=18,
                        marker='o', alpha=0.85,
                        edgecolors='none', zorder=4)
        else:
            ax1.scatter(ue.x, ue.y, color='red',
                        s=30, marker='x',
                        linewidths=1.4, zorder=4)
    _draw_bs(ax1, network)
    handles = [mpatches.Patch(
        color=bs.color,
        label=(f"BS-{bs.bs_id}" +
               (" ✗" if not bs.is_active else "")))
               for bs in network.base_stations]
    ax1.legend(handles=handles, loc='lower left',
               fontsize=6, framealpha=0.85)

    # Panel 2: SINR heatmap
    ax2 = axes[1]
    _style_ax(ax2, cfg)
    ax2.set_title("SINR Map (dB)",
                  fontsize=10, fontweight='bold')
    vmin, vmax = -10, 30
    im = ax2.imshow(sinr_grid, origin='lower',
                    extent=[0, cfg.AREA_SIZE,
                             0, cfg.AREA_SIZE],
                    cmap='RdYlGn', vmin=vmin,
                    vmax=vmax, aspect='equal',
                    interpolation='bilinear', zorder=1)
    plt.colorbar(im, ax=ax2, shrink=0.82, pad=0.02,
                 label="SINR (dB)")
    sinr_vals = np.array([
        ue.sinr_db if (ue.sinr_db is not None and
                       np.isfinite(ue.sinr_db))
        else -20 for ue in network.ues])
    norm = Normalize(vmin=vmin, vmax=vmax)
    ax2.scatter([ue.x for ue in network.ues],
                [ue.y for ue in network.ues],
                c=sinr_vals, cmap='RdYlGn', norm=norm,
                s=20, edgecolors='black',
                linewidths=0.3, zorder=5, alpha=0.9)
    _draw_bs(ax2, network)
    n_out = sum(1 for ue in network.ues
                if ue.in_outage)
    ax2.text(0.02, 0.98,
             f"Outage: {n_out}/{len(network.ues)}\n"
             f"Thr: {cfg.SINR_OUTAGE_THRESHOLD_DB}dB",
             transform=ax2.transAxes, fontsize=7,
             verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.3',
                       facecolor='white',
                       edgecolor='grey', alpha=0.85))

    # Panel 3: Coverage
    ax3 = axes[2]
    _style_ax(ax3, cfg)
    ax3.set_title("Coverage Map",
                  fontsize=10, fontweight='bold')
    cov_cmap = ListedColormap(['#f28b82', '#b7e1a1'])
    ax3.imshow(cov_grid, origin='lower',
               extent=[0, cfg.AREA_SIZE, 0, cfg.AREA_SIZE],
               cmap=cov_cmap, vmin=0, vmax=1,
               aspect='equal', interpolation='nearest',
               zorder=1, alpha=0.75)
    covered = [ue for ue in network.ues
               if not ue.in_outage]
    outage  = [ue for ue in network.ues if ue.in_outage]
    if covered:
        ax3.scatter([u.x for u in covered],
                    [u.y for u in covered],
                    color='#1a7f37', s=18, marker='o',
                    edgecolors='none', alpha=0.85,
                    zorder=5, label='Covered')
    if outage:
        ax3.scatter([u.x for u in outage],
                    [u.y for u in outage],
                    color='#c0392b', s=40, marker='x',
                    linewidths=1.6, zorder=6,
                    label='Outage')
    _draw_bs(ax3, network)
    pct_area = 100 * cov_grid.mean()
    pct_ues  = 100 * (len(covered) /
                       len(network.ues))
    ax3.text(0.02, 0.98,
             f"Area: {pct_area:.1f}%\n"
             f"UEs : {pct_ues:.1f}%",
             transform=ax3.transAxes, fontsize=7,
             verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.3',
                       facecolor='white',
                       edgecolor='grey', alpha=0.85))
    patches = [
        mpatches.Patch(color='#b7e1a1',
                        label='Covered region'),
        mpatches.Patch(color='#f28b82',
                        label='Outage region'),
        mpatches.Patch(color='#1a7f37',
                        label='Covered UE'),
        mpatches.Patch(color='#c0392b',
                        label='Outage UE'),
    ]
    ax3.legend(handles=patches, loc='lower left',
               fontsize=6, framealpha=0.85)

    plt.tight_layout()
    save_fig(filename)


# Plot normal operation
subsection("1.1 Normal Operation")
plot_network_state(
    net,
    "Scenario 1 — Normal Operation (No Outage)",
    "01_normal_operation.png")
print(f"  Coverage: {stats_normal['coverage_pct']:.1f}% "
      f"| Outage UEs: {stats_normal['ues_in_outage']}")


# ─────────────────────────────────────────────────────────────────────
# ══ SECTION 2: CELL OUTAGE AND COMPENSATION STRATEGIES ═══════════════
# ─────────────────────────────────────────────────────────────────────

section("2. CELL OUTAGE AND COMPENSATION STRATEGIES")

subsection(f"2.1 BS-{FAILED_BS_ID} Complete Outage")

net_outage = copy.deepcopy(net)
net_outage.trigger_outage(FAILED_BS_ID, severity='full')
net_outage.compute_association_and_sinr()
stats_outage = net_outage.get_stats()

print(f"\n  Post-outage stats:")
print(f"  Coverage   : {stats_outage['coverage_pct']:.1f}%")
print(f"  Outage UEs : {stats_outage['ues_in_outage']}")

plot_network_state(
    net_outage,
    f"Scenario 2 — BS-{FAILED_BS_ID} Outage (No Healing)",
    "02_outage_no_healing.png")

# Run all 6 strategies
strategies = {
    'S1_Fixed'       : ('apply_power_compensation',    {}),
    'S2_Proportional': ('apply_proportional_compensation', {}),
    'S3_BestNbr'     : ('apply_targeted_compensation', {}),
    'S4_Simultaneous': ('apply_simultaneous_compensation', {}),
    'S5_TiltOnly'    : ('apply_tilt_compensation',     {}),
    'S6_Joint'       : ('apply_joint_compensation',    {}),
}

strategy_results = {'No Healing': stats_outage}
strategy_networks = {}

for s_key, (method_name, kwargs) in strategies.items():
    subsection(f"2.2 Strategy: {s_key}")
    net_s = copy.deepcopy(net_outage)
    method = getattr(net_s, method_name)
    method(FAILED_BS_ID, **kwargs)
    net_s.compute_association_and_sinr()
    stats_s = net_s.get_stats()
    strategy_results[s_key] = stats_s
    strategy_networks[s_key] = net_s

    label = s_key.replace('_', ' ')
    plot_network_state(
        net_s,
        f"Scenario — {label}",
        f"03_strategy_{s_key.lower()}.png")

    print(f"  Coverage: {stats_s['coverage_pct']:.1f}% "
          f"| Outage UEs: {stats_s['ues_in_outage']}")

# Strategy comparison table
subsection("2.3 Strategy Comparison Summary")

rows = []
for name, stats in strategy_results.items():
    rows.append({
        'Strategy'       : name,
        'Coverage (%)'   : f"{stats['coverage_pct']:.1f}",
        'Outage UEs'     : stats['ues_in_outage'],
        'Mean SINR (dB)' : f"{stats['mean_sinr_db']:.2f}",
        '5th Pct SINR'   : f"{stats['5th_pct_sinr_db']:.2f}",
        'Mean Tput (Mbps)': f"{stats['mean_throughput']:.2f}",
    })

df_strategies = pd.DataFrame(rows).set_index('Strategy')
print(f"\n{df_strategies.to_string()}")
df_strategies.to_csv(
    os.path.join(RESULTS_DIR, 'strategy_comparison.csv'))

# SINR CDF comparison
subsection("2.4 SINR CDF — Strategy Comparison")

fig, ax = plt.subplots(figsize=(9, 5))
styles  = ['-', '--', '-.', ':', '-', '--', '-.']
colors  = ['#e74c3c', '#95a5a6', '#3498db', '#2ecc71',
           '#9b59b6', '#f39c12', '#1abc9c']
labels  = ['No Healing'] + list(strategies.keys())
networks_list = [net_outage] + list(
    strategy_networks.values())

for i, (label, network) in enumerate(
        zip(labels, networks_list)):
    vals = sorted([
        ue.sinr_db for ue in network.ues
        if (ue.sinr_db is not None and
            np.isfinite(ue.sinr_db))])
    if not vals:
        continue
    cdf = np.arange(1, len(vals)+1) / len(vals)
    ax.plot(vals, cdf, lw=2,
            linestyle=styles[i % len(styles)],
            color=colors[i % len(colors)],
            label=label.replace('_', ' '))

ax.axvline(CFG.SINR_OUTAGE_THRESHOLD_DB,
           color='red', linestyle=':', lw=1.5,
           label='Outage threshold')
ax.set_xlabel("SINR (dB)", fontsize=11)
ax.set_ylabel("CDF", fontsize=11)
ax.set_title("SINR CDF — Heuristic Strategy Comparison",
             fontsize=12, fontweight='bold')
ax.legend(fontsize=8, loc='upper left')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_facecolor('white')
ax.set_xlim(-10, 35)
plt.tight_layout()
save_fig("04_sinr_cdf_heuristics.png")


# ─────────────────────────────────────────────────────────────────────
# ══ SECTION 3: CELL OUTAGE DETECTION ═════════════════════════════════
# ─────────────────────────────────────────────────────────────────────

section("3. CELL OUTAGE DETECTION (COD)")

subsection("3.1 Generating KPI Dataset")

df_all, kpi_logger = simulate_episodes(
    config         = CFG,
    n_episodes     = N_COD_EPISODES,
    n_normal_steps = 15,
    n_outage_steps = 10,
    ue_step_size_m = 10.0,
    verbose        = True)

# Train/test split by episode
episodes    = df_all['episode'].unique()
n_train_ep  = int(0.8 * len(episodes))
train_eps   = episodes[:n_train_ep]
test_eps    = episodes[n_train_ep:]
df_train    = df_all[df_all['episode'].isin(
    train_eps)].copy()
df_test     = df_all[df_all['episode'].isin(
    test_eps)].copy()

print(f"\n  Train: {len(train_eps)} episodes "
      f"({len(df_train):,} rows)")
print(f"  Test : {len(test_eps)} episodes "
      f"({len(df_test):,} rows)")

subsection("3.2 KPI Evolution Plot")

ep_df       = df_all[df_all['episode'] == 0].copy()
failed_rows = ep_df[ep_df['label'] == 1]

if not failed_rows.empty:
    failed_bs = int(failed_rows['bs_id'].iloc[0])
    ref_row   = ep_df[ep_df['bs_id'] == failed_bs].iloc[0]
    nb_ids    = [int(ref_row.get(f'n{k}_bs_id', -1))
                 for k in range(1, 4)
                 if f'n{k}_bs_id' in ref_row.index]
    nb_ids    = [x for x in nb_ids if x >= 0]
    timesteps = sorted(ep_df['timestep'].unique())

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(
        f"KPI Evolution — Episode 0 | "
        f"BS-{failed_bs} fails at t=15",
        fontsize=12, fontweight='bold')

    kpis = [
        ('ue_count',      'UE Count',         'tab:blue'),
        ('prb_load',      'PRB Load',          'tab:green'),
        ('ue_ratio',      'UE Ratio',          'tab:orange'),
        ('delta_ue_count','Δ UE Count',        'tab:red'),
        ('delta_prb_load','Δ PRB Load',        'tab:purple'),
        ('n1_ue_count',   'Neighbour-1 UE Count',
         'tab:brown'),
    ]

    for ax, (kpi, ylabel, color) in zip(
            axes.flat, kpis):
        fd = ep_df[
            ep_df['bs_id'] == failed_bs
        ].sort_values('timestep')
        if kpi in fd.columns:
            ax.plot(fd['timestep'], fd[kpi],
                    color=color, lw=2.5,
                    marker='o', markersize=4,
                    label=f'BS-{failed_bs} (FAILED)')
        for nb_id, ls in zip(nb_ids[:2], ['--', ':']):
            nd = ep_df[ep_df['bs_id'] == nb_id
                       ].sort_values('timestep')
            if kpi in nd.columns:
                ax.plot(nd['timestep'], nd[kpi],
                        lw=1.5, alpha=0.7,
                        linestyle=ls,
                        marker='s', markersize=3,
                        label=f'BS-{nb_id} (nb)')
        ax.axvline(x=14.5, color='red',
                   linestyle=':', lw=2, alpha=0.9)
        ax.axvspan(14.5, max(timesteps)+0.5,
                   alpha=0.07, color='red')
        ax.set_xlabel("Timestep", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(ylabel, fontsize=9,
                     fontweight='bold')
        ax.legend(fontsize=6, loc='best')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor('white')
        ax.set_xticks(timesteps[::3])

    plt.tight_layout()
    save_fig("05_cod_kpi_evolution.png")

subsection("3.3 Threshold COD")

thresh_cod   = ThresholdCOD()
thresh_preds = thresh_cod.evaluate(df_test)

subsection("3.4 Random Forest COD")

rf_cod = MLCOD(model_type='random_forest')
rf_cod.fit(df_train)
rf_preds, rf_proba = rf_cod.evaluate(
    df_test, label='Test set')

subsection("3.5 Logistic Regression COD")

lr_cod = MLCOD(model_type='logistic_regression')
lr_cod.fit(df_train)
lr_preds, lr_proba = lr_cod.evaluate(
    df_test, label='Test set')

# Save RF model
import pickle
rf_path = os.path.join(MODELS_DIR,
                        'cod_random_forest.pkl')
with open(rf_path, 'wb') as f:
    pickle.dump(rf_cod, f)
print(f"\n  💾 Saved COD model: {rf_path}")

subsection("3.6 COD Feature Importance Plot")

if hasattr(rf_cod.model, 'feature_importances_'):
    imps    = rf_cod.model.feature_importances_
    feats   = rf_cod.features_
    fi      = sorted(zip(feats, imps),
                     key=lambda x: x[1],
                     reverse=True)[:15]
    names, values = zip(*fi)

    fig, ax = plt.subplots(figsize=(9, 6))
    colors  = ['#2980b9' if 'delta' in n
               else '#27ae60' if any(
                   f'n{k}' in n for k in range(1,4))
               else '#e74c3c' for n in names]
    ax.barh(range(len(names)), values[::-1],
            color=colors[::-1],
            edgecolor='black', linewidth=0.6)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names[::-1], fontsize=9)
    ax.set_xlabel("Feature Importance", fontsize=10)
    ax.set_title(
        "Random Forest COD — Feature Importance\n"
        "(blue=delta, green=neighbour, red=own cell)",
        fontsize=11, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3,
            linestyle='--')
    ax.set_facecolor('white')
    plt.tight_layout()
    save_fig("06_cod_feature_importance.png")

subsection("3.7 COD Detection Probability")

df_plot = df_test.copy().reset_index(drop=True)
df_plot['rf_outage']   = rf_proba[:, 1]
df_plot['rf_degraded'] = rf_proba[:, 2]
df_plot['lr_outage']   = lr_proba[:, 1]
df_plot['thresh_pred'] = thresh_preds.values

ts  = sorted(df_plot['timestep'].unique())
grp = df_plot.groupby('timestep')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("COD Detection Probability Over Time",
             fontsize=12, fontweight='bold')

def ts_mean(col):
    return [grp.get_group(t)[col].mean() for t in ts]

true_out = [(grp.get_group(t)['label'] == 1).mean()
             for t in ts]
true_deg = [(grp.get_group(t)['label'] == 2).mean()
             for t in ts]

ax1.fill_between(ts, true_out, alpha=0.12,
                 color='red',
                 label='True outage fraction')
ax1.plot(ts, ts_mean('rf_outage'),
         color='#2980b9', lw=2.5,
         marker='o', markersize=4,
         label='Random Forest')
ax1.plot(ts, ts_mean('lr_outage'),
         color='#27ae60', lw=2,
         linestyle='--', marker='s', markersize=3,
         label='Logistic Regression')
thresh_out = [(grp.get_group(t)['thresh_pred'] == 1
               ).mean() for t in ts]
ax1.plot(ts, thresh_out, color='#e74c3c', lw=2,
         linestyle='-.', marker='^', markersize=3,
         label='Threshold Rule')
ax1.axvline(x=14.5, color='red', linestyle=':',
            lw=2, label='Outage onset')
ax1.set_title("Outage Detection (Label=1)",
              fontsize=10, fontweight='bold')
ax1.set_xlabel("Timestep", fontsize=9)
ax1.set_ylabel("Detection Probability", fontsize=9)
ax1.set_ylim(-0.05, 1.1)
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_facecolor('white')

ax2.fill_between(ts, true_deg, alpha=0.12,
                 color='orange',
                 label='True degraded fraction')
ax2.plot(ts, ts_mean('rf_degraded'),
         color='#2980b9', lw=2.5,
         marker='o', markersize=4,
         label='Random Forest')
thresh_deg = [(grp.get_group(t)['thresh_pred'] == 2
               ).mean() for t in ts]
ax2.plot(ts, thresh_deg, color='#e74c3c', lw=2,
         linestyle='-.', marker='^', markersize=3,
         label='Threshold Rule')
ax2.axvline(x=14.5, color='red', linestyle=':',
            lw=2, label='Outage onset')
ax2.set_title("Degradation Detection (Label=2)",
              fontsize=10, fontweight='bold')
ax2.set_xlabel("Timestep", fontsize=9)
ax2.set_ylabel("Detection Probability", fontsize=9)
ax2.set_ylim(-0.05, 1.1)
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_facecolor('white')

plt.tight_layout()
save_fig("07_cod_detection_probability.png")


# ─────────────────────────────────────────────────────────────────────
# ══ SECTION 4: DRL AGENT TRAINING ════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────

section("4. DRL AGENT TRAINING (DQN + PPO)")

# Override training config for quick/full mode
TRAIN_CONFIG['total_timesteps'] = TS_TRAINING
PPO_CONFIG['total_timesteps']   = TS_TRAINING
TRAIN_CONFIG['model_dir']       = MODELS_DIR
TRAIN_CONFIG['log_dir']         = os.path.join(
    RESULTS_DIR, 'training_logs')

if args.skip_training:
    subsection("4.0 Loading Pre-trained Models")
    from stable_baselines3 import DQN, PPO
    dqn_path = os.path.join(MODELS_DIR,
                             'dqn_self_healing')
    ppo_path = os.path.join(MODELS_DIR,
                             'ppo_self_healing')
    dqn_model    = DQN.load(dqn_path)
    ppo_model    = PPO.load(ppo_path)
    dqn_callback = None
    ppo_callback = None
    print(f"  Loaded: {dqn_path}.zip")
    print(f"  Loaded: {ppo_path}.zip")
else:
    subsection("4.1 DQN Training")
    dqn_model, dqn_callback = train_dqn(
        config       = TRAIN_CONFIG,
        log_interval = 200)

    subsection("4.2 PPO Training")
    ppo_model, ppo_callback = train_ppo(
        config       = PPO_CONFIG,
        log_interval = 200)

# Training curves
if (not args.skip_training and
        dqn_callback is not None and
        ppo_callback is not None):
    subsection("4.3 Training Curves")

    dqn_sum = dqn_callback.get_summary()
    ppo_sum = ppo_callback.get_summary()

    def smooth(data, w=50):
        if len(data) < w:
            return data
        return np.convolve(data,
                           np.ones(w)/w,
                           mode='valid')

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        "DQN vs PPO Training Curves",
        fontsize=13, fontweight='bold')

    dqn_r   = smooth(dqn_sum['episode_rewards'])
    ppo_r   = smooth(ppo_sum['episode_rewards'])
    dqn_cov = smooth(dqn_sum['episode_coverages'])
    ppo_cov = smooth(ppo_sum['episode_coverages'])
    dqn_sol = smooth([x*100 for x in
                       dqn_sum['episode_solved']])
    ppo_sol = smooth([x*100 for x in
                       ppo_sum['episode_solved']])

    # Panel 1: Reward
    ax = axes[0, 0]
    ax.plot(dqn_r, color='#2980b9', lw=1.5,
            label='DQN')
    ax.plot(ppo_r, color='#e74c3c', lw=1.5,
            linestyle='--', label='PPO')
    ax.set_title("Episode Reward (smoothed)",
                 fontsize=10, fontweight='bold')
    ax.set_xlabel("Episode"); ax.set_ylabel("Reward")
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_facecolor('white')

    # Panel 2: Coverage
    ax = axes[0, 1]
    ax.plot(dqn_cov, color='#2980b9', lw=1.5,
            label='DQN')
    ax.plot(ppo_cov, color='#e74c3c', lw=1.5,
            linestyle='--', label='PPO')
    ax.axhline(y=100, color='green', lw=0.8,
               linestyle=':', label='100% target')
    ax.set_title("Coverage % (smoothed)",
                 fontsize=10, fontweight='bold')
    ax.set_xlabel("Episode"); ax.set_ylabel("Coverage %")
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_ylim(70, 105)
    ax.set_facecolor('white')

    # Panel 3: Solve rate
    ax = axes[1, 0]
    ax.plot(dqn_sol, color='#2980b9', lw=1.5,
            label='DQN')
    ax.plot(ppo_sol, color='#e74c3c', lw=1.5,
            linestyle='--', label='PPO')
    ax.set_title("Solve Rate % (smoothed)",
                 fontsize=10, fontweight='bold')
    ax.set_xlabel("Episode"); ax.set_ylabel("Solve %")
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 105)
    ax.set_facecolor('white')

    # Panel 4: Action distribution (DQN)
    ax   = axes[1, 1]
    cnt  = dqn_sum['action_counts']
    tot  = cnt.sum()
    pct  = (cnt / max(tot, 1)) * 100
    cols = ['#95a5a6', '#3498db', '#2ecc71',
            '#e74c3c', '#9b59b6', '#f39c12',
            '#1abc9c']
    bars = ax.bar(range(len(cnt)), pct, color=cols,
                  edgecolor='black', linewidth=0.7)
    ax.set_xticks(range(len(cnt)))
    ax.set_xticklabels(
        [SelfHealingNetworkEnv.ACTION_NAMES[i]
         .split(':')[-1].strip()[:8]
         for i in range(len(cnt))],
        fontsize=7, rotation=20, ha='right')
    ax.set_title("DQN Action Distribution",
                 fontsize=10, fontweight='bold')
    ax.set_ylabel("Usage %")
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_facecolor('white')
    for bar, p in zip(bars, pct):
        if p > 2:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.3,
                    f'{p:.0f}%', ha='center',
                    va='bottom', fontsize=7)

    plt.tight_layout()
    save_fig("08_training_curves.png")


# ─────────────────────────────────────────────────────────────────────
# ══ SECTION 5: EVALUATION — RL AGENTS VS HEURISTICS ══════════════════
# ─────────────────────────────────────────────────────────────────────

section("5. EVALUATION: RL AGENTS VS HEURISTICS")

subsection("5.1 Running Evaluations")

heuristic_action_map = {
    'No Healing'     : 0,
    'S1: Fixed'      : 1,
    'S2: Proportional': 2,
    'S3: Best Nbr'   : 3,
    'S4: Simultaneous': 4,
    'S5: Tilt Only'  : 5,
    'S6: Joint'      : 6,
}

all_results = {}
for name, action_id in heuristic_action_map.items():
    print(f"   Evaluating {name}...", end=" ")
    all_results[name] = evaluate_heuristic(
        name, action_id,
        n_episodes=N_EVAL_EPISODES)
    cov = np.mean(all_results[name]['coverage_pct'])
    sol = np.mean(all_results[name]['solved']) * 100
    print(f"Cov={cov:.1f}% | Solved={sol:.1f}%")

print(f"   Evaluating DQN agent...", end=" ")
all_results['DQN Agent'] = evaluate_rl_agent(
    dqn_model, n_episodes=N_EVAL_EPISODES)
cov = np.mean(all_results['DQN Agent']['coverage_pct'])
sol = np.mean(all_results['DQN Agent']['solved']) * 100
print(f"Cov={cov:.1f}% | Solved={sol:.1f}%")

print(f"   Evaluating PPO agent...", end=" ")
all_results['PPO Agent'] = evaluate_rl_agent(
    ppo_model, n_episodes=N_EVAL_EPISODES)
cov = np.mean(all_results['PPO Agent']['coverage_pct'])
sol = np.mean(all_results['PPO Agent']['solved']) * 100
print(f"Cov={cov:.1f}% | Solved={sol:.1f}%")

subsection("5.2 Results Summary Table")

rows = []
for name, res in all_results.items():
    rows.append({
        'Strategy'        : name,
        'Mean Cov (%)'    : f"{np.mean(res['coverage_pct']):.1f}",
        '5th Pct Cov (%)' : f"{np.percentile(res['coverage_pct'], 5):.1f}",
        'Solve Rate (%)'  : f"{np.mean(res['solved'])*100:.1f}",
        'Mean SINR (dB)'  : f"{np.mean(res['mean_sinr_db']):.2f}",
        'Outage UEs'      : f"{np.mean(res['ues_in_outage']):.2f}",
        'Mean Steps'      : f"{np.mean(res['steps_to_solve']):.1f}",
        'Power Boost (dB)': f"{np.mean(res['total_power_boost']):.2f}",
    })

df_eval = pd.DataFrame(rows).set_index('Strategy')
print(f"\n{df_eval.to_string()}")
df_eval.to_csv(
    os.path.join(RESULTS_DIR, 'evaluation_results.csv'))
print(f"\n  💾 Saved: results/evaluation_results.csv")

subsection("5.3 Coverage Comparison Bar Chart")

names  = list(all_results.keys())
means  = [np.mean(r['coverage_pct'])
           for r in all_results.values()]
p5s    = [np.percentile(r['coverage_pct'], 5)
           for r in all_results.values()]
p95s   = [np.percentile(r['coverage_pct'], 95)
           for r in all_results.values()]
err_lo = [m - p for m, p in zip(means, p5s)]
err_hi = [p - m for m, p in zip(means, p95s)]
colors = ['#e74c3c' if ('DQN' in n or 'PPO' in n)
           else '#95a5a6' if n == 'No Healing'
           else '#3498db' for n in names]

fig, ax = plt.subplots(figsize=(12, 5))
x    = np.arange(len(names))
bars = ax.bar(x, means, color=colors,
              edgecolor='black', linewidth=0.7,
              yerr=[err_lo, err_hi], capsize=4,
              error_kw={'elinewidth': 1.2,
                         'ecolor': 'black'})
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=20,
                    ha='right', fontsize=9)
ax.set_ylabel("Mean Coverage (%)", fontsize=11)
ax.set_title(
    "Coverage: RL Agents vs Heuristic Strategies\n"
    f"(n={N_EVAL_EPISODES} episodes, "
    f"error bars=5th–95th pct)",
    fontsize=11, fontweight='bold')
ax.set_ylim(70, 108)
ax.axhline(y=100, color='green', linestyle='--',
           lw=1, label='100% target')
ax.grid(True, axis='y', alpha=0.3, linestyle='--')
ax.set_facecolor('white')
for bar, mean in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.3,
            f'{mean:.1f}%', ha='center',
            va='bottom', fontsize=8,
            fontweight='bold')
legend_els = [
    mpatches.Patch(color='#e74c3c', label='RL Agent'),
    mpatches.Patch(color='#3498db', label='Heuristic'),
    mpatches.Patch(color='#95a5a6', label='No Healing'),
]
ax.legend(handles=legend_els, fontsize=9,
          loc='lower right')
for spine in ax.spines.values():
    spine.set_linewidth(0.8)
plt.tight_layout()
save_fig("09_coverage_comparison.png")

subsection("5.4 Per-BS Heatmap")

selected_names = ['No Healing', 'S3: Best Nbr',
                   'S5: Tilt Only', 'S6: Joint',
                   'DQN Agent', 'PPO Agent']
selected = {n: all_results[n] for n in selected_names
            if n in all_results}

bs_ids = list(range(CFG.NUM_BS))
matrix = np.zeros((len(selected), len(bs_ids)))

for si, (name, res) in enumerate(selected.items()):
    for bi, bs_id in enumerate(bs_ids):
        mask = res['failed_bs_id'] == bs_id
        if mask.sum() > 0:
            matrix[si, bi] = np.mean(
                res['coverage_pct'][mask])

fig, ax = plt.subplots(figsize=(10, 5))
im = ax.imshow(matrix, cmap='RdYlGn',
               vmin=70, vmax=100, aspect='auto')
plt.colorbar(im, ax=ax,
             label='Mean Coverage (%)', shrink=0.8)
ax.set_xticks(range(len(bs_ids)))
ax.set_xticklabels(
    [f'BS-{i}\n({"centre" if i==0 else "edge"})'
     for i in bs_ids], fontsize=9)
ax.set_yticks(range(len(selected)))
ax.set_yticklabels(list(selected.keys()), fontsize=9)
ax.set_title(
    "Coverage Heatmap: Strategy × Failed BS",
    fontsize=11, fontweight='bold')
for i in range(len(selected)):
    for j in range(len(bs_ids)):
        ax.text(j, i, f'{matrix[i,j]:.0f}%',
                ha='center', va='center',
                fontsize=8, fontweight='bold',
                color='black')
plt.tight_layout()
save_fig("10_per_bs_heatmap.png")

subsection("5.5 SINR CDF Comparison")

selected_cdf = ['No Healing', 'S3: Best Nbr',
                 'S6: Joint', 'DQN Agent', 'PPO Agent']
fig, ax = plt.subplots(figsize=(8, 5))
styles  = ['-', '--', '-.', '-', '--']
colors  = ['#95a5a6', '#3498db', '#2ecc71',
           '#e74c3c', '#c0392b']
widths  = [1.5, 1.5, 1.5, 2.5, 2.5]

for (name, ls, col, lw) in zip(
        selected_cdf, styles, colors, widths):
    if name not in all_results:
        continue
    vals = sorted(all_results[name]['mean_sinr_db'])
    cdf  = np.arange(1, len(vals)+1) / len(vals)
    ax.plot(vals, cdf, linestyle=ls, color=col,
            lw=lw, label=name)

ax.axvline(CFG.SINR_OUTAGE_THRESHOLD_DB,
           color='red', linestyle=':', lw=1.5,
           label='Outage threshold')
ax.set_xlabel("Mean SINR (dB)", fontsize=11)
ax.set_ylabel("CDF", fontsize=11)
ax.set_title("SINR CDF: RL Agents vs Heuristics",
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9, loc='upper left')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_facecolor('white')
ax.set_xlim(-5, 35)
plt.tight_layout()
save_fig("11_sinr_cdf_comparison.png")

subsection("5.6 DQN Action Preference Analysis")

dqn_res = all_results['DQN Agent']

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("DQN Action Preferences by BS Type",
             fontsize=12, fontweight='bold')

act_names = [
    SelfHealingNetworkEnv.ACTION_NAMES[i]
    .split(':')[-1].strip()
    for i in range(SelfHealingNetworkEnv.N_ACTIONS)]
cols = ['#95a5a6', '#3498db', '#2ecc71',
        '#e74c3c', '#9b59b6', '#f39c12', '#1abc9c']

for ax, (bs_type, bs_ids_sel) in zip(
        axes,
        [('Edge BS (BS-1 to BS-6)',
           list(range(1, CFG.NUM_BS))),
         ('Centre BS (BS-0)', [0])]):
    mask = np.isin(dqn_res['failed_bs_id'], bs_ids_sel)
    if mask.sum() == 0:
        ax.text(0.5, 0.5, 'No episodes',
                ha='center', transform=ax.transAxes)
        continue

    all_actions = []
    for i, ep_acts in enumerate(
            dqn_res['actions_taken']):
        if mask[i]:
            all_actions.extend(ep_acts)

    counts = np.zeros(SelfHealingNetworkEnv.N_ACTIONS)
    for a in all_actions:
        counts[a] += 1
    pct = counts / max(counts.sum(), 1) * 100

    bars = ax.bar(range(len(counts)), pct,
                  color=cols, edgecolor='black',
                  linewidth=0.7)
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(act_names, rotation=22,
                        ha='right', fontsize=8)
    ax.set_ylabel("Action Usage %", fontsize=10)
    ax.set_title(bs_type, fontsize=11,
                 fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.set_facecolor('white')
    for bar, p in zip(bars, pct):
        if p > 2:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.3,
                    f'{p:.0f}%', ha='center',
                    va='bottom', fontsize=7)

plt.tight_layout()
save_fig("12_action_analysis.png")


# ─────────────────────────────────────────────────────────────────────
# ══ SECTION 6: FINAL SUMMARY ══════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────

section("6. FINAL SUMMARY")

best_h_name = max(
    [(n, np.mean(r['coverage_pct']))
     for n, r in all_results.items()
     if n not in ('DQN Agent', 'PPO Agent')],
    key=lambda x: x[1])
dqn_cov = np.mean(all_results['DQN Agent']['coverage_pct'])
ppo_cov = np.mean(all_results['PPO Agent']['coverage_pct'])
dqn_sol = np.mean(all_results['DQN Agent']['solved'])*100
ppo_sol = np.mean(all_results['PPO Agent']['solved'])*100

print(f"\n  ┌{'─'*48}┐")
print(f"  │{'RESULTS SUMMARY':^48}│")
print(f"  ├{'─'*48}┤")
print(f"  │  Best heuristic : "
      f"{best_h_name[0]:20s} "
      f"{best_h_name[1]:5.1f}%  │")
print(f"  │  DQN Agent      : "
      f"{'':20s} {dqn_cov:5.1f}%  │")
print(f"  │  PPO Agent      : "
      f"{'':20s} {ppo_cov:5.1f}%  │")
print(f"  ├{'─'*48}┤")
print(f"  │  DQN gain vs best heuristic: "
      f"{dqn_cov-best_h_name[1]:+6.2f}%"
      f"{'':12}│")
print(f"  │  DQN solve rate: {dqn_sol:.1f}% "
      f"vs heuristic "
      f"{np.mean(all_results[best_h_name[0]]['solved'])*100:.1f}%"
      f"{'':4}│")
print(f"  └{'─'*48}┘")

print(f"\n  📊 Output figures saved to: {FIGURES_DIR}/")
figures = [
    "01_normal_operation.png",
    "02_outage_no_healing.png",
    "03_strategy_*.png",
    "04_sinr_cdf_heuristics.png",
    "05_cod_kpi_evolution.png",
    "06_cod_feature_importance.png",
    "07_cod_detection_probability.png",
    "08_training_curves.png",
    "09_coverage_comparison.png",
    "10_per_bs_heatmap.png",
    "11_sinr_cdf_comparison.png",
    "12_action_analysis.png",
]
for f in figures:
    print(f"    {f}")

print(f"\n  📋 Results saved to: {RESULTS_DIR}/")
print(f"    strategy_comparison.csv")
print(f"    evaluation_results.csv")

print(f"\n  🤖 Models saved to: {MODELS_DIR}/")
print(f"    dqn_self_healing.zip")
print(f"    ppo_self_healing.zip")
print(f"    cod_random_forest.pkl")

print(f"\n✅ Pipeline complete.")
