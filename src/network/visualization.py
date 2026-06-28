# src/network/visualization.py
"""
Publication-quality three-panel network plotting:
  Panel 1: UE association (Voronoi cells, BS colours)
  Panel 2: SINR heatmap
  Panel 3: Binary coverage map

This is the exact plotting logic validated in the project
notebooks and used to produce the paper's figures.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize, ListedColormap
from scipy.spatial import Voronoi, voronoi_plot_2d


def compute_sinr_grid(network, resolution=150):
    """
    Compute SINR grid using log-distance path loss +
    3GPP TR 36.942 antenna gain at current BS tilt angles.
    Consistent with UE association calculations.
    """
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
                d2d = max(np.sqrt((X[i, j]-bs.x)**2 +
                                   (Y[i, j]-bs.y)**2), 1.0)
                pl  = (20*np.log10(4*np.pi/3e8) +
                       20*np.log10(f) +
                       10*cfg.PATH_LOSS_EXPONENT*np.log10(d2d))
                ag  = network._antenna_gain_db(d2d, bs.tilt_deg)
                rx  = bs.tx_power_dbm - pl + ag
                rx_dbms.append(rx)
                rx_lins.append(10**(rx/10)
                              if np.isfinite(rx) else 0.0)

            active_rx    = [p for p in rx_dbms if np.isfinite(p)]
            best_pwr_dbm = (max(active_rx) if active_rx
                            else -np.inf)

            if best_pwr_dbm < cfg.MIN_RX_POWER_DBM:
                sinr_grid[i, j] = -20.0
                continue

            best_idx     = int(np.argmax(rx_lins))
            signal       = rx_lins[best_idx]
            interference = sum(rx_lins) - signal
            sinr         = signal / (interference + noise_linear)
            sinr_grid[i, j] = 10*np.log10(max(sinr, 1e-15))

    return X, Y, sinr_grid


def compute_coverage_grid(sinr_grid, network, threshold_db=None):
    if threshold_db is None:
        threshold_db = network.cfg.SINR_OUTAGE_THRESHOLD_DB
    return (sinr_grid >= threshold_db).astype(float)


def _style_ax(ax, cfg):
    ax.set_xlim(0, cfg.AREA_SIZE)
    ax.set_ylim(0, cfg.AREA_SIZE)
    ax.set_aspect('equal')
    ax.set_facecolor('white')
    ax.grid(True, color='grey', linestyle='--',
           linewidth=0.4, alpha=0.6, zorder=0)
    ax.set_xlabel("x (m)", fontsize=10)
    ax.set_ylabel("y (m)", fontsize=10)
    ax.tick_params(labelsize=9)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def _draw_bs_markers(ax, network, fontsize=7):
    for bs in network.base_stations:
        marker = '^' if bs.is_active else 'X'
        edge   = 'black' if bs.is_active else 'red'
        face   = bs.color if bs.is_active else 'red'
        ax.scatter(bs.x, bs.y, color=face, s=160,
                  marker=marker, edgecolors=edge,
                  linewidths=1.4, zorder=6)
        label = (f"BS{bs.bs_id}\n"
                 f"{bs.tx_power_dbm:.0f}dBm\n"
                 f"{bs.tilt_deg:.1f}\u00b0")
        ax.annotate(label, (bs.x, bs.y),
                   textcoords="offset points",
                   xytext=(6, 6), fontsize=fontsize,
                   color='black' if bs.is_active else 'red',
                   fontweight='bold')


def plot_network(network, title="Radio Network",
                 grid_resolution=150, figsize=(20, 6),
                 save_path=None, show=True):
    """
    Three-panel network plot: UE association, SINR map,
    binary coverage map.

    Parameters
    ----------
    network : RadioNetwork
    title : str
        Overall figure title — pass a descriptive label
        such as "Before Outage", "After BS-0 Outage",
        or "After Strategy S5 (Tilt Only)".
    grid_resolution : int
    figsize : tuple
    save_path : str or None
        If provided, saves PNG (300 dpi) to this exact path.
    show : bool

    Returns
    -------
    fig, axes
    """
    cfg = network.cfg
    print(f"  Computing SINR grid "
         f"(resolution={grid_resolution})...", end="", flush=True)
    X, Y, sinr_grid = compute_sinr_grid(network, grid_resolution)
    cov_grid = compute_coverage_grid(sinr_grid, network)
    print(" done.")

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)

    # ── Panel 1: UE Association ────────────────────────────────────
    ax1 = axes[0]
    _style_ax(ax1, cfg)
    ax1.set_title("UE Association", fontsize=11,
                 fontweight='bold', pad=8)

    bs_pts  = np.array([[bs.x, bs.y]
                        for bs in network.base_stations])
    pad_pts = []
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            for bp in bs_pts:
                pad_pts.append(bp + np.array(
                    [sx*4*cfg.AREA_SIZE, sy*4*cfg.AREA_SIZE]))
    try:
        vor = Voronoi(np.vstack([bs_pts, pad_pts]))
        voronoi_plot_2d(vor, ax=ax1, show_vertices=False,
                       line_colors='#888888', line_width=1.0,
                       line_alpha=0.5, point_size=0)
    except Exception:
        pass
    ax1.set_xlim(0, cfg.AREA_SIZE)
    ax1.set_ylim(0, cfg.AREA_SIZE)

    for ue in network.ues:
        if ue.serving_bs_id is not None:
            color = network.base_stations[ue.serving_bs_id].color
            ax1.scatter(ue.x, ue.y, color=color, s=22,
                       marker='o', alpha=0.85,
                       edgecolors='none', zorder=4)
        else:
            ax1.scatter(ue.x, ue.y, color='red', s=35,
                       marker='x', linewidths=1.5, zorder=4)

    _draw_bs_markers(ax1, network)
    handles = [mpatches.Patch(
        color=bs.color,
        label=(f"BS-{bs.bs_id}" +
              (" \u2717" if not bs.is_active else "")))
              for bs in network.base_stations]
    ax1.legend(handles=handles, loc='lower left',
              fontsize=7, framealpha=0.85, edgecolor='grey')

    # ── Panel 2: SINR heatmap ───────────────────────────────────────
    ax2 = axes[1]
    _style_ax(ax2, cfg)
    ax2.set_title("SINR Map (dB)", fontsize=11,
                 fontweight='bold', pad=8)

    vmin, vmax = -10, 30
    im = ax2.imshow(
        sinr_grid, origin='lower',
        extent=[0, cfg.AREA_SIZE, 0, cfg.AREA_SIZE],
        cmap='RdYlGn', vmin=vmin, vmax=vmax,
        aspect='equal', interpolation='bilinear', zorder=1)
    cbar = plt.colorbar(im, ax=ax2, shrink=0.82, pad=0.02)
    cbar.set_label("SINR (dB)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    sinr_vals = np.array([
        ue.sinr_db if (ue.sinr_db is not None and
                      np.isfinite(ue.sinr_db)) else -20
        for ue in network.ues])
    norm = Normalize(vmin=vmin, vmax=vmax)
    ax2.scatter(
        [ue.x for ue in network.ues],
        [ue.y for ue in network.ues],
        c=sinr_vals, cmap='RdYlGn', norm=norm,
        s=25, edgecolors='black', linewidths=0.3,
        zorder=5, alpha=0.9)
    _draw_bs_markers(ax2, network)

    n_out = sum(1 for ue in network.ues if ue.in_outage)
    ax2.text(
        0.02, 0.98,
        f"Outage UEs: {n_out}/{len(network.ues)}\n"
        f"Threshold: {cfg.SINR_OUTAGE_THRESHOLD_DB}dB",
        transform=ax2.transAxes, fontsize=8,
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.4',
                 facecolor='white', edgecolor='grey', alpha=0.85))

    # ── Panel 3: Coverage map ───────────────────────────────────────
    ax3 = axes[2]
    _style_ax(ax3, cfg)
    ax3.set_title("Coverage Map (Binary)", fontsize=11,
                 fontweight='bold', pad=8)

    cov_cmap = ListedColormap(['#f28b82', '#b7e1a1'])
    ax3.imshow(
        cov_grid, origin='lower',
        extent=[0, cfg.AREA_SIZE, 0, cfg.AREA_SIZE],
        cmap=cov_cmap, vmin=0, vmax=1,
        aspect='equal', interpolation='nearest',
        zorder=1, alpha=0.75)

    covered = [ue for ue in network.ues if not ue.in_outage]
    outage  = [ue for ue in network.ues if ue.in_outage]
    if covered:
        ax3.scatter([u.x for u in covered],
                   [u.y for u in covered],
                   color='#1a7f37', s=22, marker='o',
                   edgecolors='none', alpha=0.85,
                   zorder=5, label='Covered UE')
    if outage:
        ax3.scatter([u.x for u in outage],
                   [u.y for u in outage],
                   color='#c0392b', s=50, marker='x',
                   linewidths=1.8, zorder=6, label='Outage UE')

    _draw_bs_markers(ax3, network)
    cov_patch = mpatches.Patch(color='#b7e1a1', label='Covered region')
    out_patch = mpatches.Patch(color='#f28b82', label='Outage region')
    ue_cov    = mpatches.Patch(color='#1a7f37', label='Covered UE')
    ue_out    = mpatches.Patch(color='#c0392b', label='Outage UE')
    ax3.legend(handles=[cov_patch, out_patch, ue_cov, ue_out],
              loc='lower left', fontsize=7,
              framealpha=0.85, edgecolor='grey')

    pct_area = 100 * cov_grid.mean()
    pct_ues  = 100 * (len(covered) / len(network.ues))
    ax3.text(
        0.02, 0.98,
        f"Area covered : {pct_area:.1f}%\n"
        f"UEs covered  : {pct_ues:.1f}%",
        transform=ax3.transAxes, fontsize=8,
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.4',
                 facecolor='white', edgecolor='grey', alpha=0.85))

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved figure -> {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, axes
