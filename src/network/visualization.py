"""
Network visualization utilities.
Plots BS/UE topology, coverage, and SINR before/after outage.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def plot_network(network, title="Network State",
                  save_path=None, show=True,
                  highlight_failed=True):
    """
    Plot the current state of a RadioNetwork: BS positions,
    UE associations, and coverage/outage status.

    Parameters
    ----------
    network : RadioNetwork
        Network instance (already simulated via
        compute_association_and_sinr()).
    title : str
        Plot title.
    save_path : str or None
        If provided, saves figure (PNG, 300 dpi) to this path.
    show : bool
        Whether to display the plot inline.
    highlight_failed : bool
        Draw a red X over any inactive (failed) BS.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    fig, ax = plt.subplots(figsize=(7, 7))

    colors = plt.cm.tab10(
        np.linspace(0, 1, len(network.base_stations)))

    # Plot UEs colour-coded by serving BS; outage UEs in black
    for ue in network.ues:
        if ue.in_outage or ue.serving_bs_id is None:
            ax.scatter(ue.x, ue.y, c='black', marker='x',
                       s=35, linewidths=1.5, zorder=3)
        else:
            ax.scatter(ue.x, ue.y,
                       c=[colors[ue.serving_bs_id]],
                       marker='o', s=18, alpha=0.7, zorder=2)

    # Plot BS positions
    for bs in network.base_stations:
        marker = '^' if bs.is_active else 'X'
        size = 220 if bs.is_active else 320
        ax.scatter(bs.x, bs.y,
                  c=[colors[bs.bs_id]],
                  marker=marker, s=size,
                  edgecolors='black', linewidths=1.5,
                  zorder=5)
        ax.annotate(f"BS{bs.bs_id}", (bs.x, bs.y),
                   textcoords="offset points",
                   xytext=(0, 14), ha='center',
                   fontsize=10, fontweight='bold')

        if highlight_failed and not bs.is_active:
            ax.scatter(bs.x, bs.y, c='red', marker='X',
                      s=420, linewidths=3,
                      edgecolors='darkred', zorder=6)

    stats = network.get_stats()
    subtitle = (f"Coverage: {stats['coverage_pct']:.1f}% | "
               f"Outage UEs: {stats['ues_in_outage']} | "
               f"Mean SINR: {stats['mean_sinr_db']:.1f} dB")

    ax.set_title(f"{title}\n{subtitle}", fontsize=12)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)

    legend_elems = [
        mpatches.Patch(color='gray', label='Served UE'),
        plt.Line2D([0], [0], marker='x', color='black',
                  linestyle='None', markersize=8,
                  label='Outage UE'),
        plt.Line2D([0], [0], marker='^', color='gray',
                  linestyle='None', markersize=12,
                  label='Active BS'),
        plt.Line2D([0], [0], marker='X', color='red',
                  linestyle='None', markersize=14,
                  label='Failed BS'),
    ]
    ax.legend(handles=legend_elems, loc='upper right',
             fontsize=8, framealpha=0.9)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path),
                   exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved figure -> {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax


def plot_before_after(network_before, network_after,
                      save_path=None, show=True,
                      title_before="Before Outage",
                      title_after="After Outage"):
    """
    Side-by-side comparison of network state before and
    after an outage event.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    for ax, net, ttl in zip(
            axes, [network_before, network_after],
            [title_before, title_after]):
        colors = plt.cm.tab10(
            np.linspace(0, 1, len(net.base_stations)))

        for ue in net.ues:
            if ue.in_outage or ue.serving_bs_id is None:
                ax.scatter(ue.x, ue.y, c='black', marker='x',
                          s=30, linewidths=1.3, zorder=3)
            else:
                ax.scatter(ue.x, ue.y,
                          c=[colors[ue.serving_bs_id]],
                          marker='o', s=15, alpha=0.7,
                          zorder=2)

        for bs in net.base_stations:
            marker = '^' if bs.is_active else 'X'
            size = 200 if bs.is_active else 300
            ax.scatter(bs.x, bs.y, c=[colors[bs.bs_id]],
                      marker=marker, s=size,
                      edgecolors='black', linewidths=1.4,
                      zorder=5)
            ax.annotate(f"BS{bs.bs_id}", (bs.x, bs.y),
                       textcoords="offset points",
                       xytext=(0, 12), ha='center',
                       fontsize=9, fontweight='bold')
            if not bs.is_active:
                ax.scatter(bs.x, bs.y, c='red', marker='X',
                          s=400, linewidths=3,
                          edgecolors='darkred', zorder=6)

        stats = net.get_stats()
        sub = (f"Coverage: {stats['coverage_pct']:.1f}% | "
              f"Outage UEs: {stats['ues_in_outage']}")
        ax.set_title(f"{ttl}\n{sub}", fontsize=12)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect('equal')
        ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path),
                   exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved figure -> {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, axes
