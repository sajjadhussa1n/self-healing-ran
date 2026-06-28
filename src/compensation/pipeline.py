"""
High-level wrapper that simulates all six heuristic
compensation strategies (S1-S6) for a given outage,
plotting the resulting network state for each.
"""
import os
import copy
import pandas as pd

from src.network.visualization import plot_network


STRATEGY_FUNCS = {
    "S1": "apply_power_compensation",
    "S2": "apply_proportional_compensation",
    "S3": "apply_targeted_compensation",
    "S4": "apply_simultaneous_compensation",
    "S5": "apply_tilt_compensation",
    "S6": "apply_joint_compensation",
}


def simulate_coc_strategies(network_after_outage,
                            failed_bs_id,
                            strategies=None,
                            plot=True,
                            save_dir="docs/figures",
                            verbose=True):
    """
    Apply each of the six heuristic compensation strategies
    (independently, starting from the same post-outage
    network state) and report/plot the resulting coverage.

    Parameters
    ----------
    network_after_outage : RadioNetwork
        Network state immediately after the outage
        (before any compensation). This object is not
        modified; each strategy runs on its own deep copy.
    failed_bs_id : int
        The BS that failed.
    strategies : list[str] or None
        Subset of {"S1",...,"S6"} to run.
        Defaults to all six.
    plot : bool
        If True, saves a per-strategy network plot.
    save_dir : str
        Directory for saved figures.
    verbose : bool

    Returns
    -------
    results : dict
        {strategy_name: {"network": RadioNetwork,
                          "stats": dict}}
    summary_df : pd.DataFrame
        Tabular summary (coverage %, outage UEs, mean SINR)
        across all strategies, plus the no-action baseline.
    """
    strategies = strategies or list(STRATEGY_FUNCS.keys())
    results = {}
    rows = []

    baseline_stats = network_after_outage.get_stats()
    rows.append({"strategy": "No Action",
                **baseline_stats})

    if verbose:
        print("=" * 60)
        print(f"COC STRATEGY SIMULATION (BS-{failed_bs_id})")
        print("=" * 60)
        print(f"No Action -> Coverage: "
             f"{baseline_stats['coverage_pct']:.1f}%, "
             f"Outage UEs: "
             f"{baseline_stats['ues_in_outage']}")

    for name in strategies:
        method_name = STRATEGY_FUNCS[name]
        net = copy.deepcopy(network_after_outage)
        getattr(net, method_name)(failed_bs_id)
        net.compute_association_and_sinr()

        stats = net.get_stats()
        results[name] = {"network": net, "stats": stats}
        rows.append({"strategy": name, **stats})

        if verbose:
            print(f"{name} ({method_name}) -> "
                 f"Coverage: {stats['coverage_pct']:.1f}%, "
                 f"Outage UEs: {stats['ues_in_outage']}")
            
        if plot:
            os.makedirs(save_dir, exist_ok=True)
            plot_network(
                net,
                title=f"After Strategy {name} "
                     f"({STRATEGY_FUNCS[name].replace('apply_', '').replace('_', ' ').title()})",
                save_path=os.path.join(save_dir, f"coc_{name.lower()}.png"),
                show=True)

    summary_df = pd.DataFrame(rows)
    return results, summary_df
