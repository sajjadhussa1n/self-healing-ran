"""
High-level factory functions for creating and manipulating
a RadioNetwork: network creation and outage simulation.
"""
import copy
from src.config import SimConfig
from src.network.radio_network import RadioNetwork


def create_network(config=None, seed=None, verbose=True):
    """
    Create and initialise a RadioNetwork: places BSs,
    drops UEs, and computes the initial (no-outage)
    association/SINR state.

    Parameters
    ----------
    config : SimConfig or None
        Simulation configuration. Uses default SimConfig()
        if None.
    seed : int or None
        Overrides config.RANDOM_SEED if provided.
    verbose : bool
        Print a short summary after creation.

    Returns
    -------
    network : RadioNetwork
    """
    cfg = config if config is not None else SimConfig()
    if seed is not None:
        cfg.RANDOM_SEED = seed

    network = RadioNetwork(cfg)
    network.compute_association_and_sinr()

    if verbose:
        stats = network.get_stats()
        print(f"[create_network] {cfg.NUM_BS} BSs, "
              f"{cfg.NUM_UE} UEs | "
              f"Coverage: {stats['coverage_pct']:.1f}% | "
              f"Mean SINR: {stats['mean_sinr_db']:.1f} dB")

    return network


def simulate_outage(network, failed_bs_id,
                    severity='full', verbose=True):
    """
    Simulate a BS outage event on a (deep) copy of the
    given network, returning both the original (pre-outage)
    and the new post-outage network for comparison.

    Parameters
    ----------
    network : RadioNetwork
        A network already created via create_network().
        This object is NOT modified; a deep copy is taken.
    failed_bs_id : int
        Index of the BS to fail.
    severity : str
        Outage severity passed to network.trigger_outage().
    verbose : bool
        Print before/after coverage stats.

    Returns
    -------
    network_before : RadioNetwork
        Deep copy of the network prior to the outage.
    network_after : RadioNetwork
        Deep copy of the network after the outage, with
        association/SINR recomputed.
    """
    network_before = copy.deepcopy(network)

    network_after = copy.deepcopy(network)
    network_after.trigger_outage(failed_bs_id,
                                 severity=severity)
    network_after.compute_association_and_sinr()

    if verbose:
        sb = network_before.get_stats()
        sa = network_after.get_stats()
        print(f"[simulate_outage] BS-{failed_bs_id} "
              f"({severity}):")
        print(f"  Before -> Coverage: "
              f"{sb['coverage_pct']:.1f}%, "
              f"Outage UEs: {sb['ues_in_outage']}")
        print(f"  After  -> Coverage: "
              f"{sa['coverage_pct']:.1f}%, "
              f"Outage UEs: {sa['ues_in_outage']}")

    return network_before, network_after
