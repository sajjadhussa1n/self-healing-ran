# src/detection/kpi_logger.py
"""
KPI Logger
==========
Monitors and logs per-BS KPIs at each simulation timestep.
Implements three-class labelling:
  0 = Normal
  1 = Outage (root cause BS)
  2 = Collaterally degraded (overloaded neighbour)

All features are observable through the O-RAN E2 interface
as standardised PM counters. No simulation artefacts
(outage flags, tx_power, synthetic SINR) are included.
"""

import numpy as np
import pandas as pd
from typing import Optional


class KPISnapshot:
    """One KPI observation for one BS at one timestep."""

    def __init__(self, episode, timestep, bs_id, label,
                 own_kpis, delta_kpis, neighbour_kpis):
        self.episode        = episode
        self.timestep       = timestep
        self.bs_id          = bs_id
        self.label          = label
        self.own_kpis       = own_kpis
        self.delta_kpis     = delta_kpis
        self.neighbour_kpis = neighbour_kpis

    def to_dict(self):
        d = {
            'episode' : self.episode,
            'timestep': self.timestep,
            'bs_id'   : self.bs_id,
            'label'   : self.label,
        }
        d.update(self.own_kpis)
        d.update(self.delta_kpis)
        d.update(self.neighbour_kpis)
        return d


class KPILogger:
    """
    Monitors and logs per-BS KPIs at each simulation timestep.

    Three-class labelling:
      0 = Normal
      1 = Outage (root cause BS)
      2 = Collaterally degraded neighbour

    Features:
      Own-cell : ue_count, prb_load, ue_ratio
      Delta    : delta_ue_count, delta_prb_load
      Neighbours: ue_count, delta_ue, mean_sinr,
                  prb_load × 3 nearest BSs

    NOTE: Own-cell SINR, throughput, rx_power excluded —
    they collapse to 0 when BS has no UEs (degenerate).
    outage_ues excluded — circular reasoning.
    """

    SINR_DROP_THRESHOLD_DB = 3.0
    OUTAGE_UE_THRESHOLD    = 2

    def __init__(self, network=None):
        self.snapshots = []
        self._prev     = {}
        self._network  = None
        self._max_ues  = 1.0
        if network is not None:
            self.network = network

    @property
    def network(self):
        return self._network

    @network.setter
    def network(self, net):
        self._network = net
        if net is not None:
            self._max_ues = (net.cfg.NUM_UE /
                             max(net.cfg.NUM_BS, 1))

    def reset(self):
        """Call at start of each new episode."""
        self._prev = {}

    def _own_kpis(self, bs) -> dict:
        served   = [ue for ue in self.network.ues
                    if ue.serving_bs_id == bs.bs_id]
        ue_count = len(served)
        prb_load = min(ue_count /
                       max(self._max_ues, 1), 1.0)
        prev     = self._prev.get(bs.bs_id, {})
        prev_ue  = prev.get('ue_count', ue_count)
        ue_ratio = (ue_count / max(prev_ue, 1)
                    if prev_ue > 0 else 1.0)
        return {
            'ue_count' : ue_count,
            'prb_load' : round(prb_load, 3),
            'ue_ratio' : round(ue_ratio,  3),
            'is_active': int(bs.is_active),
        }

    def _delta_kpis(self, bs_id, current) -> dict:
        if bs_id not in self._prev:
            return {'delta_ue_count': 0,
                    'delta_prb_load': 0.0}
        prev = self._prev[bs_id]
        return {
            'delta_ue_count': (current['ue_count'] -
                                prev['ue_count']),
            'delta_prb_load': round(
                current['prb_load'] -
                prev['prb_load'], 3),
        }

    def _neighbour_kpis(self, bs) -> dict:
        net   = self.network
        dists = []
        for other in net.base_stations:
            if other.bs_id == bs.bs_id:
                continue
            d = np.sqrt((other.x - bs.x)**2 +
                        (other.y - bs.y)**2)
            dists.append((d, other))
        dists.sort(key=lambda x: x[0])
        neighbours = [b for _, b in dists[:3]]

        result = {}
        for k, nb in enumerate(neighbours, start=1):
            served  = [ue for ue in net.ues
                       if ue.serving_bs_id == nb.bs_id]
            nb_ue   = len(served)
            sinrs   = [ue.sinr_db for ue in served
                       if (ue.sinr_db is not None and
                           np.isfinite(ue.sinr_db))]
            nb_sinr = np.mean(sinrs) if sinrs else 0.0
            prev_nb = self._prev.get(nb.bs_id, {})
            prev_ue = prev_nb.get('ue_count', nb_ue)
            result[f'n{k}_bs_id']    = nb.bs_id
            result[f'n{k}_ue_count'] = nb_ue
            result[f'n{k}_delta_ue'] = nb_ue - prev_ue
            result[f'n{k}_mean_sinr']= round(nb_sinr, 3)
            result[f'n{k}_prb_load'] = round(
                min(nb_ue / max(self._max_ues, 1), 1.0),
                3)
        return result

    def _get_degraded_bs_ids(self,
                              failed_bs_id) -> set:
        degraded = set()
        if failed_bs_id is None:
            return degraded
        net = self.network
        for bs in net.base_stations:
            if bs.bs_id == failed_bs_id:
                continue
            if not bs.is_active:
                continue
            served = [ue for ue in net.ues
                      if ue.serving_bs_id == bs.bs_id]
            if not served:
                continue
            sinrs = [ue.sinr_db for ue in served
                     if (ue.sinr_db is not None and
                         np.isfinite(ue.sinr_db))]
            if not sinrs:
                continue
            mean_sinr  = np.mean(sinrs)
            n_outage   = sum(1 for ue in served
                              if ue.in_outage)
            prev          = self._prev.get(bs.bs_id, {})
            baseline_sinr = prev.get('mean_sinr_db',
                                      mean_sinr)
            sinr_drop     = baseline_sinr - mean_sinr
            if (sinr_drop > self.SINR_DROP_THRESHOLD_DB
                    or n_outage > self.OUTAGE_UE_THRESHOLD):
                degraded.add(bs.bs_id)
        return degraded

    def log(self, episode: int, timestep: int,
             failed_bs_id: int = None) -> None:
        """Record one KPI snapshot for every BS."""
        net      = self.network
        degraded = self._get_degraded_bs_ids(failed_bs_id)

        for bs in net.base_stations:
            own   = self._own_kpis(bs)
            delta = self._delta_kpis(bs.bs_id, own)
            nb    = self._neighbour_kpis(bs)

            if (failed_bs_id is not None and
                    bs.bs_id == failed_bs_id):
                label = 1
            elif bs.bs_id in degraded:
                label = 2
            else:
                label = 0

            self.snapshots.append(KPISnapshot(
                episode        = episode,
                timestep       = timestep,
                bs_id          = bs.bs_id,
                label          = label,
                own_kpis       = own,
                delta_kpis     = delta,
                neighbour_kpis = nb,
            ))
            self._prev[bs.bs_id] = own

    def get_dataframe(self) -> pd.DataFrame:
        """Return all snapshots as a pandas DataFrame."""
        return pd.DataFrame(
            [s.to_dict() for s in self.snapshots])

    def clear(self) -> None:
        self.snapshots = []
        self._prev     = {}
