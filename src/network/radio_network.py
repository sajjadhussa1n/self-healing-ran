# src/network/radio_network.py
"""
RadioNetwork — core simulation class.
Manages topology, channel model, UE association,
SINR computation, cell outage, and compensation strategies.
"""

import numpy as np
import random
import matplotlib.pyplot as plt
from typing import List, Optional, Dict, Tuple

from src.config import SimConfig, CFG
from src.network.base_station import BaseStation
from src.network.user_equipment import UserEquipment
from src.compensation.strategies import CompensationStrategies


class RadioNetwork(CompensationStrategies):
    """
    Core radio network simulation.

    Implements:
      - Hexagonal BS topology
      - Log-distance path loss + 3GPP antenna model
      - UE association (max received power)
      - SINR computation
      - Cell outage (complete / partial / soft)
      - Six compensation strategies (S1-S6)
    """

    def __init__(self, config: SimConfig = None):
        self.cfg           = config if config else CFG
        self.base_stations: List[BaseStation] = []
        self.ues:           List[UserEquipment] = []
        self._setup_topology()
        self._assign_bs_colors()

    # ── Construction ─────────────────────────────────────────────────

    def _setup_topology(self) -> None:
        """Place BSs in hexagonal layout and cluster UEs."""
        np.random.seed(self.cfg.RANDOM_SEED)
        random.seed(self.cfg.RANDOM_SEED)

        cx, cy = self.cfg.AREA_SIZE / 2, self.cfg.AREA_SIZE / 2
        r      = self.cfg.HEX_RADIUS
        positions = [(cx, cy)]
        for k in range(6):
            angle = np.radians(k * 60)
            positions.append(
                (cx + r * np.cos(angle),
                 cy + r * np.sin(angle)))

        for i, (x, y) in enumerate(
                positions[:self.cfg.NUM_BS]):
            self.base_stations.append(
                BaseStation(i, x, y,
                            self.cfg.BS_TX_POWER_DBM))

        n_bs      = len(self.base_stations)
        base_cnt  = self.cfg.NUM_UE // n_bs
        remainder = self.cfg.NUM_UE % n_bs
        counts    = [base_cnt + (1 if i < remainder else 0)
                     for i in range(n_bs)]

        ue_id = 0
        for bs, count in zip(self.base_stations, counts):
            for _ in range(count):
                while True:
                    x = np.random.normal(
                        bs.x, self.cfg.UE_CLUSTER_STD)
                    y = np.random.normal(
                        bs.y, self.cfg.UE_CLUSTER_STD)
                    if (0 <= x <= self.cfg.AREA_SIZE and
                            0 <= y <= self.cfg.AREA_SIZE):
                        break
                self.ues.append(UserEquipment(ue_id, x, y))
                ue_id += 1

    def _assign_bs_colors(self) -> None:
        palette = plt.cm.Set1(
            np.linspace(0, 1, len(self.base_stations)))
        for bs, col in zip(self.base_stations, palette):
            bs.color = col

    # ── Channel model ─────────────────────────────────────────────────

    def _distance(self, ue: UserEquipment,
                   bs: BaseStation) -> float:
        return max(np.sqrt((ue.x - bs.x)**2 +
                            (ue.y - bs.y)**2), 1.0)

    def _path_loss_db(self, distance_m: float) -> float:
        """Log-distance path loss model."""
        f  = self.cfg.FREQUENCY_GHZ * 1e9
        pl = (20 * np.log10(4 * np.pi / 3e8) +
              20 * np.log10(f) +
              10 * self.cfg.PATH_LOSS_EXPONENT *
              np.log10(distance_m))
        if self.cfg.SHADOW_FADING_STD > 0:
            pl += np.random.normal(
                0, self.cfg.SHADOW_FADING_STD)
        return pl

    def _antenna_gain_db(self, distance_2d_m: float,
                          tilt_deg: float) -> float:
        """
        3GPP TR 36.942 vertical antenna pattern gain (dB).
        Reducing tilt widens the coverage footprint outward.
        """
        cfg = self.cfg
        d   = max(distance_2d_m, 1.0)
        dh  = cfg.BS_HEIGHT_M - cfg.UE_HEIGHT_M
        vertical_angle = np.degrees(np.arctan(dh / d))
        theta = vertical_angle - tilt_deg
        gain  = -min(
            12.0 * (theta /
                     cfg.ANTENNA_VERTICAL_BEAMWIDTH_DEG)**2,
            cfg.ANTENNA_SLA_DB)
        return gain

    def _rx_power_dbm(self, ue: UserEquipment,
                       bs: BaseStation) -> float:
        """Received power including path loss + antenna gain."""
        if not bs.is_active:
            return -np.inf
        d2d      = self._distance(ue, bs)
        pl       = self._path_loss_db(d2d)
        ant_gain = self._antenna_gain_db(d2d, bs.tilt_deg)
        return bs.tx_power_dbm - pl + ant_gain

    def _pl(self, x1: float, y1: float,
             x2: float, y2: float,
             tilt_deg: float = None) -> float:
        """
        Effective path loss between two coordinate pairs.
        Used by compensation strategies (no UE/BS objects).
        """
        if tilt_deg is None:
            tilt_deg = self.cfg.BS_TILT_DEFAULT_DEG
        d  = max(np.sqrt((x1-x2)**2 + (y1-y2)**2), 1.0)
        f  = self.cfg.FREQUENCY_GHZ * 1e9
        pl = (20 * np.log10(4 * np.pi / 3e8) +
              20 * np.log10(f) +
              10 * self.cfg.PATH_LOSS_EXPONENT *
              np.log10(d))
        ant_gain = self._antenna_gain_db(d, tilt_deg)
        return pl - ant_gain

    # ── Association & SINR ────────────────────────────────────────────

    def compute_association_and_sinr(self) -> None:
        """Associate every UE to best BS and compute SINR."""
        noise_linear = 10**(self.cfg.NOISE_POWER_DBM / 10)

        for ue in self.ues:
            rx_powers_dbm    = [self._rx_power_dbm(ue, bs)
                                 for bs in self.base_stations]
            rx_powers_linear = [
                10**(p / 10) if np.isfinite(p) else 0.0
                for p in rx_powers_dbm]

            active_rx = [
                (p, i) for i, p in enumerate(rx_powers_dbm)
                if self.base_stations[i].is_active]

            if not active_rx:
                ue.serving_bs_id      = None
                ue.received_power_dbm = -np.inf
                ue.sinr_db            = -np.inf
                ue.throughput_mbps    = 0.0
                ue.in_outage          = True
                continue

            best_rx_dbm, best_idx = max(
                active_rx, key=lambda x: x[0])

            if best_rx_dbm < self.cfg.MIN_RX_POWER_DBM:
                ue.serving_bs_id      = None
                ue.received_power_dbm = best_rx_dbm
                ue.sinr_db            = -np.inf
                ue.throughput_mbps    = 0.0
                ue.in_outage          = True
                continue

            ue.serving_bs_id      = best_idx
            ue.received_power_dbm = best_rx_dbm
            signal       = rx_powers_linear[best_idx]
            interference = sum(rx_powers_linear) - signal
            sinr         = signal / (interference +
                                      noise_linear)
            ue.sinr_db   = 10 * np.log10(max(sinr, 1e-15))
            ue.throughput_mbps = (
                self.cfg.BANDWIDTH_MHZ *
                np.log2(1 + max(sinr, 1e-15)))
            ue.in_outage = (
                ue.sinr_db <
                self.cfg.SINR_OUTAGE_THRESHOLD_DB or
                ue.received_power_dbm <
                self.cfg.MIN_RX_POWER_DBM)

    # ── Cell outage ───────────────────────────────────────────────────

    def trigger_outage(self, bs_id: int,
                        severity: str = 'full') -> None:
        """
        Trigger BS outage.

        severity:
          'full'    — complete failure (power = -inf)
          'partial' — -20 dB hardware fault
          'soft'    — -10 dB software fault
        """
        bs = self.base_stations[bs_id]
        if severity == 'full':
            bs.toggle_active(False)
            print(f"⚠️  BS-{bs_id} COMPLETE OUTAGE")
        elif severity == 'partial':
            new_pwr = (bs.nominal_power_dbm -
                       self.cfg.OUTAGE_POWER_REDUCTION_DB)
            bs.set_power(new_pwr)
            print(f"⚠️  BS-{bs_id} PARTIAL OUTAGE "
                  f"({self.cfg.OUTAGE_POWER_REDUCTION_DB}"
                  f"dB reduction → {new_pwr:.1f}dBm)")
        elif severity == 'soft':
            new_pwr = bs.nominal_power_dbm - 10.0
            bs.set_power(new_pwr)
            print(f"⚠️  BS-{bs_id} SOFT FAULT "
                  f"(-10dB → {new_pwr:.1f}dBm)")

    def restore_bs(self, bs_id: int) -> None:
        """Restore BS to nominal state."""
        bs = self.base_stations[bs_id]
        bs.toggle_active(True)
        bs.tx_power_dbm = bs.nominal_power_dbm
        bs.tilt_deg     = bs.nominal_tilt_deg
        print(f"✅ BS-{bs_id} restored.")

    def reset_all_powers(self) -> None:
        """Return all active BSs to nominal power and tilt."""
        for bs in self.base_stations:
            if bs.is_active:
                bs.tx_power_dbm = bs.nominal_power_dbm
                bs.tilt_deg     = bs.nominal_tilt_deg

    # ── Neighbour utilities ───────────────────────────────────────────

    def _neighbour_distances(
            self, failed_bs_id: int
    ) -> List[Tuple[float, BaseStation]]:
        fbs    = self.base_stations[failed_bs_id]
        result = []
        for bs in self.base_stations:
            if bs.bs_id == failed_bs_id or not bs.is_active:
                continue
            d = np.sqrt((bs.x - fbs.x)**2 +
                        (bs.y - fbs.y)**2)
            result.append((d, bs))
        result.sort(key=lambda x: x[0])
        return result

    def _auto_neighbour_count(self,
                               failed_bs_id: int) -> int:
        dist_bs = self._neighbour_distances(failed_bs_id)
        if not dist_bs:
            return 1
        dists = [d for d, _ in dist_bs]
        cv    = np.std(dists) / (np.mean(dists) + 1e-9)
        print(f"  [Auto] BS-{failed_bs_id}: "
              f"CV={cv:.4f}", end="")
        if cv < 0.15:
            n = len(dists)
            print(f" → centre BS → all {n} neighbours, "
                  f"extended cap")
            return n
        print(f" → edge BS → 3 nearest neighbours")
        return 3

    def _power_cap(self, failed_bs_id: int) -> float:
        dists = [d for d, _ in
                 self._neighbour_distances(failed_bs_id)]
        if not dists:
            return self.cfg.MAX_POWER_BOOST_DB
        cv = np.std(dists) / (np.mean(dists) + 1e-9)
        return (self.cfg.MAX_POWER_BOOST_CENTRE_DB
                if cv < 0.15
                else self.cfg.MAX_POWER_BOOST_DB)

    def get_neighbours(
            self, failed_bs_id: int,
            max_neighbours: int = None
    ) -> List[BaseStation]:
        dist_bs = self._neighbour_distances(failed_bs_id)
        if max_neighbours is None:
            return [bs for _, bs in dist_bs]
        return [bs for _, bs in dist_bs[:max_neighbours]]

    # ── Frozen interference helper ────────────────────────────────────

    def _frozen_interference(
            self, ue: UserEquipment,
            serving_bs_id: int,
            use_nominal: bool = True) -> float:
        """
        Interference at UE from all active BSs except serving.
        use_nominal=True: frozen pre-boost snapshot.
        """
        total = 0.0
        for obs in self.base_stations:
            if (obs.bs_id == serving_bs_id or
                    not obs.is_active):
                continue
            pwr  = (obs.nominal_power_dbm if use_nominal
                    else obs.tx_power_dbm)
            tilt = (obs.nominal_tilt_deg if use_nominal
                    else obs.tilt_deg)
            pl   = self._pl(obs.x, obs.y,
                             ue.x, ue.y, tilt)
            total += 10**((pwr - pl) / 10)
        return total

    # ── Joint network evaluation ──────────────────────────────────────

    def _evaluate_network_state(
            self,
            proposed_tilts: Dict[int, float],
            proposed_powers: Dict[int, float]
    ) -> Tuple[Dict, int, int, int]:
        """
        Evaluate SINR for all UEs under proposed parameters
        without modifying the actual network.
        Returns: (sinrs, n_outage, n_rescued, n_collateral)
        """
        noise_linear = 10**(self.cfg.NOISE_POWER_DBM / 10)

        def get_proposed_rx(ue, bs):
            if not bs.is_active:
                return -np.inf
            pwr  = proposed_powers.get(
                bs.bs_id, bs.tx_power_dbm)
            tilt = proposed_tilts.get(
                bs.bs_id, bs.tilt_deg)
            d2d      = self._distance(ue, bs)
            pl       = self._path_loss_db(d2d)
            ant_gain = self._antenna_gain_db(d2d, tilt)
            return pwr - pl + ant_gain

        sinrs        = {}
        n_outage     = 0
        n_rescued    = 0
        n_collateral = 0

        for ue in self.ues:
            rx_dbms = [get_proposed_rx(ue, bs)
                       for bs in self.base_stations]
            rx_lins = [10**(p/10) if np.isfinite(p)
                       else 0.0 for p in rx_dbms]
            active  = [(p, i) for i, p
                       in enumerate(rx_dbms)
                       if self.base_stations[i].is_active]

            if not active:
                sinrs[ue.ue_id] = -np.inf
                n_outage += 1
                continue

            best_rx, best_idx = max(
                active, key=lambda x: x[0])
            if best_rx < self.cfg.MIN_RX_POWER_DBM:
                sinrs[ue.ue_id] = -np.inf
                n_outage += 1
                continue

            signal       = rx_lins[best_idx]
            interference = sum(rx_lins) - signal
            sinr         = signal / (interference +
                                      noise_linear)
            sinr_db      = 10 * np.log10(
                max(sinr, 1e-15))
            sinrs[ue.ue_id] = sinr_db

            was_out = ue.in_outage
            now_out = (
                sinr_db <
                self.cfg.SINR_OUTAGE_THRESHOLD_DB or
                best_rx < self.cfg.MIN_RX_POWER_DBM)

            if now_out:
                n_outage += 1
            if was_out and not now_out:
                n_rescued += 1
            if not was_out and now_out:
                n_collateral += 1

        return sinrs, n_outage, n_rescued, n_collateral

    # ── Tilt safety check ─────────────────────────────────────────────

    def _tilt_is_safe(self, bs: BaseStation,
                       new_tilt: float) -> bool:
        """
        Check that reducing tilt on bs does not drop
        any currently-served UE into outage.
        """
        noise_linear = 10**(self.cfg.NOISE_POWER_DBM / 10)
        served_ues   = [ue for ue in self.ues
                        if ue.serving_bs_id == bs.bs_id
                        and not ue.in_outage]
        for ue in served_ues:
            d2d      = self._distance(ue, bs)
            pl       = self._path_loss_db(d2d)
            ant_gain = self._antenna_gain_db(d2d, new_tilt)
            new_rx   = bs.tx_power_dbm - pl + ant_gain
            interf   = self._frozen_interference(
                ue, bs.bs_id)
            sinr_lin = (10**(new_rx / 10) /
                        (interf + noise_linear))
            sinr_db  = 10 * np.log10(
                max(sinr_lin, 1e-15))
            if sinr_db < self.cfg.SINR_OUTAGE_THRESHOLD_DB:
                return False
        return True

    # ── Stats ─────────────────────────────────────────────────────────

    def get_stats(self) -> Dict:
        """Return network-wide performance statistics."""
        sinrs  = [ue.sinr_db for ue in self.ues
                  if ue.sinr_db is not None and
                  np.isfinite(ue.sinr_db)]
        thrpts = [ue.throughput_mbps for ue in self.ues
                  if ue.throughput_mbps is not None]
        n_out  = sum(1 for ue in self.ues if ue.in_outage)
        n_deep = sum(1 for ue in self.ues
                     if ue.sinr_db is not None and
                     np.isfinite(ue.sinr_db) and
                     ue.sinr_db <
                     self.cfg.SINR_COMPENSATION_THRESHOLD_DB)
        n_marg = sum(1 for ue in self.ues
                     if ue.sinr_db is not None and
                     np.isfinite(ue.sinr_db) and
                     self.cfg.SINR_COMPENSATION_THRESHOLD_DB
                     <= ue.sinr_db <
                     self.cfg.SINR_OUTAGE_THRESHOLD_DB)
        return {
            'mean_sinr_db'      : (np.mean(sinrs)
                                    if sinrs else -np.inf),
            'median_sinr_db'    : (np.median(sinrs)
                                    if sinrs else -np.inf),
            '5th_pct_sinr_db'   : (np.percentile(sinrs, 5)
                                    if sinrs else -np.inf),
            'mean_throughput'   : (np.mean(thrpts)
                                    if thrpts else 0.0),
            '5th_pct_throughput': (np.percentile(thrpts, 5)
                                    if thrpts else 0.0),
            'ues_in_outage'     : n_out,
            'ues_deep_outage'   : n_deep,
            'ues_marginal'      : n_marg,
            'coverage_pct'      : 100 * (1 - n_out /
                                          len(self.ues)),
        }
