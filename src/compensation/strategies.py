# src/compensation/strategies.py
"""
Compensation Strategies (S1-S6)
================================
Six analytically-designed compensation strategies used as
heuristic baselines and RL agent actions.

All strategies are methods mixed into RadioNetwork via
CompensationStrategies mixin. Import pattern:

    from src.compensation.strategies import (
        CompensationStrategies)

    class RadioNetwork(CompensationStrategies, ...):
        pass
"""

import numpy as np
from typing import Optional


class CompensationStrategies:
    """
    Mixin class providing all 6 compensation strategies.
    Requires self to be a RadioNetwork instance.
    """

    # ── Strategy 1: Fixed nearest-neighbour power boost ───────────────

    def apply_power_compensation(
            self,
            failed_bs_id: int,
            boost_db: float = None,
            max_neighbours: int = None) -> None:
        """
        S1 — Fixed power boost on nearest neighbours.
        All selected neighbours boosted by the same amount.
        Simple baseline.
        """
        if boost_db is None:
            boost_db = self.cfg.MAX_POWER_BOOST_DB
        if max_neighbours is None:
            max_neighbours = self._auto_neighbour_count(
                failed_bs_id)
        neighbours = self.get_neighbours(
            failed_bs_id, max_neighbours)
        print(f"\n  Strategy 1 (Power): "
              f"+{boost_db:.1f}dB on "
              f"{len(neighbours)} neighbours")
        for bs in neighbours:
            bs.set_power(bs.nominal_power_dbm + boost_db)
            d = np.sqrt(
                (bs.x -
                 self.base_stations[failed_bs_id].x)**2 +
                (bs.y -
                 self.base_stations[failed_bs_id].y)**2)
            print(f"    BS-{bs.bs_id} (dist={d:.0f}m): "
                  f"{bs.nominal_power_dbm:.1f} → "
                  f"{bs.tx_power_dbm:.1f}dBm")

    # ── Strategy 2: Proportional power boost ──────────────────────────

    def apply_proportional_compensation(
            self,
            failed_bs_id: int,
            max_neighbours: int = None) -> None:
        """
        S2 — Boost proportional to proximity to outage UEs.
        Neighbour closest to outage zone gets full boost.
        """
        if max_neighbours is None:
            max_neighbours = self._auto_neighbour_count(
                failed_bs_id)
        cap        = self._power_cap(failed_bs_id)
        neighbours = self.get_neighbours(
            failed_bs_id, max_neighbours)
        outage_ues = [ue for ue in self.ues
                      if ue.in_outage]
        if not outage_ues:
            print("  No outage UEs."); return

        avg_dists = [
            np.mean([np.sqrt((ue.x - bs.x)**2 +
                              (ue.y - bs.y)**2)
                     for ue in outage_ues])
            for bs in neighbours]
        min_d = min(avg_dists)

        print(f"\n  Strategy 2 (Proportional Power): "
              f"{len(neighbours)} neighbours, "
              f"cap={cap:.1f}dB")
        for bs, avg_d in zip(neighbours, avg_dists):
            boost = cap * (min_d / avg_d)
            bs.set_power(bs.nominal_power_dbm + boost)
            d_f = np.sqrt(
                (bs.x -
                 self.base_stations[failed_bs_id].x)**2 +
                (bs.y -
                 self.base_stations[failed_bs_id].y)**2)
            print(f"    BS-{bs.bs_id} "
                  f"(dist={d_f:.0f}m, "
                  f"avg_to_outage={avg_d:.0f}m): "
                  f"{bs.nominal_power_dbm:.1f} → "
                  f"{bs.tx_power_dbm:.1f}dBm "
                  f"(+{boost:.2f}dB)")

    # ── Strategy 3: Best single neighbour power boost ─────────────────

    def apply_targeted_compensation(
            self,
            failed_bs_id: int,
            max_neighbours: int = None) -> None:
        """
        S3 — Single best neighbour power boost.
        Selects the neighbour that rescues the most outage
        UEs with the least required boost.
        Best power-only strategy for edge BS outage.
        """
        if max_neighbours is None:
            max_neighbours = self._auto_neighbour_count(
                failed_bs_id)
        cap          = self._power_cap(failed_bs_id)
        neighbours   = self.get_neighbours(
            failed_bs_id, max_neighbours)
        outage_ues   = [ue for ue in self.ues
                        if ue.in_outage]
        if not outage_ues:
            print("  No outage UEs."); return

        noise_linear = 10**(self.cfg.NOISE_POWER_DBM / 10)
        sinr_target  = 10**(
            self.cfg.SINR_OUTAGE_THRESHOLD_DB / 10)

        candidates = []
        for bs in neighbours:
            worst_boost = 0.0
            for ue in outage_ues:
                rx_nom = (bs.nominal_power_dbm -
                          self._pl(bs.x, bs.y,
                                   ue.x, ue.y,
                                   bs.nominal_tilt_deg))
                interf = self._frozen_interference(
                    ue, bs.bs_id)
                req    = 10 * np.log10(max(
                    sinr_target * (interf + noise_linear),
                    1e-15))
                worst_boost = max(worst_boost, req - rx_nom)

            actual  = min(max(worst_boost, 0.0), cap)
            rescued = sum(
                1 for ue in outage_ues
                if 10 * np.log10(max(
                    10**((bs.nominal_power_dbm + actual -
                          self._pl(bs.x, bs.y, ue.x, ue.y,
                                   bs.nominal_tilt_deg))
                         / 10) /
                    (self._frozen_interference(
                        ue, bs.bs_id) + noise_linear),
                    1e-15))
                >= self.cfg.SINR_OUTAGE_THRESHOLD_DB)
            candidates.append((bs, actual, rescued))

        candidates.sort(key=lambda x: (-x[2], x[1]))
        best_bs, best_boost, best_rescued = candidates[0]

        print(f"\n  Strategy 3 (Best Neighbour Power):")
        for bs, boost, rescued in candidates:
            d_f = np.sqrt(
                (bs.x -
                 self.base_stations[failed_bs_id].x)**2 +
                (bs.y -
                 self.base_stations[failed_bs_id].y)**2)
            print(f"    BS-{bs.bs_id} (dist={d_f:.0f}m): "
                  f"rescues {rescued}/"
                  f"{len(outage_ues)} "
                  f"needing +{boost:.2f}dB")
        print(f"  → Selecting BS-{best_bs.bs_id}")
        best_bs.set_power(
            best_bs.nominal_power_dbm + best_boost)

    # ── Strategy 4: Simultaneous frozen-snapshot boost ────────────────

    def apply_simultaneous_compensation(
            self,
            failed_bs_id: int,
            max_neighbours: int = None) -> None:
        """
        S4 — Simultaneous boost with frozen interference.
        Each outage UE assigned exclusively to nearest
        neighbour. Avoids sequential contamination.
        Best for centre BS outage with power only.
        """
        if max_neighbours is None:
            max_neighbours = self._auto_neighbour_count(
                failed_bs_id)
        cap        = self._power_cap(failed_bs_id)
        neighbours = self.get_neighbours(
            failed_bs_id, max_neighbours)
        outage_ues = [ue for ue in self.ues
                      if ue.in_outage]
        if not outage_ues:
            print("  No outage UEs."); return

        noise_linear = 10**(self.cfg.NOISE_POWER_DBM / 10)
        sinr_target  = 10**(
            self.cfg.SINR_OUTAGE_THRESHOLD_DB / 10)

        ue_assignments = {bs.bs_id: []
                          for bs in neighbours}
        for ue in outage_ues:
            nearest = min(
                neighbours,
                key=lambda bs: np.sqrt(
                    (ue.x - bs.x)**2 +
                    (ue.y - bs.y)**2))
            ue_assignments[nearest.bs_id].append(ue)

        print(f"\n  Strategy 4 (Simultaneous Power): "
              f"{len(neighbours)} neighbours, "
              f"{len(outage_ues)} outage UEs, "
              f"cap=+{cap:.1f}dB")

        boosts = {}
        for bs in neighbours:
            assigned = ue_assignments[bs.bs_id]
            d_f      = np.sqrt(
                (bs.x -
                 self.base_stations[failed_bs_id].x)**2 +
                (bs.y -
                 self.base_stations[failed_bs_id].y)**2)
            if not assigned:
                boosts[bs.bs_id] = 0.0
                print(f"    BS-{bs.bs_id} "
                      f"(dist={d_f:.0f}m): "
                      f"no UEs → no boost")
                continue

            worst_boost = 0.0
            for ue in assigned:
                rx_nom = (bs.nominal_power_dbm -
                          self._pl(bs.x, bs.y,
                                   ue.x, ue.y,
                                   bs.nominal_tilt_deg))
                interf = self._frozen_interference(
                    ue, bs.bs_id)
                req    = 10 * np.log10(max(
                    sinr_target * (interf + noise_linear),
                    1e-15))
                worst_boost = max(worst_boost,
                                   req - rx_nom)

            actual           = min(max(worst_boost, 0.0),
                                    cap)
            boosts[bs.bs_id] = actual
            print(f"    BS-{bs.bs_id} "
                  f"(dist={d_f:.0f}m, "
                  f"{len(assigned)} UEs): "
                  f"{bs.nominal_power_dbm:.1f} → "
                  f"{bs.nominal_power_dbm+actual:.1f}dBm "
                  f"(+{actual:.2f}dB)")

        for bs in neighbours:
            bs.set_power(bs.nominal_power_dbm +
                         boosts[bs.bs_id])

    # ── Strategy 5: Tilt reduction only ──────────────────────────────

    def apply_tilt_compensation(
            self,
            failed_bs_id: int,
            max_neighbours: int = None) -> None:
        """
        S5 — Antenna tilt reduction on nearest neighbours.

        Physically: reducing downtilt angles the beam upward
        and outward, extending coverage toward the outage zone
        without raising the interference floor uniformly.

        Best strategy for centre BS outage.
        Uses joint evaluation to prevent collateral outage.
        """
        if max_neighbours is None:
            max_neighbours = self._auto_neighbour_count(
                failed_bs_id)

        neighbours = self.get_neighbours(
            failed_bs_id, max_neighbours)
        outage_ues = [ue for ue in self.ues
                      if ue.in_outage]
        if not outage_ues:
            print("  No outage UEs."); return

        print(f"\n  Strategy 5 (Tilt Search — Joint Eval)"
              f": {len(neighbours)} neighbours, "
              f"{len(outage_ues)} outage UEs")
        print(f"  Tilt range: "
              f"{self.cfg.BS_TILT_MIN_DEG}° – "
              f"{self.cfg.BS_TILT_MAX_DEG}°")

        best_tilts  = {bs.bs_id: bs.nominal_tilt_deg
                       for bs in neighbours}
        best_powers = {bs.bs_id: bs.tx_power_dbm
                       for bs in neighbours}

        for bs in neighbours:
            d_f = np.sqrt(
                (bs.x -
                 self.base_stations[failed_bs_id].x)**2 +
                (bs.y -
                 self.base_stations[failed_bs_id].y)**2)
            best_tilt_this = bs.nominal_tilt_deg
            best_rescued   = 0

            for candidate_tilt in np.arange(
                    bs.nominal_tilt_deg,
                    self.cfg.BS_TILT_MIN_DEG - 0.5,
                    -1.0):
                proposed_tilts = dict(best_tilts)
                proposed_tilts[bs.bs_id] = candidate_tilt
                _, _, rescued, collateral = \
                    self._evaluate_network_state(
                        proposed_tilts, best_powers)
                if collateral > 0:
                    continue
                if rescued > best_rescued:
                    best_rescued   = rescued
                    best_tilt_this = candidate_tilt

            best_tilts[bs.bs_id] = best_tilt_this
            delta = bs.nominal_tilt_deg - best_tilt_this
            print(f"    BS-{bs.bs_id} "
                  f"(dist={d_f:.0f}m): "
                  f"tilt {bs.nominal_tilt_deg:.1f}° → "
                  f"{best_tilt_this:.1f}° "
                  f"(-{delta:.1f}°) "
                  f"rescues {best_rescued}/"
                  f"{len(outage_ues)} "
                  f"(joint eval)")

        _, n_out, n_rescued, n_coll = \
            self._evaluate_network_state(
                best_tilts, best_powers)
        print(f"\n  Joint evaluation: "
              f"rescued={n_rescued}, "
              f"collateral={n_coll}, "
              f"total outage after={n_out}")
        print(f"  Applying tilts:")
        for bs in neighbours:
            old_tilt = bs.tilt_deg
            bs.set_tilt(best_tilts[bs.bs_id])
            print(f"    BS-{bs.bs_id}: "
                  f"{old_tilt:.1f}° → "
                  f"{bs.tilt_deg:.1f}°")

    # ── Strategy 6: Joint power boost + tilt reduction ────────────────

    def apply_joint_compensation(
            self,
            failed_bs_id: int,
            max_neighbours: int = None) -> None:
        """
        S6 — Joint power boost AND tilt reduction.

        Most sophisticated strategy:
        1. Search for best safe tilt (joint evaluation)
        2. Compute required power boost at that tilt
        3. Apply power boost ONLY if it rescues more UEs
           than tilt alone AND causes zero collateral outage
        4. Apply tilt and power simultaneously

        Falls back to tilt-only when power boost causes
        collateral outage — always safe.
        """
        if max_neighbours is None:
            max_neighbours = self._auto_neighbour_count(
                failed_bs_id)

        cap         = self._power_cap(failed_bs_id)
        neighbours  = self.get_neighbours(
            failed_bs_id, max_neighbours)
        outage_ues  = [ue for ue in self.ues
                       if ue.in_outage]
        if not outage_ues:
            print("  No outage UEs."); return

        noise_linear = 10**(self.cfg.NOISE_POWER_DBM / 10)
        sinr_target  = 10**(
            self.cfg.SINR_OUTAGE_THRESHOLD_DB / 10)

        ue_assignments = {bs.bs_id: []
                          for bs in neighbours}
        for ue in outage_ues:
            nearest = min(
                neighbours,
                key=lambda bs: np.sqrt(
                    (ue.x - bs.x)**2 +
                    (ue.y - bs.y)**2))
            ue_assignments[nearest.bs_id].append(ue)

        print(f"\n  Strategy 6 (Joint Power + Tilt — "
              f"Joint Eval): "
              f"{len(neighbours)} neighbours, "
              f"{len(outage_ues)} outage UEs, "
              f"power cap=+{cap:.1f}dB")

        # Step 1: Find best safe tilt per neighbour
        best_tilts  = {bs.bs_id: bs.nominal_tilt_deg
                       for bs in neighbours}
        best_powers = {bs.bs_id: bs.tx_power_dbm
                       for bs in neighbours}

        print(f"\n  Step 1: Tilt search (joint evaluation)")
        for bs in neighbours:
            d_f = np.sqrt(
                (bs.x -
                 self.base_stations[failed_bs_id].x)**2 +
                (bs.y -
                 self.base_stations[failed_bs_id].y)**2)
            best_tilt_this = bs.nominal_tilt_deg
            best_rescued   = 0

            for candidate_tilt in np.arange(
                    bs.nominal_tilt_deg,
                    self.cfg.BS_TILT_MIN_DEG - 0.5,
                    -1.0):
                proposed_tilts = dict(best_tilts)
                proposed_tilts[bs.bs_id] = candidate_tilt
                _, _, rescued, collateral = \
                    self._evaluate_network_state(
                        proposed_tilts, best_powers)
                if collateral > 0:
                    continue
                if rescued > best_rescued:
                    best_rescued   = rescued
                    best_tilt_this = candidate_tilt

            best_tilts[bs.bs_id] = best_tilt_this
            delta = bs.nominal_tilt_deg - best_tilt_this
            print(f"    BS-{bs.bs_id} (dist={d_f:.0f}m): "
                  f"tilt {bs.nominal_tilt_deg:.1f}° → "
                  f"{best_tilt_this:.1f}° "
                  f"(-{delta:.1f}°) "
                  f"rescues {best_rescued}/"
                  f"{len(outage_ues)}")

        # Step 2: Compute power boost at new tilts
        print(f"\n  Step 2: Power boost (at new tilts)")
        boosts = {}
        for bs in neighbours:
            assigned = ue_assignments[bs.bs_id]
            new_tilt = best_tilts[bs.bs_id]
            d_f      = np.sqrt(
                (bs.x -
                 self.base_stations[failed_bs_id].x)**2 +
                (bs.y -
                 self.base_stations[failed_bs_id].y)**2)

            if not assigned:
                boosts[bs.bs_id] = 0.0
                print(f"    BS-{bs.bs_id}: "
                      f"no UEs assigned → no boost")
                continue

            worst_boost = 0.0
            for ue in assigned:
                d2d          = self._distance(ue, bs)
                pl           = self._path_loss_db(d2d)
                ant_gain_new = self._antenna_gain_db(
                    d2d, new_tilt)
                rx_at_tilt   = (bs.nominal_power_dbm -
                                 pl + ant_gain_new)
                interf       = self._frozen_interference(
                    ue, bs.bs_id)
                req          = 10 * np.log10(max(
                    sinr_target * (interf + noise_linear),
                    1e-15))
                worst_boost  = max(worst_boost,
                                    req - rx_at_tilt)

            actual_boost     = min(max(worst_boost, 0.0),
                                    cap)
            boosts[bs.bs_id] = actual_boost
            print(f"    BS-{bs.bs_id} "
                  f"({len(assigned)} UEs): "
                  f"needs +{worst_boost:.2f}dB → "
                  f"applied +{actual_boost:.2f}dB")

        # Step 3: Compare tilt-only vs tilt+power
        proposed_powers_joint = {
            bs.bs_id: bs.nominal_power_dbm +
                      boosts[bs.bs_id]
            for bs in neighbours}
        proposed_powers_tilt  = {
            bs.bs_id: bs.tx_power_dbm
            for bs in neighbours}

        _, n_out_j, n_resc_j, n_coll_j = \
            self._evaluate_network_state(
                best_tilts, proposed_powers_joint)
        _, n_out_t, n_resc_t, n_coll_t = \
            self._evaluate_network_state(
                best_tilts, proposed_powers_tilt)

        print(f"\n  Step 3: Joint evaluation:")
        print(f"    Tilt only    — rescued={n_resc_t}, "
              f"collateral={n_coll_t}, "
              f"outage={n_out_t}")
        print(f"    Tilt + power — rescued={n_resc_j}, "
              f"collateral={n_coll_j}, "
              f"outage={n_out_j}")

        use_power = (n_coll_j == 0 and
                     n_resc_j > n_resc_t)

        if use_power:
            final_powers = proposed_powers_joint
            print(f"  → Decision: tilt + power boost")
        else:
            final_powers  = proposed_powers_tilt
            boosts        = {bs.bs_id: 0.0
                             for bs in neighbours}
            reason        = ("power causes collateral"
                             if n_coll_j > 0
                             else "no extra rescue")
            print(f"  → Decision: tilt only "
                  f"({reason})")

        # Step 4: Apply simultaneously
        print(f"\n  Applying final actions:")
        for bs in neighbours:
            old_tilt  = bs.nominal_tilt_deg
            new_tilt  = best_tilts[bs.bs_id]
            new_power = final_powers[bs.bs_id]
            bs.set_tilt(new_tilt)
            bs.set_power(new_power)
            d_f = np.sqrt(
                (bs.x -
                 self.base_stations[failed_bs_id].x)**2 +
                (bs.y -
                 self.base_stations[failed_bs_id].y)**2)
            print(f"    BS-{bs.bs_id} (dist={d_f:.0f}m): "
                  f"tilt {old_tilt:.1f}° → "
                  f"{new_tilt:.1f}° | "
                  f"power {bs.nominal_power_dbm:.1f} → "
                  f"{new_power:.1f}dBm")
