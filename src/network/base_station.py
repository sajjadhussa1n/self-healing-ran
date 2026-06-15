# src/network/base_station.py
"""
BaseStation class — represents a single BS in the network.
"""

import numpy as np
from src.config import CFG


class BaseStation:
    """
    Represents a single base station.

    Attributes:
        bs_id            : unique identifier
        x, y             : position in metres
        tx_power_dbm     : current TX power (dBm)
        nominal_power_dbm: original TX power (dBm)
        tilt_deg         : current electrical downtilt
        nominal_tilt_deg : original tilt
        is_active        : False if BS is in outage
        color            : matplotlib colour for plotting
    """

    def __init__(self, bs_id: int, x: float, y: float,
                 tx_power_dbm: float = None):
        self.bs_id             = bs_id
        self.x                 = x
        self.y                 = y
        self.tx_power_dbm      = (tx_power_dbm
                                   if tx_power_dbm is not None
                                   else CFG.BS_TX_POWER_DBM)
        self.nominal_power_dbm = self.tx_power_dbm
        self.tilt_deg          = CFG.BS_TILT_DEFAULT_DEG
        self.nominal_tilt_deg  = CFG.BS_TILT_DEFAULT_DEG
        self.is_active         = True
        self.color             = None

    def set_power(self, power_dbm: float) -> None:
        """Set TX power in dBm."""
        self.tx_power_dbm = power_dbm

    def set_tilt(self, tilt_deg: float) -> None:
        """Set electrical downtilt, clamped to valid range."""
        self.tilt_deg = float(np.clip(
            tilt_deg,
            CFG.BS_TILT_MIN_DEG,
            CFG.BS_TILT_MAX_DEG))

    def toggle_active(self, active: bool) -> None:
        """Activate or deactivate BS."""
        self.is_active = active
        if not active:
            self.tx_power_dbm = -np.inf

    def reset(self) -> None:
        """Restore BS to nominal power and tilt."""
        if self.is_active:
            self.tx_power_dbm = self.nominal_power_dbm
            self.tilt_deg     = self.nominal_tilt_deg

    def __repr__(self) -> str:
        return (f"BS-{self.bs_id} @ ({self.x:.0f},"
                f"{self.y:.0f}) "
                f"Pwr={self.tx_power_dbm:.1f}dBm "
                f"Tilt={self.tilt_deg:.1f}° "
                f"Active={self.is_active}")
