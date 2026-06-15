# src/network/user_equipment.py
"""
UserEquipment class — represents a single UE.
"""


class UserEquipment:
    """
    Represents a single User Equipment (UE / mobile device).

    Attributes:
        ue_id              : unique identifier
        x, y               : position in metres
        serving_bs_id      : ID of serving BS (None if unserved)
        received_power_dbm : Rx power from serving BS (dBm)
        sinr_db            : SINR (dB)
        throughput_mbps    : Shannon throughput (Mbps)
        in_outage          : True if below outage threshold
    """

    def __init__(self, ue_id: int, x: float, y: float):
        self.ue_id              = ue_id
        self.x                  = x
        self.y                  = y
        self.serving_bs_id      = None
        self.received_power_dbm = None
        self.sinr_db            = None
        self.throughput_mbps    = 0.0
        self.in_outage          = False

    def __repr__(self) -> str:
        sinr_str = (f"{self.sinr_db:.1f}dB"
                    if self.sinr_db is not None
                    else "N/A")
        return (f"UE-{self.ue_id} @ ({self.x:.0f},"
                f"{self.y:.0f}) "
                f"BS={self.serving_bs_id} "
                f"SINR={sinr_str}")
