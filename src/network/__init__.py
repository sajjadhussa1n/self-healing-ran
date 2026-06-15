# src/network/__init__.py
"""
RadioNetwork with CompensationStrategies mixin.
The final RadioNetwork class inherits from both
to keep network simulation and compensation logic
in separate, maintainable files.
"""

from .base_station import BaseStation
from .user_equipment import UserEquipment
from .radio_network import RadioNetwork as _RadioNetworkBase
from src.compensation.strategies import CompensationStrategies


class RadioNetwork(CompensationStrategies,
                   _RadioNetworkBase):
    """
    Complete RadioNetwork with all 6 compensation
    strategies (S1-S6) mixed in.
    """
    pass


__all__ = ['BaseStation', 'UserEquipment', 'RadioNetwork']
