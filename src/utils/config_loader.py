"""
Loads pipeline_config.yaml and applies its values onto a
SimConfig instance and a flat dict of pipeline settings,
so main.py and other entry points never hardcode parameters.
"""
import yaml
from src.config import SimConfig


def load_pipeline_config(path="pipeline_config.yaml"):
    """
    Parameters
    ----------
    path : str
        Path to the YAML pipeline config file.

    Returns
    -------
    sim_config : SimConfig
        SimConfig instance with overrides from the
        'network' section applied as attributes.
    pipeline_cfg : dict
        Raw parsed YAML (everything else: demo, cod,
        agents, evaluation, paths sections), used directly
        by main.py for orchestration parameters.
    """
    with open(path, "r") as f:
        raw_cfg = yaml.safe_load(f)

    sim_config = SimConfig()
    for key, value in raw_cfg.get("network", {}).items():
        setattr(sim_config, key, value)

    return sim_config, raw_cfg
