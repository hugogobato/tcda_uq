"""Datasets: synthetic generators + real-data loaders.

  * :mod:`.orbit`       -- ORBIT linked-twist-map testbed (numpy port).
  * :mod:`.simulation`  -- tri-oracle harness exposing TATE/CTATE/ITTE truth (Level A).
  * :mod:`.topological` -- point-cloud causal DGP with X,A-controlled homology (Level B, Phase 6/7A.2).
  * :mod:`.sarscov2`    -- SARS-CoV-2 CT loader (needs the ``[data]`` extra).
"""

from .orbit import gen_orbits, make_orbit_causal
from .simulation import TriOracleSimulation, SimulationSample
from .topological import TopologicalCausalSimulation, TopologicalSample
from .covariates import gen_covariate, gen_trt_prob

__all__ = [
    "gen_orbits",
    "make_orbit_causal",
    "TriOracleSimulation",
    "SimulationSample",
    "TopologicalCausalSimulation",
    "TopologicalSample",
    "gen_covariate",
    "gen_trt_prob",
]
