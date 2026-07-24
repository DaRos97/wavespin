""" Bogoliubov diagonalization, correlator computation, decay rates, and momentum
transformations for both open and periodic boundary conditions.
"""

from wavespin.static.open import openHamiltonian, openCorrelators
from wavespin.static.periodic import (
    periodicHamiltonian,
    periodicRamp,
    quantizationAxis,
    computePs,
    computeTs,
)
