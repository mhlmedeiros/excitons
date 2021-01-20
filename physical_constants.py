import numpy as np

## RECORRENT PHYSICAL CONSTANTS FOR BSE-CODE AND ABSORPTION-CODE:
EPSILON_0       = 55.26349406               # e^2 GeV^{-1}fm^{-1} == e^2 (1e9 eV 1e-15 m)^{-1}
HBAR            = 1.23984193/(2*np.pi)      # eV 1e-6 m/c
M_0             = 0.51099895000             # MeV/c^2
hbar2_over2m    = HBAR**2/(2*M_0)*1e3       # meV nm^2
