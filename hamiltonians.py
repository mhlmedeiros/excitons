

import numpy as np
import numpy.linalg as LA
from numba import njit, int32, float32
from numba.experimental import jitclass

EPSILON_0 = 55.26349406             # e^2 GeV^{-1}fm^{-1} == e^2 (1e9 eV 1e-15 m)^{-1}
HBAR = 1.23984193/(2*np.pi)         # eV 1e-6 m/c
M_0  = 0.51099895000                # MeV/c^2
hbar2_over2m = HBAR**2/(2*M_0)*1e3  # meV nm^2


fields = [
    ('alphac', float32),
    ('alphav', float32),
    ('gap', float32),
    ('gamma', float32),
    ('condBands', int32),
    ('valeBands', int32),
]

@jitclass(fields)
class H2x2:
    def __init__(self, alphac, alphav, gap, gamma):
        self.alphac = alphac
        self.alphav = alphav
        self.gap = gap
        self.gamma = gamma
        self.valeBands = 1
        self.condBands = 1

    def call(self, kx, ky):
        E_gap = self.gap
        Alpha_c = self.alphac
        Alpha_v = self.alphav
        Gamma = self.gamma
        k2 = kx**2 + ky**2
        H = np.array([
            [E_gap + hbar2_over2m * Alpha_c * k2, Gamma*(kx+1j*ky)],
            [Gamma*(kx-1j*ky), hbar2_over2m * Alpha_v * k2]])
        return H

    def split(self,vectors):
        ## Revert the order (of valence bands: index [0] -> closer to the gap)
        valence_vectors = vectors[:,:self.valeBands]
        valence_vectors = valence_vectors[:,::-1]
        ## All the remaining vectors are conduction states
        conduct_vectors = vectors[:,self.valeBands:]
        return valence_vectors, conduct_vectors

    def delta(self, c1, v1, c2, v2, k1=(0,0), k2=(0,0)):
        _, vectors_1 = LA.eigh(self.call(*k1))
        _, vectors_2 = LA.eigh(self.call(*k2))
        vectors_1_v, vectors_1_c = self.split(vectors_1)
        vectors_2_v, vectors_2_c = self.split(vectors_2)
        cond_1 = vectors_1_c[:,c1].conjugate()
        cond_2 = vectors_2_c[:,c2]
        vale_1 = vectors_1_v[:,v1]
        vale_2 = vectors_2_v[:,v2].conjugate()
        # The operators "@" and "np.dot()" return a warning 
        return np.sum(cond_1*cond_2) * np.sum(vale_2*vale_1)
