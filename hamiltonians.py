

import numpy as np
import numpy.linalg as LA
from numba import jit, njit, int32, float32
from numba.experimental import jitclass

EPSILON_0 = 55.26349406             # e^2 GeV^{-1}fm^{-1} == e^2 (1e9 eV 1e-15 m)^{-1}
HBAR = 1.23984193/(2*np.pi)         # eV 1e-6 m/c
M_0  = 0.51099895000                # MeV/c^2
hbar2_over2m = HBAR**2/(2*M_0)*1e3  # meV nm^2

#===============================================================================
# NOTE THAT WE CANNOT INHERITATE FROM A "jitclass".
# SO, AN WORKAROUND IS TO DEFINE A PYTHON CLASS THAT BEHAVES LIKE A
# ABSTRACT CLASS. SUCH CLASS IS NOT ALLOWED TO INSTANTIATE OBJECTS
# BUT IT CAN SERVE AS PARENT OF 'jitclasses'.
#===============================================================================

fields2x2 = [
    ('alphac', float32),
    ('alphav', float32),
    ('gap', float32),
    ('gamma', float32),
    ('condBands', int32),
    ('valeBands', int32),
]

@jitclass(fields2x2)
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

#===============================================================================
fields4x4 = [
    ('alphac_up', float32),
    ('alphav_up', float32),
    ('gap_up', float32),
    ('gamma_up', float32),
    ('alphac_down', float32),
    ('alphav_down', float32),
    ('gap_down', float32),
    ('gamma_down', float32),
    ('condBands', int32),
    ('valeBands', int32),
]

class H4x4_general:
    """
    The instances of this class cannot be passed to numba-compiled functions.
    But, since it is a Python class it can be parent of other classes, it
    includes 'jitclasses'.
    """
    def __init__(self, alphac_up, alphav_up, gap_up, gamma_up, alphac_down, alphav_down, gap_down, gamma_down):
        ## SPIN-UP:
        self.alphac_up = alphac_up
        self.alphav_up = alphav_up
        self.gap_up = gap_up
        self.gamma_up = gamma_up
        ## SPIN-DOWN:
        self.alphac_down = alphac_down
        self.alphav_down = alphav_down
        self.gap_down = gap_down
        self.gamma_down = gamma_down
        ## HAMILTONIAN:
        self.valeBands = 2
        self.condBands = 2

    def call(self, kx, ky):
        Eg_up, Eg_down = self.gap_up, self.gap_down
        alpha_c_up, alpha_c_down = self.alphac_up, self.alphac_down
        alpha_v_up, alpha_v_down = self.alphav_up, self.alphav_down
        gamma_up, gamma_down = self.gamma_up, self.gamma_down
        k2 = kx**2 + ky**2
        H = np.array([
            [Eg_up + hbar2_over2m * alpha_c_up * k2, gamma_up*(kx+1j*ky), 0 , 0],
            [gamma_up*(kx-1j*ky), hbar2_over2m * alpha_v_up * k2, 0, 0],
            [0, 0, Eg_down + hbar2_over2m * alpha_c_down * k2, gamma_down*(kx+1j*ky)],
            [0, 0, gamma_down*(kx-1j*ky), hbar2_over2m * alpha_v_down * k2]])
        return H

@jitclass(fields4x4)
class H4x4(H4x4_general):
    """
    This is the instantiable version of "H4x4_general".
    Like happens in Julia: this is the leaf of the inheritance tree.
    """
    pass

@jitclass(fields4x4)
class H4x4_equal(H4x4_general):
    def __init__(self, alphac, alphav, gap, gamma):
        ## SPIN-UP:
        self.alphac_up = alphac
        self.alphav_up = alphav
        self.gap_up = gap
        self.gamma_up = gamma
        ## SPIN-DOWN:
        self.alphac_down = alphac
        self.alphav_down = alphav
        self.gap_down = gap
        self.gamma_down = gamma
        ## HAMILTONIAN:
        self.valeBands = 2
        self.condBands = 2




    # def call(self, kx, ky):
    #     Eg_up, Eg_down = self.gap_up, self.gap_down
    #     alpha_c_up, alpha_c_down = self.alphac_up, self.alphac_down
    #     alpha_v_up, alpha_v_down = self.alphav_up, self.alphav_down
    #     gamma_up, gamma_down = self.gamma_up, self.gamma_down
    #     k2 = kx**2 + ky**2
    #     H = np.array([
    #         [Eg_up + hbar2_over2m * alpha_c_up * k2, gamma_up*(kx+1j*ky), 0 , 0],
    #         [gamma_up*(kx-1j*ky), hbar2_over2m * alpha_v_up * k2, 0, 0],
    #         [0, 0, Eg_down + hbar2_over2m * alpha_c_down * k2, gamma_down*(kx+1j*ky)],
    #         [0, 0, gamma_down*(kx-1j*ky), hbar2_over2m * alpha_v_down * k2]])
    #     return H



    # def split(self,vectors):
    #     ## Revert the order (of valence bands: index [0] -> closer to the gap)
    #     valence_vectors = vectors[:,:self.valeBands]
    #     conduction_vectors = vectors[:,::-1]
    #     ## All the remaining vectors are conduction states
    #     conduct_vectors = vectors[:,self.valeBands:]
    #     return valence_vectors, conduct_vectors
    #
    # def delta(self, c1, v1, c2, v2, k1=(0,0), k2=(0,0)):
    #     _, vectors_1 = LA.eigh(self.call(*k1))
    #     _, vectors_2 = LA.eigh(self.call(*k2))
    #     vectors_1_v, vectors_1_c = self.split(vectors_1)
    #     vectors_2_v, vectors_2_c = self.split(vectors_2)
    #     cond_1 = vectors_1_c[:,c1].conjugate()
    #     cond_2 = vectors_2_c[:,c2]
    #     vale_1 = vectors_1_v[:,v1]
    #     vale_2 = vectors_2_v[:,v2].conjugate()
    #     # The operators "@" and "np.dot()" return a warning
    #     return np.sum(cond_1*cond_2) * np.sum(vale_2*vale_1)

@njit
def eig_vals(M):
    return LA.eigvals(M.call(0,0))

def main():
    alphac = 0
    alphav = 0
    E_gap = 2.4e3 # meV ~ 2.4 eV
    gamma = 2.6e2 # meV*nm ~ 2.6 eV*AA
    H = H4x4(alphac, alphav, E_gap, gamma, alphac, alphav, E_gap, gamma)
    print(H.call(0,0))
    print(eig_vals(H))



if __name__=='__main__':
    main()
