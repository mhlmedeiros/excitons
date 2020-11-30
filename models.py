import numpy as np

"""
In this module we have all the Hamiltonian models of interest, also the
physical constants and the effective potential for the  excitons.

List of function/classes:


"""

EPSILON_0 = 55.26349406             # e^2 GeV^{-1}fm^{-1} == e^2 (1e9 eV 1e-15 m)^{-1}
HBAR = 1.23984193/(2*np.pi)         # eV 1e-6 m/c
M_0  = 0.51099895000                # MeV/c^2
hbar2_over2m = HBAR**2/(2*M_0)*1e3  # meV nm^2

# ============================================================================= #
##                              Hamiltonian models:
# ============================================================================= #
def hamiltonian_2x2(kx, ky, E_gap=0.5, Gamma=1, Alpha_c=1, Alpha_v=-1):
    """
    Simple Hamiltonian to test the implementation:

    In its definition we have the effective masses "m_e" and "m_h",
    we also have the energy "gap". The model include one conduction-band
    and one valence-band, the couplings between these states are
    mediated by the named parameter "gamma".

    """
    k2 = kx**2 + ky**2
    H = np.array([[E_gap + hbar2_over2m * Alpha_c * k2, Gamma*(kx+1j*ky)],
                  [Gamma*(kx-1j*ky), hbar2_over2m * Alpha_v * k2]])
    return H


# ============================================================================= #
##                              Exciton potential:
# ============================================================================= #
def calculate_distance_k_pontual(k1_ind, k2_ind, kx_flat, ky_flat):
    """
    Like the version used in the Wannier this function calculates the "distance" between two points
    in the reciprocal space. The difference is that here it just returns one value instead of
    the whole matrix with all possible pairs' distances.
    """
    dist = np.sqrt((kx_flat[k1_ind]-kx_flat[k2_ind])**2 + (ky_flat[k1_ind]-ky_flat[k2_ind])**2)
    return dist


def rytova_keldysh_pontual(k1_ind, k2_ind, kx_flat, ky_flat, dk2, epsilon=2, r_0=4.51):
    """
    The "pontual" version of the that one in Wannier script. Instead of return the whole matrix
    this function returns only the value asked.
    """
    q = calculate_distance_k_pontual(k1_ind, k2_ind, kx_flat, ky_flat)
    Vkk_const = 1e6/(2*EPSILON_0)
    V =  1/(epsilon*q + r_0*q**2)
    return - Vkk_const * dk2/(2*np.pi)**2 * V
