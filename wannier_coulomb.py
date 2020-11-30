import time
import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt

EPSILON_0 = 55.26349406 # e^2 GeV^{-1}fm^{-1}

def st_time(func):
    """
    st decorator to calculate the total time of a func
    """
    def st_func(*args, **keyArgs):
        t1 = time.time()
        r = func(*args, **keyArgs)
        t2 = time.time()
        print("Function=%s, Time=%s" % (func.__name__, t2 - t1))
        return r
    return st_func

def calculate_distances_k(kx_matrix, ky_matrix):
    """
    This function gets the Nk x Nk meshgrid matrices and populates a
    matrix with the modules of the distances in k-space:
    +--------------+
    |    |k-k'|    |
    +--------------+
    Such matrix is crucial to calculate the potential matrix.
    Note that we're adopting the order 'C' (row-major) to flat
    the matrices.
    """
    kx_vec = kx_matrix.flatten()
    ky_vec = ky_matrix.flatten()
    L = len(kx_vec)
    subtract_matrix = np.zeros((L,L)) # initialize the matrix

    ## Populates the upper triangular half of the matrix:
    for n in range(0, L-1):
        for m in range(n+1, L):
            subtract_matrix[n,m] = np.sqrt((kx_vec[n]-kx_vec[m])**2 + (ky_vec[n]-ky_vec[m])**2)

    # The next step is to fill in the bottom triangular half in a symmetric fashion:
    subtract_matrix += subtract_matrix.T

    return subtract_matrix

def kinetic_wannier(kx_matrix, ky_matrix):
    """
    Diagonal matrix with the kinetic contribution of the energy.
    Note that some important constants are defined here that may be
    usefull for other functions. Such constants define the units
    of energy and wave number:

    [E] = meV
    [L^-1] = nm^-1

    """
    m0 = 0.510 # MeV/c^2
    hbar = 1.23984193/(2*np.pi) # eV 1e-6 m/c
    mu = 0.16 * m0
    Tk_const = 1e3 * hbar**2/(2*mu) # meV nm^2

    k_sqrd = kx_matrix**2 + ky_matrix**2
    Tk = Tk_const * np.diagflat(k_sqrd)

    return Tk

def define_grid_k(size_pos, n_points):
    """
    "whole interval" = [-size_pos, +size_pos]

    sizes_positive: final of the interval of wave number; unit = nm^{-1}

    Nk: number of discrete points in the whole interval.
    """
    k = np.linspace(-size_pos, size_pos, n_points)
    delta_k = k[1] - k[0]
    delta_k_sqrd = delta_k**2
    kx_matrix, ky_matrix = np.meshgrid(k,k)
    return kx_matrix, ky_matrix, delta_k_sqrd

def coulomb_wannier(kx_matrix, ky_matrix, dk2):
    """
    Symmetric matrix with the potential contribution of the energy.
    The adopted model here is the Coulomb potential.
    Note that some important constants are defined here that may be
    usefull for other functions. Such constants define the units
    of energy and wave number:

    [E] = meV
    [L^-1] = nm^-1

    """
    ## Definitions (units)
    epsilon_r = 10
    Vkk_const = 1e6/(2 * epsilon_r * EPSILON_0) # meV nm

    M_distances = calculate_distances_k(kx_matrix, ky_matrix)
    n,_ = M_distances.shape
    aux_eye = np.eye(n)
    V = 1/(M_distances + aux_eye)
    V -= aux_eye
    return - Vkk_const * dk2/(2*np.pi)**2 *  V

@st_time
def main():
    ## Define the sizes of the region in k-space to be investigated:
    min_size = 1 # nm^-1
    max_size = 5 # nm^-1
    L_values = range(min_size, max_size + 1) # [min (min+1) ... max] # nm^-1
    list(L_values)

    ## Choose the number of discrete points will used to investigate the convergence:
    min_points = 121
    max_points = 121
    n_points = list(range(min_points, max_points+1, 10)) # [11 21 31 ... 101]
    # n_points = [min_points]

    file_name_metadata = "../Data/info_wannier_conv_and_wave_funct_121"
    np.savez(file_name_metadata, L_values=L_values,
                                 n_points=n_points)

    ## Matrices to hold the eigenvalues and the eigenvectors:
    ##
    ## For each value of discrete points adopted and also for each size for
    ## the system we have to save at least 4 states (1s,2p1,2p2,2s).
    ## The ordering may change depending on the discretization and system size.
    ##
    eigvals_holder = np.zeros((4, len(n_points), len(L_values)))
    eigvecs_holder = np.zeros((max_points**2, 4, len(L_values)))

    for ind_L in range(len(L_values)):
        print("\nCalculating the system with {} Angstroms.".format(L_values[ind_L]))
        for ind_Nk in range(len(n_points)):
            print("Nk: {}".format(n_points[ind_Nk]))
            Kx, Ky, dk2 = define_grid_k(L_values[ind_L], n_points[ind_Nk])
            Wannier_matrix = kinetic_wannier(Kx, Ky) + coulomb_wannier(Kx, Ky, dk2)
            values, vectors = LA.eigh(Wannier_matrix)
            # SAVE THE FIRST 2 EIGENVALUES
            eigvals_holder[:, ind_Nk, ind_L] = values[:4]
        # SAVE THE VECTORS
        eigvecs_holder[:, :, ind_L] = vectors[:,:4]


    # SAVE MATRICES WITH THE RESULTS
    print("\n\nSaving...")
    file_name = "../Data/data_wannier_conv_and_wave_funct_121"

    np.savez(file_name, eigvals_holder=eigvals_holder,
                        eigvecs_holder=eigvecs_holder)

    print("Done!")

"""
Notes about performance:

    - For 5 different systems, each of which with 111 x 111 discrete points it lasts for 46 min;

    - For 5 different systems, with 121 x 121 discrete points each it took 67 min;
"""


if __name__=='__main__':
    main()
