import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
from numba import njit
from wannier_coulomb_numba import st_time, define_grid_k, calculate_distances_k, kinetic_wannier
from wannier_coulomb_numba import EPSILON_0, M_0

@njit
def rytova_keldysh(kx_matrix, ky_matrix, dk2, epsilon=2, r_0=4.51):
    M_distances = calculate_distances_k(kx_matrix, ky_matrix)
    n,_ = M_distances.shape
    aux_eye = np.eye(n)
    Vkk_const = 1e6/(2*EPSILON_0)
    V =  1/(epsilon*M_distances + r_0*M_distances**2 + aux_eye)
    # V =  1/(M_distances + aux_eye)
    V -= aux_eye
    return - Vkk_const * dk2/(2*np.pi)**2 * V

def effective_rytova_keldysh_1D(k, dk2, epsilon=2, r_0=4.51):
    """
    This is the generic
    """
    arg_aux = r_0/epsilon * k
    Vkk_const = 1e6/(2*EPSILON_0)
    V = (r_0/epsilon**2 * (1/(arg_aux**2) +
        np.pi/(2*arg_aux) * (sp.yv(1,arg_aux)+sp.struve(-1,arg_aux))-
        (np.exp(1)-np.log(2))/(np.sqrt(1+arg_aux))**3))
    return -Vkk_const * dk2/(2*np.pi)**2 * V

def effective_rytova_keldysh_potential_op(kx_matrix, ky_matrix, dk2, epsilon, r_0):
    """
    This function returns a matrix with that contains the effective Rytova-Keldysh potential
    in k-space. The matrix elements are in the correct order to build up the Hamiltonian.
    """
    M_distances = calculate_distances_k(kx_matrix, ky_matrix)
    n,_ = M_distances.shape
    aux_eye = np.eye(n)
    M_distances_safer = M_distances + aux_eye
    V = effective_rytova_keldysh_1D(M_distances_safer,dk2, epsilon, r_0)
    V -= np.diagflat(np.diag(V))
    return V

def first_call_to_compile_RK_potential():
    ## Just to do the first compilation:
    size_compilation = 5
    n_points_compilation = 11
    Kx, Ky, dk2 = define_grid_k(size_compilation, n_points_compilation)
    rytova_keldysh(Kx, Ky, dk2)
    return 0


@st_time
def main():
    # ============================================================================= #
    ##                              Outuput options:
    # ============================================================================= #
    save = False
    preview = True

    # ============================================================================= #
    ##                      Effective mass and confinement:
    # ============================================================================= #
    mc = 0.2834
    mv = -0.3636
    m_eff = 1/(1/mc - 1/mv) * M_0
    r0_WSe2 = 4.51
    epsilon_eff = 1
    # m_eff / M_0

    # ============================================================================ #
    ## Define the sizes of the region in k-space to be investigated:
    # ============================================================================ #
    min_size = 1 # nm^-1
    max_size = 1 # nm^-1
    L_values = range(min_size, max_size + 1) # [min (min+1) ... max] # nm^-1
    list(L_values)

    # ============================================================================ #
    ## Choose the number of discrete points to investigate the convergence:
    # ============================================================================ #
    min_points = 41
    max_points = 41
    n_points = list(range(min_points, max_points+1, 10)) # [11 21 ... 111]


    # ============================================================================ #
    ##              Matrices to hold the eigenvalues and the eigenvectors:
    # ============================================================================ #
    number_of_recorded_states = 10
    eigvals_holder = np.zeros((number_of_recorded_states, len(n_points), len(L_values)))
    eigvecs_holder = np.zeros((max_points**2, number_of_recorded_states, len(L_values)))

    first_call_to_compile_RK_potential() ## Just to pre-compilate: doesn't return anything

    # ============================================================================ #
    ##                                  main_loop
    # ============================================================================ #
    for ind_L in range(len(L_values)):
        print("\nCalculating the system with {} nm^(-1).".format(L_values[ind_L]))
        for ind_Nk in range(len(n_points)):
            print("Discretization: {}".format(n_points[ind_Nk]))
            Kx, Ky, dk2 = define_grid_k(L_values[ind_L], n_points[ind_Nk])
            Tk = kinetic_wannier(Kx, Ky, mu=m_eff)
            Vk = rytova_keldysh(Kx, Ky, dk2, epsilon=epsilon_eff,r_0=r0_WSe2)
            Wannier_matrix = Tk + Vk
            values, vectors = LA.eigh(Wannier_matrix)
            # SAVE THE FIRST STATES ("number_of_recorded_states"):
            eigvals_holder[:, ind_Nk, ind_L] = values[:number_of_recorded_states]
        # SAVE THE VECTORS WITH THE FINEST DISCRETIZATION:
        eigvecs_holder[:, :, ind_L] = vectors[:,:number_of_recorded_states]


    if save:
        print("\n\nSaving...")
        # ======================================================================== #
        #                           SAVE SOME INFO
        # ======================================================================== #
        file_name_metadata = "../Data/info_wannier_rytova_keldysh_WSe2_many_sizes_eps_1_from_11_through_111"
        np.savez(file_name_metadata, L_values=L_values, n_points=n_points)

        # ======================================================================== #
        #              SAVE MATRICES WITH THE RESULTS
        # ======================================================================== #
        file_name = "../Data/data_wannier_rytova_keldysh_WSe2_many_sizes_eps_1_from_11_through_111"

        np.savez(file_name, eigvals_holder=eigvals_holder, eigvecs_holder=eigvecs_holder)
        print("Done!")

    if preview:
        print("Non-extrapolated binding-energies:")
        print("\t[#]\t|\tValue [meV]")
        for val_index in range(number_of_recorded_states):
            print("\t%i\t|\t%.2f" % (val_index, values[val_index]))


if __name__ == '__main__':
    main()
