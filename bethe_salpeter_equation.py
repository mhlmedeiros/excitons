#!/usr/bin/env python3

import sys
import time
import numpy as np
import scipy.linalg as LA
# import sympy
import matplotlib.pyplot as plt
from matplotlib import cm
import itertools as it
from numba import njit

import wannier_coulomb_numba as wannier

EPSILON_0 = 55.26349406             # e^2 GeV^{-1}fm^{-1} == e^2 (1e9 eV 1e-15 m)^{-1}
HBAR = 1.23984193/(2*np.pi)         # eV 1e-6 m/c
M_0  = 0.51099895000                # MeV/c^2
hbar2_over2m = HBAR**2/(2*M_0)*1e3  # meV nm^2

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

def hamiltonian(kx, ky, E_gap=0.5, Gamma=1, Alpha_c=1, Alpha_v=-1):
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

def values_and_vectors(hamiltonian, kx_matrix, ky_matrix, **kwargs):
    """
    This function calculates all the eigenvalues-eingenvectors pairs and return them
    into two multidimensional arrays named here as W and V respectively.

    The dimensions of such arrays depend on the number of sampled points of the
    reciprocal space and on the dimensions of our model Hamiltonian.

    W.shape = (# kx-points, # ky-points, # rows of "H")
    V.shape = (# kx-points, # ky-points, # rows of "H", # columns of "H")

    For "W" the order is straightforward:
    W[i,j,0]  = "the smallest eigenvalue for kx[i] and ky[j]"
    W[i,j,-1] = "the biggest eigenvalue for kx[i] and ky[j]"

    For "V" we have:
    V[i,j,:,0] = "first eigenvector which one corresponds to the smallest eigenvalue"
    V[i,j,:,-1] = "last eigenvector which one corresponds to the biggest eigenvalue"

    """
    n, m = kx_matrix.shape
    l, _ = hamiltonian(0,0,**kwargs).shape
    W = np.zeros((n,m,l))
    V = np.zeros((n,m,l,l),dtype=complex)
    for i in range(n):
        for j in range(m):
            W[i,j,:], V[i,j,:,:]  = LA.eigh(hamiltonian(kx_matrix[i,j], ky_matrix[i,j], **kwargs))
    return W,V

def split_values(values_array):
    # This function splits the eigenvalues into "condution" and "valence" sets

    # The "conduction_values" has stricly positive values
    conduction_values = [value for value in values_array if value > 0]

    # The negative or null values are labeled as "valence_values"
    valence_values = [value for value in values_array if value <= 0]
    return conduction_values, valence_values

def diagonal_elements(Values, dk2, N_submesh, epsilon, r_0):

    # Lets start with the potential around |k-k'| = 0
    k_vec_diff = np.array([0,0])
    V_0 = rytova_keldysh_average(k_vec_diff, dk2, N_submesh, epsilon, r_0)

    # First we get some information about the shape of "Values"
    kx_len, ky_len, num_vals = Values.shape
    cond_v, valen_v = split_values(Values[0,0,:]) # Just to set the size of the holder matrix

    # Then we initiate the Matrix that will contain the values (Ec - Ev) in its diagonal
    Wdim = kx_len * ky_len * len(cond_v) * len(valen_v)
    # print("W_diagonal_len : ", Wdim)
    W_diagonal = np.zeros(Wdim, dtype=float)
    n = 0 # initiate the counter to popupate W_diagonal

    # Now we run over all (kx, ky) pairs and see and split the values into "conduction" and "valence"
    for i, j in it.product(range(kx_len), range(ky_len)):
        conduction_values, valence_values = split_values(Values[i,j,:])
        # For each pair conduction valence we calculate (Ec - Ev)
        for p in it.product(conduction_values, valence_values[::-1]):
            diag_val = p[0]-p[1]
            W_diagonal[n] = diag_val + V_0 # save in right place
            n += 1 # update the counter

    # Now we put this array into the main diagonal of a matrix
    W_diag_matrix = np.diagflat(W_diagonal)
    return W_diag_matrix

def calculate_distance_k_pontual(k1_vec, k2_vec):
    """
    Like the version used in the Wannier this function calculates the "distance" between two points
    in the reciprocal space. The difference is that here it just returns one value instead of
    the whole matrix with all possible pairs' distances.
    """
    k_rel_vec = k1_vec-k2_vec
    dist = np.sqrt(k_rel_vec @ k_rel_vec)
    return dist


def delta_k1k2_cv(k1_ind, k2_ind, c1_ind, c2_ind, v1_ind, v2_ind, Vectors_flattened):
    """
    This function perfoms the inner product of the eigenstates of the model
    Hamiltonian in order to build the mixing terms for BSE matrix.

    k1_ind := 'target-k' index
    c1_ind := 'target-c' index
    v1_ind := 'target-v' index
    (line_ind := c1_ind + v1_ind*(#_conduction) + k1_ind * (#_conduction * #_valence))

    k2_ind := column index component
    c2_ind := column index component
    v2_ind := column index component
    (column_ind := c2_ind + v2_ind*(#_conduction) + k2_ind * (#_conduction * #_valence))
    """
    beta_c1_k1 = Vectors_flattened[k1_ind,:,c1_ind] # c-state
    beta_c2_k2 = Vectors_flattened[k2_ind,:,c2_ind] # c'-state
    beta_v1_k1 = Vectors_flattened[k1_ind,:,v1_ind] # v-state
    beta_v2_k2 = Vectors_flattened[k2_ind,:,v2_ind] # v'-state
    return (beta_c1_k1.conj() @ beta_c2_k2) * (beta_v2_k2.conj() @ beta_v1_k1)

def delta_k1k2(k1_ind, k2_ind, Vectors, Values):
    """
    This function generates the matrix for "Delta_k1_k2".

    As inputs we have to pass
        * the indices for the values of k1 and k2;
        * the 4-dimension array holding the Vectors
        * the 3-dimension array holding the Values

    considering that the 2D reciprocal space-section is taken to be 'flat'.

    """

    # To know how many conduction and valence bands we have
    # I SUPPOSE IT WILL BE FIXED OVER THE WHOLE RECIPROCAL SPACE
    cond_vs, vale_vs = split_values(Values[0,0,:])
    cond_n = len(cond_vs)
    vale_n = len(vale_vs)
    Delta_k1_k2 = np.zeros((cond_n*vale_n)**2, dtype=complex)

    n, m, s = Values.shape # n -> kx-points; m -> ky -> points; s -> # of eigenvalues
    V_flat = Vectors.reshape(n*m, s, s)

    # The order of the values is always from the smallest to the largest
    # Remember that the smallest value for conduction bands is also the closest
    # one of the gap among all the conduction states
    # On the other hand, the largest valence band is the closest of the gap
    # among valence bands.

    cond_inds = list(range(cond_n))
    vale_inds = list(range(-1,-1*(vale_n+1),-1))

    #==============#
    #  main loop:  #
    #==============#
    n = 0
    for (v,c,vl,cl) in it.product(vale_inds, cond_inds, vale_inds, cond_inds):
        Delta_k1_k2[n] = delta_k1k2_cv(k1_ind, k2_ind, c, cl, v, vl, V_flat)
        n += 1

    return Delta_k1_k2.reshape(cond_n*vale_n, cond_n*vale_n)


# ============================================================================= #
##                              Rytova-Keldysh:
# ============================================================================= #
@njit
def rytova_keldysh_pontual(q, dk2, epsilon, r_0):
    """
    The "pontual" version of the function in Wannier script.
    Instead of return the whole matrix this function returns
    only the value asked.
    """
    Vkk_const = 1e6/(2*EPSILON_0)
    V =  1/(epsilon*q + r_0*q**2)
    return - Vkk_const * dk2/(2*np.pi)**2 * V

@njit
def rytova_keldysh_average(k_vec_diff, dk2, N_submesh, epsilon, r_0):
    """
    As we've been using a square lattice, we can use
    * w_x_array == w_y_array -> w_array
    * with limits:  -dw/2, +dw/2
    * where: dw = sqrt(dk2)
    """
    if N_submesh==None:
        q = np.sqrt(k_vec_diff[0]**2 + k_vec_diff[1]**2)
        Potential_value = rytova_keldysh_pontual(q, dk2, epsilon, r_0)
    else:
        dk = np.sqrt(dk2)
        w_array = np.linspace(-dk/2, dk/2, N_submesh)
        Potential_value = 0
        N_sing = 0
        for wx in w_array:
            for wy in w_array:
                w_vec = np.array([wx, wy])
                q_vec = k_vec_diff + w_vec
                # q = np.sqrt(q_vec[0]**2 + q_vec[1]**2)
                q = np.linalg.norm(q_vec)
                if q == 0:
                    N_sing += 1
                    continue; # skip singularities
                Potential_value += rytova_keldysh_pontual(q, dk2, epsilon, r_0)
        if N_sing != 0 :
            print("\t\t\tFor k-k' = ", k_vec_diff ," the number of singular points was ", N_sing)
        Potential_value = Potential_value/(N_submesh**2 - N_sing)
    return Potential_value

@njit
def smart_rytova_keldysh_matrix(kx_flat, ky_flat, dk2, N_submesh, epsilon, r_0):
    """
    CONSIDERING A SQUARE K-SPACE GRID
    """
    n_all_k_space = len(kx_flat)
    n_first_row_k = int(np.sqrt(n_all_k_space)) # number of points in the first row of the grid
    M_first_rows = np.zeros((n_first_row_k, n_all_k_space))
    M_complete = np.zeros((n_all_k_space, n_all_k_space))
    print("\t\tCalculating the first rows (it may take a while)...")
    for k1_ind in range(n_first_row_k):
        for k2_ind in range(k1_ind+1, n_all_k_space):
            k1_vec = np.array((kx_flat[k1_ind], ky_flat[k1_ind]))
            k2_vec = np.array((kx_flat[k2_ind], ky_flat[k2_ind]))
            k_diff = k1_vec - k2_vec
            M_first_rows[k1_ind, k2_ind] = rytova_keldysh_average(k_diff, dk2, N_submesh, epsilon, r_0)

    print("\t\tOrganizing the the calculated values...")
    M_complete[:n_first_row_k,:] = M_first_rows
    for row in range(1, n_first_row_k):
        ni, nf = row * n_first_row_k, (row+1) * n_first_row_k
        mi, mf = ni, -ni
        M_complete[ni:nf, mi:] = M_first_rows[:, :mf]

    M_complete += M_complete.T
    # plt.imshow(M_complete)
    return M_complete


def out_of_diagonal(Vectors, Values, kx_matrix, ky_matrix, dk2, N_submesh, epsilon, r_0):
    # First we need some information about the shape of "Values"
    # kx_len, ky_len, num_states = Values.shape
    cond_v, vale_v = split_values(Values[0,0,:]) # Just to set the size of the holder matrix

    # To calculate the Rytova-Keldysh potential using just one index to identify a k-point:
    Kx_flat = kx_matrix.flatten()
    Ky_flat = ky_matrix.flatten()

    # It is crucial to know how many conduction and valence bands we have:
    S = len(cond_v) * len(vale_v) # This is the amount of combinations of conduction and valence bands
    Z = len(Kx_flat) # number of points in reciprocal space
    W_ND = np.zeros((S*Z,S*Z), dtype=complex)
    # Just for test purpose:
    # indice_k2_test_1, indice_k2_test_2 = np.random.randint(1,Z,2)

    #==================#
    #  Rytova-Keldysh  #
    #==================#
    print("\tCalculating the Rytova-Keldysh potential...")
    V_RK = smart_rytova_keldysh_matrix(Kx_flat, Ky_flat, dk2, N_submesh, epsilon, r_0)

    #==============#
    #  main loop:  #
    #==============#
    print("\tInserting the mixing terms Delta(kcv,k'c'v')...")
    for k1 in range(Z-1):
        for k2 in range(k1+1, Z):
            # Signal included in "rytova_keldysh_pontual":
            delta = delta_k1k2(k1, k2, Vectors, Values)
            # k1_vec = np.array([Kx_flat[k1], Ky_flat[k1]])
            # k2_vec = np.array([Kx_flat[k2], Ky_flat[k2]])
            # k_diff = k1_vec - k2_vec
            Dk1k2 = delta * V_RK[k1, k2] # USE THIS ONE WITH "smart_rytova_keldysh_matrix"
            # Dk1k2 = delta * rytova_keldysh_average(k_diff, dk2, N_submesh, epsilon, r_0)
            # if k1 == 0 and k2 == indice_k2_test_1 : print("Delta_k1_k2: ", delta)
            # elif k1 == 0 and k2 == indice_k2_test_2 : print("Delta_k1_k2: ",delta)
            W_ND[k1*S:(k1+1)*S, k2*S:(k2+1)*S] = Dk1k2
            W_ND[k2*S:(k2+1)*S, k1*S:(k1+1)*S] = Dk1k2.T.conj()
    return W_ND


# ============================================================================= #
##                     Rytova-Keldysh Average around zero:
# ============================================================================= #

def potential_matrix(kx_flat, ky_flat, dk2, epsilon, r_0, N_submesh=None):
    """
    This function generates a square matrix that contains the values of
    the potential for each pair of vectors k & k'.

    Dimensions = Nk x Nk
    where Nk = (Nk_x * Nk_y)
    """

    # OUT OF DIAGONAL: SMART SCHEME
    V_main = smart_rytova_keldysh_matrix(kx_flat, ky_flat, dk2, N_submesh, epsilon, r_0)

    # DIAGONAL VALUE: EQUAL FOR EVERY POINT (WHEN USING SUBMESH)
    if N_submesh != None:
        print("\t\tCalculating the potential around zero...")
        k_0 = np.array([0,0])
        V_0 = rytova_keldysh_average(k_0, dk2, N_submesh, epsilon, r_0)
        # PUT ALL TOGETHER
        V_main = np.fill_diagonal(V_main, V_0)

    return V_main


def include_deltas(V_RK, Values, Vectors, N_submesh):
    """
    Once potential matrix [V(k-k')] is available one can add the 'mixing'
    term. This mixing term is defined for every c-c' and v-v' pairs
    and also for every k-k' combination:

    Delta([c,v,k],[c',v',k']) = < c,k |c',k'> * <v',k'|v,k >

    """
    # It is crucial to know how many conduction and valence bands we have:
    cond_v, vale_v = split_values(Values[0,0,:]) # Just to set the size of the holder matrix

    S = len(cond_v) * len(vale_v) # This is the amount of combinations of conduction and valence bands
    Z,_ = V_RK.shape # number of points in reciprocal space
    W_ND = np.zeros((S*Z,S*Z), dtype=complex) # initiate an empty matrix

    correction = (N_submesh == None)

    ## NON-DIAGONAL && DIAGONAL BLOCKS (if correction == False):
    for k1 in range(Z - correction):
        for k2 in range(k1 + correction, Z):
            # THE DELTAS CAN BE MATRICES, IT DEPENDS ON THE HAMILTONIAN MODEL:
            delta = delta_k1k2(k1, k2, Vectors, Values)
            Dk1k2 = delta * V_RK[k1, k2] # USE THIS ONE WITH
            W_ND[k1*S:(k1+1)*S, k2*S:(k2+1)*S] = Dk1k2
            if k1 != k2:
                # FOR NON-DIAGONAL BLOCKS:
                W_ND[k2*S:(k2+1)*S, k1*S:(k1+1)*S] = Dk1k2.T.conj()

    return W_ND


# ============================================================================= #
##                              Visualization:
# ============================================================================= #
def plot_wave_function(eigvecs_holder, state_preview):
    N = int(np.sqrt(eigvecs_holder.shape[0]))
    wave_funct = np.reshape(eigvecs_holder[:, state_preview, 0],(N,N))
    fig, ax = plt.subplots(figsize=(10,12))
    mapping = ax.pcolormesh(abs(wave_funct)**2 , cmap=cm.inferno)
    fig.colorbar(mapping, ax=ax)
    plt.show()


@st_time
def main():
    # ============================================================================= #
    ##                              Outuput options:
    # ============================================================================= #
    save = True
    preview = True

    # ============================================================================= #
    ##                      Hamiltonian and Potential parameters:
    # ============================================================================= #
    # mc = 0.2834  # M_0
    # mv = -0.3636 # M_0
    # gamma = 2.6e2 # meV*nm ~ 2.6 eV*AA
    # Egap = 2.4e3 # meV ~ 2.4 eV
    r0_chosen = 4.51 # nm (WSe2)

    ## PAULO'S TEST:
    gamma =  3.91504469e2# meV*nm ~ 2.6 eV*AA
    Egap = 1311.79 # meV ~ 2.4 eV


    epsilon_eff = 1
    alpha_choice = 0

    alpha_options = ['zero', 'masses', 'corrected']
    # alpha_choice = int((input('''Enter the 'alphas-choice'(0/1/2):
    #         option (0) : alphas == 0 (default)
    #         option (1) : alphas == 1/m_j (WSe2 masses)
    #         option (2) : alphas == 'corrected'
    #         your option = ''')) or "0")

    if alpha_choice in (0,1,2):
        alpha_option = alpha_options[alpha_choice]
    else:
        alpha_option = alpha_options[0]

    print("Option adopted: %s" % alpha_option)

    if alpha_choice == 1:
        alphac, alphav = 1/mc, 1/mv
    elif alpha_choice == 2:
        alphac = 1/mc + 1/hbar2_over2m * (gamma**2/Egap)
        alphav = 1/mv - 1/hbar2_over2m * (gamma**2/Egap)
    else:
        alphac, alphav = 0, 0

    ## TERMINAL OPTIONS:
    while len(sys.argv) > 1:
        option = sys.argv[1];               del sys.argv[1]
        if option == '-Egap':
            Egap = float(sys.argv[1]);     del sys.argv[1]
        # elif option == '-alpha_op':
        #     alpha_option = sys.argv[1];     del sys.argv[1]
        # elif option == '-alphac':
        #     alphac = float(sys.argv[1]);    del sys.argv[1]
        # elif option == '-alphav':
        #     alphav = float(sys.argv[1]);    del sys.argv[1]
        elif option == '-gamma':
            gamma = float(sys.argv[1]);     del sys.argv[1]
        elif option == '-ns':
            save = False
        elif option == '-np':
            preview = False
        else:
            print(sys.argv[0],': invalid option', option)
            sys.exit(1)

    if Egap == 0:
        # When we don't have a gap it possible to have an error
        # due to the current strategy using "split_values" function,
        # this artificial gap prevent this problem.
        Egap = 1e-5

    # HAMILTONIAN PARAMS
    hamiltonian_params = dict(E_gap=Egap, Alpha_c=alphac,
                             Alpha_v=alphav, Gamma=gamma)

    # POTENTIAL PARAMS
    epsilon = epsilon_eff
    r_0 = r0_chosen


    # ============================================================================ #
    ## Define the sizes of the region in k-space to be investigated:
    # ============================================================================ #
    min_size = 5 # nm^-1
    max_size = 5 # nm^-1
    L_values = range(min_size, max_size + 1) # [min (min+1) ... max] # nm^-1
    list(L_values)


    # ============================================================================ #
    ## Choose the number of discrete points to investigate the convergence:
    # ============================================================================ #
    min_points = 101
    max_points = 101
    N_submesh = None
    n_points = list(range(min_points, max_points+1, 2)) # [107 109 111]


    # ============================================================================ #
    ##              Matrices to hold the eigenvalues and the eigenvectors:
    # ============================================================================ #
    number_of_recorded_states = 100
    eigvals_holder = np.zeros((number_of_recorded_states, len(n_points), len(L_values)))
    eigvecs_holder = np.zeros((max_points**2, number_of_recorded_states, len(L_values)),dtype=complex)


    # ============================================================================ #
    ##                                  main_loop
    # ============================================================================ #
    for ind_L in range(len(L_values)):
        print("\nCalculating the system with {} nm^(-1).".format(L_values[ind_L]))
        for ind_Nk in range(len(n_points)):
            print("Discretization: {} x {} ".format(n_points[ind_Nk], n_points[ind_Nk]))
            # First we have to define the grid:
            Kx, Ky, dk2 = wannier.define_grid_k(L_values[ind_L], n_points[ind_Nk])
            # Then, we need the eigenvalues and eigenvectors of our model for eack k-point
            Values3D, Vectors4D = values_and_vectors(hamiltonian, Kx, Ky, **hamiltonian_params)

            # The Bethe-Salpeter Equation:
            print("\tBuilding Potential matrix (Nk x Nk)... ")
            V_kk = potential_matrix(kx_flat, ky_flat, dk2, epsilon, r_0, N_submesh=None)

            print("\tIncluding 'mixing' terms (Deltas)... ")
            W_non_diag = include_deltas(V_RK, Values, Vectors, N_submesh)

            print("\tIncluding 'pure' diagonal elements..")
            W_diag = diagonal_elements(Values3D)
            W_total = W_diag + W_non_diag

            # print("Building the BSE matrix...")
            # W_diag = diagonal_elements(Values3D)
            # W_non_diag = out_of_diagonal(Vectors4D, Values3D, Kx, Ky,
            #                             dk2, N_submesh, epsilon, r_0)
            # W_total = W_diag + W_non_diag

            # Solutions of the BSE:
            print("\tDiagonalizing the BSE matrix...")
            values, vectors = LA.eigh(W_total)

            # SAVE THE FIRST STATES ("number_of_recorded_states"):
            # Note, as we want the binding energies, we have to discount the gap
            eigvals_holder[:, ind_Nk, ind_L] = values[:number_of_recorded_states] - Egap



        # SAVE THE VECTORS WITH THE FINEST DISCRETIZATION:
        eigvecs_holder[:, :, ind_L] = vectors[:,:number_of_recorded_states]

    if save:
        common_path = "../Data/BSE_results/"
        common_name = (
                        "alphas_" + alpha_option +
                        "_gamma_" + str(gamma*1e-2)  +
                        "_eV_AA_Eg_" + str(Egap*1e-3) +
                        "_eV_size_" + str(max_size) +
                        "_eps_" + str(epsilon_eff) +
                        "_discrete_" + str(max_points)+
                        "_sub_mesh_" + str(N_submesh) +
                        "_with_smart_rytova_keldysh"+
                        "_with_potential_average_around_zero"
                        )
        info_file_path_and_name = common_path + "info_BSE_" + common_name
        data_file_path_and_name = common_path + "data_BSE_" + common_name


        print("\n\nSaving...")
        # ======================================================================== #
        #                           SAVE SOME INFO
        # ======================================================================== #
        np.savez(info_file_path_and_name, L_values=L_values, n_points=n_points)

        # ======================================================================== #
        #              SAVE MATRICES WITH THE RESULTS
        # ======================================================================== #
        np.savez(data_file_path_and_name, eigvals_holder=eigvals_holder, eigvecs_holder=eigvecs_holder)
        print("Done!")

    if preview:
        print("Non-extrapolated binding-energies:")
        print("\t[#]\t|\tValue [meV]")
        for val_index in range(number_of_recorded_states):
            print("\t%i\t|\t%.2f" % (val_index, values[val_index]-Egap))
        # fig, ax = plt.subplots(figsize=(10,10))
        # ax.imshow(np.(W_non_diag))
        # plt.show()
        # state_preview = 0
        # plot_wave_function(eigvecs_holder, state_preview)


if __name__ == '__main__':
    # import timeit
    # setup = "from __main__ import main"
    # Ntimes = 1
    # print(timeit.timeit("main()", setup=setup, number=Ntimes))
    main()
