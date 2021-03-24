#!/usr/bin/env python3

import sys
import time
import numpy as np
import scipy.linalg as LA
import matplotlib.pyplot as plt
from matplotlib import cm
import itertools as it
from numba import njit
import hamiltonians as ham
import physical_constants as const

# ============================================================================= #
##                      FUNCTION TO CREATE A K-SPACE GRID
# ============================================================================= #
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

# ============================================================================= #
##              BSE-Matrix Diagonal elements (no potential nor exchange):
# ============================================================================= #

def diagonal_elements(Values, Ham):
    """
    Function tha returns a diagonal matrix which has the elements defined as

        W_diag_matrix[n,n] = E_c(k) - E_v(k)

    Note that a diagonal element in the BSE-matrix is defined by
        * c = c'
        * v = v'
        * k = k'

    Of course the value given above isn't the only contribution to the diagonal
    elements of BSE-matrix. The following contribution also has diagonal contribution:

        * Exchange term: X (NOT IMPLEMENTED YET)
        * Potential term: when average around the |k-k'|=0 is considered.
    """
    # First we get some information about the shape of "Values"
    kx_len, ky_len, num_vals = Values.shape
    cond_n, vale_n = Ham.condBands, Ham.valeBands # Just to set the size of the holder matrix
    cond_inds = list(range(-1,-1*(cond_n+1),-1))    # NEW VERSION
    vale_inds = list(range(vale_n))                 # NEW VERSION

    # Then we initiate the Matrix that will contain the values (Ec - Ev) in its diagonal
    Wdim = kx_len * ky_len * cond_n * vale_n
    # print("W_diagonal_len : ", Wdim)
    W_diagonal = np.zeros(Wdim, dtype=float)

    n = 0 # initiate the counter to popupate W_diagonal
    # Now we run over all (kx, ky) pairs and split the values into "conduction" and "valence"
    for i, j in it.product(range(ky_len), range(kx_len)):
        # REMEMBER THAT THE SECOND INDEX IS THE FASTEST
        # For each pair conduction valence we calculate (Ec - Ev)
        for v_ind, c_ind in it.product(vale_inds, cond_inds):
            # REMEMBER THAT THE SECOND INDEX IS THE FASTEST
            diag_val = Values[i,j,c_ind] - Values[i,j,v_ind]
            W_diagonal[n] = diag_val # save in right place
            n += 1 # update the counter

    # Now we put this array into the main diagonal of a matrix
    W_diag_matrix = np.diagflat(W_diagonal)
    return W_diag_matrix

# ============================================================================= #
##              Deltas (Mixing):
# ============================================================================= #
@njit
def include_deltas(V_RK, Values, Vectors, N_submesh, Ham, scheme='H'):
    """
    Once potential matrix [V(k-k')] is available one can add the 'mixing'
    term. This mixing term is defined for every c-c' and v-v' pairs
    and also for every k-k' combination:

    Delta([c,v,k],[c',v',k']) = < c,k |c',k'> * <v',k'|v,k >

    """
    # It is crucial to know how many conduction and valence bands we have:
    cond_n, vale_n = Ham.condBands, Ham.valeBands

    S = cond_n * vale_n # This is the amount of combinations of conduction and valence bands
    Z,_ = V_RK.shape # number of points in reciprocal space
    W_ND = 1j * np.zeros((S*Z,S*Z)) # initiate an empty matrix

    ## If the 'submesh-strategy' is not being used  we need to avoid
    ## the diagonal elements. When the 'correction' is True (or 1), this
    ## avoidance is implemented in the nested for-loop below.
    correction = (N_submesh == None)


    ## NON-DIAGONAL && DIAGONAL BLOCKS (if correction == False):
    if scheme == 'H':
        "HALF-FILLING"
        for k1 in range(Z - correction):
            for k2 in range(k1 + correction, Z):
                # THE DELTAS CAN BE MATRICES, IT DEPENDS ON THE HAMILTONIAN MODEL:
                delta = delta_k1k2(k1, k2, Vectors, Values, Ham)
                Dk1k2 = delta * V_RK[k1, k2] # USE THIS ONE WITH
                W_ND[k1*S:(k1+1)*S, k2*S:(k2+1)*S] = Dk1k2
                if k1 != k2:
                    # FOR NON-DIAGONAL BLOCKS:
                    W_ND[k2*S:(k2+1)*S, k1*S:(k1+1)*S] = Dk1k2.T.conj()
    else:
        "FULL-FILLING"
        for k1 in range(Z):
            for k2 in range(Z):
                # THE DELTAS CAN BE MATRICES, IT DEPENDS ON THE HAMILTONIAN MODEL:
                delta = delta_k1k2(k1, k2, Vectors, Values, Ham)
                Dk1k2 = delta * V_RK[k1, k2] # USE THIS ONE WITH
                W_ND[k1*S:(k1+1)*S, k2*S:(k2+1)*S] = Dk1k2
    return W_ND

@njit
def delta_k1k2(k1_ind, k2_ind, Vectors, Values, Ham):
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
    cond_n, vale_n = Ham.condBands, Ham.valeBands
    Delta_k1_k2 = 1j * np.zeros((cond_n*vale_n)**2)

    nkx, nky, nstates = Values.shape # nkx -> kx-points; nky -> ky -> points; nstates -> # of eigenvalues
    V_flat = Vectors.reshape(nkx*nky, nstates, nstates)

    # The order of the values is always from the smallest to the largest
    # Remember that the smallest value for conduction bands is also the closest
    # one of the gap among all the conduction states
    # On the other hand, the largest valence band is the closest of the gap
    # among valence bands.

    # cond_inds = list(range(cond_n))                 # OLD VERSION
    # vale_inds = list(range(-1,-1*(vale_n+1),-1))    # OLD VERSION
    cond_inds = list(range(-1,-1*(cond_n+1),-1))    # NEW VERSION
    vale_inds = list(range(vale_n))                 # NEW VERSION

    #===================================================================#
    #                       main loop of this function:                 #
    #===================================================================#
    ## TO USE 'NUMBA COMPILATION IN THIS FUNCTION WE CAN'T USE 'itertools'
    n = 0
    for v in vale_inds:
        for c in cond_inds:
            for vl in vale_inds:
                for cl in cond_inds:
                    Delta_k1_k2[n] = delta_k1k2_cv(k1_ind, k2_ind, c, cl, v, vl, V_flat)
                    n += 1

    return Delta_k1_k2.reshape(cond_n*vale_n, cond_n*vale_n)

@njit
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
    return np.sum(beta_c1_k1.conj() * beta_c2_k2) * np.sum(beta_v2_k2.conj() * beta_v1_k1)


# ============================================================================= #
##              Rytova-Keldysh:
# ============================================================================= #
def potential_matrix(V, kx_matrix, ky_matrix, N_submesh, submesh_radius=0):
    """
    This function generates a square matrix that contains the values of
    the potential for each pair of vectors k & k'.

    Dimensions = Nk x Nk
    where Nk = (Nk_x * Nk_y)
    """
    kx_flat = kx_matrix.flatten()
    ky_flat = ky_matrix.flatten()

    # OUT OF DIAGONAL: SMART SCHEME
    # N_submesh_off = N_submesh if submesh_off_diag == True else None
    V_main = smart_potential_matrix(V, kx_flat, ky_flat, N_submesh, submesh_radius)

    # DIAGONAL VALUE: EQUAL FOR EVERY POINT (WHEN USING SUBMESH)
    if N_submesh != None:
        # print("\t\tCalculating the potential around zero...")
        k_0 = np.array([0,0])
        V_0 = potential_average(V, k_0, N_submesh, submesh_radius)
        np.fill_diagonal(V_main, V_0) # PUT ALL TOGETHER

    return V_main

@njit
def smart_potential_matrix(V, kx_flat, ky_flat, N_submesh, submesh_radius):
    """
    CONSIDERING A SQUARE K-SPACE GRID:

    This function explore the regularity in the meshgrid that defines the k-space
    to build the potential-matrix [V(k-k')].

    Note that it is exclusive for the Rytova-Keldysh potential.

    # TODO: to make this function more general in the sense that any other potential
    function could be adopted: Define a Potential-class and  pass instances of such
    class instead of pass attributes of Rytova-Keldysh potential.

    """
    n_all_k_space = len(kx_flat)
    n_first_row_k = int(np.sqrt(n_all_k_space)) # number of points in the first row of the grid
    M_first_rows = np.zeros((n_first_row_k, n_all_k_space))
    M_complete = np.zeros((n_all_k_space, n_all_k_space))
    # print("\t\tCalculating the first rows (it may take a while)...")
    for k1_ind in range(n_first_row_k):
        for k2_ind in range(k1_ind+1, n_all_k_space):
            k1_vec = np.array([kx_flat[k1_ind], ky_flat[k1_ind]])
            k2_vec = np.array([kx_flat[k2_ind], ky_flat[k2_ind]])
            k_diff = k1_vec - k2_vec
            M_first_rows[k1_ind, k2_ind] = potential_average(V, k_diff, N_submesh, submesh_radius)

    # print("\t\tOrganizing the the calculated values...")
    M_complete[:n_first_row_k,:] = M_first_rows
    for row in range(1, n_first_row_k):
        ni, nf = row * n_first_row_k, (row+1) * n_first_row_k
        mi, mf = ni, -ni
        M_complete[ni:nf, mi:] = M_first_rows[:, :mf]

    M_complete += M_complete.T
    # plt.imshow(M_complete)
    return M_complete

@njit
def potential_average(V, k_vec_diff, N_submesh, submesh_radius):
    """
    As we've been using a square lattice, we can use
    * w_x_array == w_y_array -> w_array
    * with limits:  -dw/2, +dw/2
    * where: dw = sqrt(dk2)
    """
    k_diff_norm = np.sqrt(k_vec_diff[0]**2 + k_vec_diff[1]**2)
    dk = np.sqrt(V.dk2)
    threshold = submesh_radius * dk

    # print('threshold: ', threshold)
    # print('k_diff: ', k_diff_norm)

    if N_submesh==None or k_diff_norm > threshold:
        Potential_value = V.call(k_diff_norm)
    else:
        # THIS BLOCK WILL RUN ONLY IF "k_diff_norm" IS EQUAL OR SMALLER
        # THAN A LIMIT, DENOTED HERE BY "threshold":
        w_array = np.linspace(-dk/2, dk/2, N_submesh)
        Potential_value = 0
        number_of_sing_points = 0
        for wx in w_array:
            for wy in w_array:
                w_vec = np.array([wx, wy])
                q_vec = k_vec_diff + w_vec
                q = np.linalg.norm(q_vec)
                if q == 0: number_of_sing_points += 1; continue; # skip singularities
                Potential_value += V.call(q)
        # if number_of_sing_points != 0 :
            # print("\t\t\tFor k-k' = ", k_vec_diff ," the number of singular points was ", number_of_sing_points)
        Potential_value = Potential_value/(N_submesh**2 - number_of_sing_points)
    return Potential_value

# ============================================================================= #
##                              BSE - Exchange term:
# ============================================================================= #
@njit
def include_X(Values, Vectors, r_0, d, Ham, scheme='H'):
    """
    # TODO: Write a doc string
    """
    # It is crucial to know how many conduction and valence bands we have:
    cond_n, vale_n = Ham.condBands, Ham.valeBands # Just to set the size of the holder matrix

    S = cond_n * vale_n # This is the amount of combinations of conduction and valence bands
    grid_nrows, grid_ncolumns, _ = Values.shape # number of points in reciprocal space
    Z = grid_nrows * grid_ncolumns
    # print(Z)
    # TODO: CONFIRM IF THE ENTRIES OF THE MATRIX ARE COMPLEX
    X = 1j * np.zeros((S*Z,S*Z)) # initiate an empty matrix

    if scheme == 'H':
        "HALF-FILLING"
        for k1 in range(Z):
            for k2 in range(k1, Z):
                # THE DIMENSIONS OF THE X_k1k2 DEPENDS ON THE HAMILTONIAN MODEL:
                X_k1k2_block = X_k1k2(k1, k2, Vectors, Values, r_0, d, Ham)
                X[k1*S:(k1+1)*S, k2*S:(k2+1)*S] = X_k1k2_block
                if k1 != k2:
                    # FOR NON-DIAGONAL BLOCKS:
                    X[k2*S:(k2+1)*S, k1*S:(k1+1)*S] = X_k1k2_block.T.conj()

    else:
        "FULL-FILLING"
        for k1 in range(Z):
            for k2 in range(Z):
                # THE DIMENSIONS OF THE X_k1k2 DEPENDS ON THE HAMILTONIAN MODEL:
                X_k1k2_block = X_k1k2(k1, k2, Vectors, Values, r_0, d, Ham)
                # print("Block = \n \t", X_k1k2_block)
                X[k1*S:(k1+1)*S, k2*S:(k2+1)*S] = X_k1k2_block
    return X

@njit
def X_k1k2(k1_ind, k2_ind, Vectors, Values, r_0, d, Ham):
    """
    # TODO: DOC STRING
    """
    cond_n, vale_n = Ham.condBands, Ham.valeBands
    X_k1_k2 = 1j * np.zeros((cond_n*vale_n)**2)
    nkx, nky, nstates = Values.shape # nkx -> kx-points; nky -> ky -> points; nstates -> # of eigenvalues
    V_flat = Vectors.reshape(nkx*nky, nstates, nstates)
    W_flat = Values.reshape(nkx*nky, nstates)

    # The order of the values is always from the smallest to the largest
    # Remember that the smallest value for conduction bands is also the closest
    # one of the gap among all the conduction states
    # On the other hand, the largest valence band is the closest of the gap
    # among valence bands.

    cond_inds = list(range(-1,-1*(cond_n+1),-1))    # [ 0, 1, ... ,  cond_n-1] # NEW VERSION
    vale_inds = list(range(vale_n))                 # [-1,-2, ... , -vale_n]   # NEW VERSION

    #===================================================================#
    #                       main loop of this function:                 #
    #===================================================================#
    ## TO USE 'NUMBA COMPILATION IN THIS FUNCTION WE CAN'T USE 'itertools'
    n = 0
    for v in vale_inds:
        for c in cond_inds:
            for vl in vale_inds:
                for cl in cond_inds:
                    X_k1_k2[n] = X_k1k2_cv(k1_ind, k2_ind,
                                            c, cl, v, vl,
                                            Ham, V_flat, W_flat)
                    n += 1

    # ================================================================= #
    epsilon_m = 2 * r_0/d
    int_laplacian_V = - 1e6/(const.EPSILON_0 * epsilon_m)

    return int_laplacian_V * X_k1_k2.reshape(cond_n*vale_n, cond_n*vale_n)

@njit
def X_k1k2_cv(k1_ind, k2_ind, c1_ind, c2_ind, v1_ind, v2_ind, Ham, Vectors_flattened, Values_flattened):
    """
    # TODO: DOC STRING
    """
    Pix, Piy = Ham.Pi()
    # print(Pix)
    # #
    c1_k1 = np.copy(Vectors_flattened[k1_ind,:,c1_ind]) # c-state
    c2_k2 = np.copy(Vectors_flattened[k2_ind,:,c2_ind]) # c'-state
    v1_k1 = np.copy(Vectors_flattened[k1_ind,:,v1_ind]) # v-state
    v2_k2 = np.copy(Vectors_flattened[k2_ind,:,v2_ind]) # v'-state
    #
    E_cv   = Values_flattened[k1_ind,c1_ind] - Values_flattened[k1_ind,v1_ind]
    E_vc_p = Values_flattened[k2_ind,v2_ind] - Values_flattened[k2_ind,c2_ind]
    #
    Ax = c1_k1.conj().dot(Pix.dot(v1_k1))
    Ay = c1_k1.conj().dot(Piy.dot(v1_k1))
    Bx = v2_k2.conj().dot(Pix.dot(c2_k2))
    By = v2_k2.conj().dot(Piy.dot(c2_k2))
    #
    vA = 1/np.sqrt(2) * (1/E_cv) * np.array([Ax,Ay])
    vB = 1/np.sqrt(2) * (1/E_vc_p) * np.array([Bx,By])
    return vA.dot(vB)


# ============================================================================= #
##              Visualization:
# ============================================================================= #
def plot_wave_function(eigvecs_holder, state_preview):
    N = int(np.sqrt(eigvecs_holder.shape[0]))
    wave_funct = np.reshape(eigvecs_holder[:, state_preview, 0],(N,N))
    fig, ax = plt.subplots(figsize=(10,12))
    mapping = ax.pcolormesh(abs(wave_funct)**2 , cmap=cm.inferno)
    fig.colorbar(mapping, ax=ax)
    plt.show()


def main():
    # ========================================================================= #
    ##              Outuput options:
    # ========================================================================= #
    save = True
    preview = True

    # ========================================================================= #
    ##              Hamiltonian and Potential parameters:
    # ========================================================================= #
    # mc = 0.2834  # M_0
    # mv = -0.3636 # M_0
    # gamma = 2.6e2 # meV*nm ~ 2.6 eV*AA
    # Egap = 2.4e3 # meV ~ 2.4 eV


    ## PAULO'S TEST:
    # gamma =  3.91504469e2 # meV*nm ~ 2.6 eV*AA
    # Egap = 1311.79 # meV ~ 2.4 eV
    # r0_chosen = 4.51 # nm (WSe2)

    ## TEST OF ABSORPTION
    r0_chosen = 5.0 # nm
    Egap = 1e3 # meV
    gamma = 0
    mc = 0.2
    mv = -0.4

    epsilon_eff = 1
    alpha_choice = 1

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
        alphac = 1/mc + 1/const.hbar2_over2m * (gamma**2/Egap)
        alphav = 1/mv - 1/const.hbar2_over2m * (gamma**2/Egap)
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

    #==================#
    # HAMILTONIAN PARAMS
    #==================#
    hamiltonian_params = dict(E_gap=Egap, Alpha_c=alphac,
                             Alpha_v=alphav, Gamma=gamma)
    #=================#
    # POTENTIAL PARAMS
    #=================#
    epsilon = epsilon_eff
    r_0 = r0_chosen

    #================#
    # EXCHANGE PARAMS
    #================#
    d_chosen = 60       # nm
    p = 1/np.sqrt(2) * np.array([1, 1j])
    m = 1/np.sqrt(2) * np.array([1,-1j])
    O = np.array([0,0])

    Pi4x4 = gamma * np.array([
        [O,p,O,O],
        [m,O,O,O],
        [O,O,O,p],
        [O,O,m,O]
    ])

    Pi2x2 = gamma * np.array([
        [O,p],
        [m,O]
    ])


    # ========================================================================= #
    ## Define the Hamiltonian:
    # ========================================================================= #
    hamiltonian = ham.H4x4_equal(alphac, alphav, Egap, gamma)
    # hamiltonian = ham.H2x2(alphac, alphav, Egap, gamma)


    # ========================================================================= #
    ## Define the sizes of the region in k-space to be investigated:
    # ========================================================================= #
    min_size = 5 # nm^-1
    max_size = 5 # nm^-1
    L_values = range(min_size, max_size + 1) # [min (min+1) ... max] # nm^-1
    list(L_values)


    # ========================================================================= #
    ## Choose the number of discrete points to investigate the convergence:
    # ========================================================================= #
    min_points = 101
    max_points = 101
    N_submesh = 101
    submesh_limit = 1000 # units of Delta_k (square lattice)
    n_points = list(range(min_points, max_points+1, 2)) # [107 109 111]


    # ========================================================================= #
    ## Matrices to hold the eigenvalues and the eigenvectors:
    # ========================================================================= #
    # Number of valence-conduction bands combinations:
    n_val_cond_comb = hamiltonian.condBands * hamiltonian.valeBands
    number_of_recorded_states = 100
    eigvals_holder = np.zeros((number_of_recorded_states, len(n_points),
                            len(L_values)))
    eigvecs_holder = np.zeros((n_val_cond_comb * max_points**2, number_of_recorded_states,
                            len(L_values)), dtype=complex)


    # ========================================================================= #
    ##                                  main_loop
    # ========================================================================= #
    def build_bse_matrix(ind_L, ind_Nk):
        # First we have to define the grid:
        Kx, Ky, dk2 = define_grid_k(L_values[ind_L], n_points[ind_Nk])

        # Then, we need the eigenvalues and eigenvectors of our model for eack k-point
        # Values3D, Vectors4D = values_and_vectors(hamiltonian2x2, Kx, Ky, **hamiltonian_params)
        Values3D, Vectors4D = ham.values_and_vectors(hamiltonian, Kx, Ky)

        # Defining the potential
        V = ham.Rytova_Keldysh(dk2=dk2, r_0=r_0, epsilon=epsilon_eff)

        ### THE BETHE-SALPETER MATRIX CONSTRUCTION:
        print("\tBuilding Potential matrix (Nk x Nk)... ")
        V_kk = potential_matrix(V, Kx, Ky, N_submesh, submesh_radius=submesh_limit)

        print("\tIncluding 'mixing' terms (Deltas)... ")
        W_non_diag = include_deltas(V_kk, Values3D, Vectors4D, N_submesh)

        print("\tIncluding Exchange term...")
        # X_term = dk2/(2*np.pi)**2 * include_X(Values3D, Vectors4D, r_0, d_chosen, hamiltonian)
        X_term = 0

        print("\tIncluding 'pure' diagonal elements..")
        W_diag = diagonal_elements(Values3D)

        return W_diag + W_non_diag + X_term

    def diagonalize_bse(BSE_MATRIX):
        print("\tDiagonalizing the BSE matrix...")
        return LA.eigh(BSE_MATRIX)

    for ind_L in range(len(L_values)):
        print("\nCalculating the system with {} nm^(-1).".format(L_values[ind_L]))
        for ind_Nk in range(len(n_points)):
            print("Discretization: {} x {} ".format(n_points[ind_Nk], n_points[ind_Nk]))

            # BUILD THE MATRIX:
            W_total = build_bse_matrix(ind_L, ind_Nk)

            # DIAGONALIZATION:
            values, vectors = LA.eigh(W_total)

            # SAVE THE FIRST STATES ("number_of_recorded_states"):
            # Note, as we want the binding energies, we have to discount the gap
            eigvals_holder[:, ind_Nk, ind_L] = values[:number_of_recorded_states] - Egap


        # SAVE THE VECTORS WITH THE FINEST DISCRETIZATION:
        eigvecs_holder[:, :, ind_L] = vectors[:,:number_of_recorded_states]

    if save:
        common_path = "../Data/BSE_results/"
        common_name = ( "alphas_" + alpha_option +
                        "_gamma_" + str(gamma*1e-2)  +
                        "_eV_AA_Eg_" + str(Egap*1e-3) +
                        "_eV_size_" + str(max_size) +
                        "_eps_" + str(epsilon_eff) +
                        "_discrete_" + str(max_points)+
                        "_sub_mesh_" + str(N_submesh) +
                        "_submesh_radius_" + str(submesh_limit))
        info_file_path_and_name = common_path + "info_BSE_" + common_name
        data_file_path_and_name = common_path + "data_BSE_" + common_name


        print("\n\nSaving...")
        # ===================================================================== #
        #                           SAVE SOME INFO
        # ===================================================================== #
        np.savez(info_file_path_and_name, L_values=L_values, n_points=n_points)

        # ===================================================================== #
        #              SAVE MATRICES WITH THE RESULTS
        # ===================================================================== #
        np.savez(data_file_path_and_name, eigvals_holder=eigvals_holder, eigvecs_holder=eigvecs_holder)
        print("Done!")

    if preview:
        print("Non-extrapolated binding-energies:")
        print("\t[#]\t|\tValue [meV]")
        # fig, ax = plt.subplots(figsize=(10,10))
        # ax.imshow(np.(W_non_diag))
        # plt.show()
        # state_preview = 0
        # plot_wave_function(eigvecs_holder, state_preview)
        number_of_energies_to_show = 30
        for val_index in range(number_of_energies_to_show):
            print("\t%i\t|\t%.2f" % (val_index, values[val_index]-Egap))


if __name__ == '__main__':
    # import timeit
    # setup = "from __main__ import main"
    # Ntimes = 1
    # print(timeit.timeit("main()", setup=setup, number=Ntimes))
    main()
