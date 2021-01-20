#!/home/marcos/anaconda3/envs/numba/bin/python

import os
import hamiltonians as ham
import numpy as np

print('\n**********************************')
print("Hallo, Pythonista! Wie geht's dir!")
print('**********************************\n')

current_path = os.getcwd()

print("Você está em %s " % current_path)

file  = open('template_infile.txt','r');
lines = file.readlines()
file.close()

params = {}
for line in lines:
    list_line = line.split()
    if len(list_line) == 0 or list_line[0]=='##': continue
    else: params[list_line[0]] = list_line[2]
# print(params.keys())
# print(params.values())

# ========================================================================= #
##                   BUILDING FUNCTION & SOLVING FUNCTION
# ========================================================================= #
def build_bse_matrix(hamiltonian, Lk, Nk, Nsub):
    # First we have to define the grid:
    Kx, Ky, dk2 = wannier.define_grid_k(Lk, Nk)

    # Then, we need the eigenvalues and eigenvectors of our model for eack k-point
    # Values3D, Vectors4D = values_and_vectors(hamiltonian2x2, Kx, Ky, **hamiltonian_params)
    Values3D, Vectors4D = ham.values_and_vectors(hamiltonian, Kx, Ky)

    # Defining the potential
    V = ham.Rytova_Keldysh(dk2=dk2, r_0=r_0, epsilon=epsilon)

    ### THE BETHE-SALPETER MATRIX CONSTRUCTION:
    print("\tBuilding Potential matrix (Nk x Nk)... ")
    V_kk = potential_matrix(V, Kx, Ky, N_submesh, submesh_radius=submesh_limit)

    print("\tIncluding 'mixing' terms (Deltas)... ")
    W_non_diag = include_deltas(V_kk, Values3D, Vectors4D, N_submesh)

    print("\tIncluding 'pure' diagonal elements..")
    W_diag = diagonal_elements(Values3D)
    W_total = W_diag + W_non_diag

    return W_total

def diagonalize_bse(BSE_MATRIX):
    print("\tDiagonalizing the BSE matrix...")
    return LA.eigh(BSE_MATRIX)

# ========================================================================= #
##                            READING INPUT FILE
# ========================================================================= #
def from_dic_to_var(Ham, gamma, Egap, r0, mc, mv, alpha_option, epsilon, L_k, n_mesh, n_sub, submesh_radius, n_rec_states):

    ## MODEL HAMILTONIAN
    Ham           = eval('ham.' + Ham)  # chosen Hamiltonian has to be a class in 'hamiltonians.py'
    gamma         = float(gamma)        # meV*nm
    Egap          = float(Egap)         # meV
    r0            = float(r0)           # nm
    mc            = float(mc)           # conduction band effective mass
    mv            = float(mv)           # valence band effective mass
    alpha_option  = str(alpha_option)   # options: (zero, masses, corrected)
    epsilon       = float(epsilon)      # effective dielectric constant

    ## K-SPACE
    L_k             = float(L_k)                # the discretized k-space limits: k_x(y) ∈ [-L,L]
    n_mesh          = int(n_mesh)               # quantity of points in each direction
    n_sub           = int(n_sub)                # quantity of points in each of the submesh direction
    submesh_radius  = int(submesh_radius)       # in number of sites; radius of the region where the submesh will be considered

    return Ham, gamma, Egap, r0, mc, mv, alpha_option, epsilon, L_k, n_mesh, n_sub, submesh_radius, n_rec_states


# ========================================================================= #
##                            MAIN FUNCTION
# ========================================================================= #

def main():
    # ========================================================================= #
    ##              Outuput options:
    # ========================================================================= #
    save = True
    preview = True

    Ham, gamma, Egap, r0, mc, mv, alpha_option, epsilon, L_k, n_mesh, n_sub, submesh_radius, n_rec_states = from_dic_to_var(**params)

    print("Option adopted: %s" % alpha_option)

    if alpha_option == 'masses':
        alphac, alphav = 1/mc, 1/mv
    elif alpha_option == 'corrected':
        alphac = 1/mc + 1/hbar2_over2m * (gamma**2/Egap)
        alphav = 1/mv - 1/hbar2_over2m * (gamma**2/Egap)
    else:
        alphac, alphav = 0, 0

    ## TERMINAL OPTIONS:
    while len(sys.argv) > 1:
        option = sys.argv[1];               del sys.argv[1]
        if option == '-ns':
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

    # ========================================================================= #
    ## Define the Hamiltonian:
    # ========================================================================= #
    hamiltonian = Ham(alphac, alphav, Egap, gamma)


    # ========================================================================= #
    ## Matrices to hold the eigenvalues and the eigenvectors:
    # ========================================================================= #
    n_val_cond_comb = hamiltonian.condBands * hamiltonian.valeBands # Number of valence-conduction bands combinations
    eigvals_holder = np.zeros(number_of_recorded_states)
    eigvecs_holder = np.zeros((n_val_cond_comb * max_points**2, number_of_recorded_states), dtype=complex)

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
                        "_eps_" + str(epsilon) +
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
        number_of_energies_to_show = 15
        for val_index in range(number_of_energies_to_show):
            print("\t%i\t|\t%.2f" % (val_index, values[val_index]-Egap))


if __name__ == '__main__':
    # import timeit
    # setup = "from __main__ import main"
    # Ntimes = 1
    # print(timeit.timeit("main()", setup=setup, number=Ntimes))
    main()





# print("gamma = %f" % gamma)

# print("\nMODEL HAMILTONIAN: insert the desired Hamiltonian number")
# for ind_h in range(len(ham.list_of_hamiltonians)):
#     print("\t (%d) \t: %s" % (ind_h, ham.list_of_hamiltonians[ind_h]))
#
# while True:
#     ham_ind_chosen = int(input())
#     if ham_ind_chosen in range(len(ham.list_of_hamiltonians)) : break
#     else: print("Please choose a hamiltonian from the list.")
#
# H_str = ham.list_of_hamiltonians[ham_ind_chosen]
# print("You have chosen ", H_str)


# with open('template_infile.txt','r') as file:
#     for line in file:
#         print(line.split())

# eval('1e3')


# H = ham.H4x4
# str(H)
