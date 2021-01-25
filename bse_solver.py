#!/home/marcos/anaconda3/envs/numba/bin/python

import numpy as np
import scipy.linalg as LA
import hamiltonians as ham
import wannier_coulomb_numba as wannier
import treat_files as files
import bethe_salpeter_equation as bse
import os, sys, subprocess, shutil, glob



# ========================================================================= #
##                   BUILDING FUNCTION & SOLVING FUNCTION
# ========================================================================= #
def build_bse_matrix(hamiltonian, potential_obj, grid_tuple, N_submesh, submesh_radius):
    # RECOVER THE GRID INFORMATION:
    Kx, Ky, dk2 = grid_tuple

    # EIGENVALUES AND EIGENVECTORS OF AOR MODEL FOR EACH OF THE K-POINTS
    Values3D, Vectors4D = ham.values_and_vectors(hamiltonian, Kx, Ky)

    ### THE BETHE-SALPETER MATRIX CONSTRUCTION:
    Nk = Kx.shape[0]
    print("\tBuilding Potential matrix ({} x {})... ".format(Nk,Nk))
    V_kk = bse.potential_matrix(potential_obj, Kx, Ky, N_submesh, submesh_radius=submesh_radius)

    print("\tIncluding 'mixing' terms (Deltas)... ")
    W_non_diag = bse.include_deltas(V_kk, Values3D, Vectors4D, N_submesh)

    print("\tIncluding 'pure' diagonal elements..")
    W_diag = bse.diagonal_elements(Values3D)
    W_total = W_diag + W_non_diag

    return W_total

def diagonalize_bse(BSE_MATRIX):
    print("\tDiagonalizing the BSE matrix...")
    return LA.eigh(BSE_MATRIX)



# ========================================================================= #
##                            MAIN FUNCTION
# ========================================================================= #
def main():
    print('\n**************************************************')
    print("                      BSE-SOLVER                  ")
    print('**************************************************\n')

    # =============================== #
    ##          Outuput options:
    # =============================== #
    save = True
    preview = True

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

    # =============================== #
    ##     READING THE INPUT FILE:
    # =============================== #
    #*********************************#
    output_name = 'results_bse'
    if files.verify_output(output_name) == 'N': return 0
    #*********************************#
    main_input_file = 'infile.txt'
    files.verify_file_or_template(main_input_file)
    params = files.read_file(main_input_file)

    Ham, gamma, Egap, r_0, mc, mv, alpha_option, epsilon, L_k, n_mesh, n_sub, submesh_radius, n_rec_states = files.from_dic_to_var(**params)
    # print("Option adopted: %s" % alpha_option)
    if alpha_option == 'masses':
        alphac, alphav = 1/mc, 1/mv
    elif alpha_option == 'corrected':
        alphac = 1/mc + 1/hbar2_over2m * (gamma**2/Egap)
        alphav = 1/mv - 1/hbar2_over2m * (gamma**2/Egap)
    else:
        alphac, alphav = 0, 0
    if Egap == 0:
        # REVIEW: I THINK  THIS IS NOT NEEDED ANY MORE
        # When we don't have a gap it possible to have an error
        # due to the current strategy using "split_values" function,
        # this artificial gap prevents this problem.
        Egap = 1e-5

    # =============================== #
    ##    DEFINE THE K-SPACE GRID:
    # =============================== #
    Kx, Ky, dk2 = wannier.define_grid_k(L_k, n_mesh)
    grid = (Kx, Ky, dk2)

    # =============================== #
    ##    DEFINE THE HAMILTONIAN:
    # =============================== #
    hamiltonian = Ham(alphac, alphav, Egap, gamma)
    potential_obj = ham.Rytova_Keldysh(dk2=dk2, r_0=r_0, epsilon=epsilon)


    # =============================== #
    #       BUILD THE MATRIX:
    # =============================== #
    BSE_MATRIX = build_bse_matrix(hamiltonian, potential_obj, grid, n_sub, submesh_radius)

    # =============================== #
    #       DIAGONALIZATION:
    # =============================== #
    values, vectors = diagonalize_bse(BSE_MATRIX)

    # =============================== #
    ## PLACE HOLDERS FOR BSE-RESULTS:
    # =============================== #
    eigvals_holder = values[:n_rec_states] - Egap
    eigvecs_holder = vectors[:,:n_rec_states]

    # =============================== #
    #       SAVING THE RESULTS:
    # =============================== #
    data_dic_to_save = dict(eigvals_holder=eigvals_holder, eigvecs_holder=eigvecs_holder)
    if save: files.output_file(output_name, data_dic_to_save)

    if preview:
        print("Non-extrapolated binding-energies:")
        print("\t[#]\t|\tValue [meV]")
        number_of_energies_to_show = 15
        for val_index in range(number_of_energies_to_show):
            print("\t%i\t|\t%.2f" % (val_index, values[val_index]-Egap))


if __name__ == '__main__':
    main()
    # open_template()
