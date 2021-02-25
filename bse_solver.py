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
def build_bse_matrix(hamiltonian, potential_obj, exchange, r_0, d_chosen, grid_tuple, N_submesh, submesh_radius, scheme='H'):
    # RECOVER THE GRID INFORMATION:
    Kx, Ky, dk2 = grid_tuple

    # EIGENVALUES AND EIGENVECTORS OF AOR MODEL FOR EACH OF THE K-POINTS
    Values3D, Vectors4D = ham.values_and_vectors(hamiltonian, Kx, Ky)

    ### THE BETHE-SALPETER MATRIX CONSTRUCTION:
    Nk = Kx.shape[0]
    print("\tBuilding Potential matrix ({} x {})... ".format(Nk,Nk))
    V_kk = bse.potential_matrix(potential_obj, Kx, Ky, N_submesh, submesh_radius=submesh_radius)

    print("\tIncluding 'mixing' terms (Deltas)... ")
    BSE_MATRIX = bse.include_deltas(V_kk, Values3D, Vectors4D, N_submesh, hamiltonian, scheme=scheme)

    if bool(exchange):
        print("\tIncluding Exchange term...")
        # if isinstance(hamiltonian, ham.H4x4_Kormanyos_Fabian):
        #     Pi_matrix = ham.Pi_KormamyosFabian(hamiltonian)
        # elif isinstance(hamiltonian, ham.H4x4):
        #     Pi_matrix = ham.Pi4x4(gamma)
        # else:
        #     Pi_matrix = ham.Pi2x2(gamma)
        # if (hamiltonian.condBands, hamiltonian.valeBands) == (2,2): Pi_matrix = ham.Pi4x4(gamma)
        # else: Pi_matrix = ham.Pi2x2(gamma)
        # Pi_matrix = hamiltonian.Pi()
        BSE_MATRIX += dk2/(2*np.pi)**2 * bse.include_X(Values3D, Vectors4D, r_0, d_chosen, hamiltonian, scheme=scheme)

    print("\tIncluding 'pure' diagonal elements..")
    BSE_MATRIX += bse.diagonal_elements(Values3D, hamiltonian)

    return BSE_MATRIX

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
    output_name = 'results_bse'
    if files.verify_output(output_name) == 'N': return 0
    main_input_file = 'infile.txt'
    files.verify_file_or_template(main_input_file)
    # params = files.read_file(main_input_file)
    params = files.read_params(main_input_file)

    Ham, r_0, epsilon, exchange, d_chosen, L_k, n_mesh, n_sub, submesh_radius, n_rec_states = files.pop_out_model(params)
    # ham_id = params['Ham']
    # if ham_id == 'H4x4_Kormanyos_Fabian':
    #     Ham, E_c, E_v, alpha_up, alpha_dn, beta_up, beta_dn, gamma, delta_c, delta_v, kappa_up, kappa_dn, valey, epsilon, r_0, exchange, d_chosen, L_k, n_mesh, n_sub, submesh_radius, n_rec_states = files.from_dic_to_var_kormanyos_fabian(**params)
    #     Egap = E_c
    # else:
        # Ham, gamma, Egap, r_0, mc, mv, alpha_option, epsilon, exchange, d_chosen, L_k, n_mesh, n_sub, submesh_radius, n_rec_states = files.from_dic_to_var(**params)
        # # print("Option adopted: %s" % alpha_option)
        # if alpha_option == 'masses':
        #     alphac, alphav = 1/mc, 1/mv
        # elif alpha_option == 'corrected':
        #     alphac = 1/mc + 1/hbar2_over2m * (gamma**2/Egap)
        #     alphav = 1/mv - 1/hbar2_over2m * (gamma**2/Egap)
        # else:
        #     alphac, alphav = 0, 0
        #     if Egap == 0:
        #         # REVIEW: I THINK  THIS IS NOT NEEDED ANY MORE
        #         # When we don't have a gap it possible to have an error
        #         # due to the current strategy using "split_values" function,
        #         # this artificial gap prevents this problem.
        #         Egap = 1e-5


    # =============================== #
    ##    DEFINE THE K-SPACE GRID:
    # =============================== #
    Kx, Ky, dk2 = wannier.define_grid_k(L_k, n_mesh)
    grid = (Kx, Ky, dk2)

    # =============================== #
    ##    DEFINE THE HAMILTONIAN:
    # =============================== #
    # if main_input_file == 'infile_Kormanyos_Fabian.txt':
    #     hamiltonian =  Ham(E_c, E_v, alpha_up, alpha_dn,
    #                     beta_up, beta_dn, gamma, delta_c,
    #                     delta_v, kappa_up, kappa_dn, valey)
    # else: hamiltonian = Ham(alphac, alphav, Egap, gamma)

    hamiltonian = Ham(**params)
    potential_obj = ham.Rytova_Keldysh(dk2=dk2, r_0=r_0, epsilon=epsilon)


    # =============================== #
    #       BUILD THE MATRIX:
    # =============================== #
    # print('Exchange: ', exchange)
    answer = (input("The value for 'exchange' is {}; do you want to proceed? (y/N)\n".format(exchange)) or 'N').upper()
    if answer == 'N': return 0
    BSE_MATRIX = build_bse_matrix(hamiltonian, potential_obj, exchange, r_0, d_chosen, grid, n_sub, submesh_radius)

    # =============================== #
    #       DIAGONALIZATION:
    # =============================== #
    values, vectors = diagonalize_bse(BSE_MATRIX)

    # =============================== #
    ## PLACE HOLDERS FOR BSE-RESULTS:
    # =============================== #
    # print(dir(Ham))
    eigvals_holder = values[:n_rec_states] - 2.42e3
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
            print("\t%i\t|\t%.2f" % (val_index, values[val_index]-2.42e3))


if __name__ == '__main__':
    main()
    # open_template()
