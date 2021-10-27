#!/home/marcos/anaconda3/envs/numba/bin/python

import sys
import numpy as np
import scipy.linalg as LA
import scipy.sparse.linalg as LAS
import hamiltonians as ham
import treat_files as files
import bethe_salpeter_equation as bse

# ========================================================================= #
##                   BUILDING FUNCTION & SOLVING FUNCTION
# ========================================================================= #
def build_wannier_matrix(hamiltonian,potential_obj,r_0,grid_tuple,N_submesh,submesh_radius):
    # RECOVER THE GRID INFORMATION:
    Kx, Ky, dk2 = grid_tuple
    ## KINETIC CONTRIBUTION
    wannier_matrix = ham.kinetic_wannier(hamiltonian, Kx, Ky)
    ## POTENTIAL CONTRIBUTION
    wannier_matrix += bse.potential_matrix(potential_obj, Kx, Ky, N_submesh, submesh_radius=submesh_radius)
    return wannier_matrix

def build_bse_matrix(hamiltonian, potential_obj, exchange, r_0, d_chosen, grid_tuple, N_submesh, submesh_radius, scheme='H'):
    # RECOVER THE GRID INFORMATION:
    Kx, Ky, dk2 = grid_tuple

    # EIGENVALUES AND EIGENVECTORS OF AOR MODEL FOR EACH OF THE K-POINTS
    Values3D, Vectors4D = ham.values_and_vectors(hamiltonian, Kx, Ky)

    ### THE BETHE-SALPETER MATRIX CONSTRUCTION:
    Nk = Kx.shape[0]
    # print("\tBuilding Potential matrix ({} x {})... ".format(Nk,Nk))
    V_kk = bse.potential_matrix(potential_obj, Kx, Ky, N_submesh, submesh_radius=submesh_radius)

    # print("\tIncluding 'mixing' terms (Deltas)... ")
    BSE_MATRIX = bse.include_deltas(V_kk, Values3D, Vectors4D, N_submesh, hamiltonian, scheme=scheme)

    if bool(exchange):
        # print("\tIncluding Exchange term...")
        BSE_MATRIX += dk2/(2*np.pi)**2 * bse.include_X(Values3D, Vectors4D, r_0, d_chosen, hamiltonian, scheme=scheme)

    # print("\tIncluding 'pure' diagonal elements..")
    BSE_MATRIX += bse.diagonal_elements(Values3D, hamiltonian)

    return BSE_MATRIX

def diagonalize_bse(BSE_MATRIX, n_states, arpack=False):
    # print("\tDiagonalizing the BSE matrix...")
    bse_matrix_dim,_ = BSE_MATRIX.shape
    n_saved_states = min(bse_matrix_dim, n_states)
    if arpack:
        values, vectors = LAS.eigsh(BSE_MATRIX, k=n_saved_states, which='SA')
    else:
        v, w = LA.eigh(BSE_MATRIX)
        values  = v[ : n_saved_states]
        vectors = w[ :, : n_saved_states]
    return values, vectors

# ========================================================================= #
##                            PARSING THE INPUTS
# ========================================================================= #
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Path for the input file")
    parser.add_argument("output_file", help='First part of the output file name')
    parser.add_argument("-e", "--epsilon", default=1.0, type=float, help="Dielectric constant")
    parser.add_argument("-ns", "--not_save", action="store_false", help="Tell to not save the bse solutions")
    parser.add_argument("-p", "--preview", action="store_true", help="Show the first bse eigenvals")
    args = parser.parse_args()
    input_file  = args.input_file
    output_file = args.output_file
    epsilon = args.epsilon
    save = not args.not_save
    preview = args.preview
    return input_file, output_file, epsilon, save, preview

# ========================================================================= #
##                            MAIN FUNCTION
# ========================================================================= #
def main():
    # =============================== #
    ##     READING ARGUMENTS:
    # =============================== #
    input_file, output_file, epsilon, save, preview = parse_arguments()
    params_master = files.read_params(input_file)
    params = files.read_params(input_file)


    # =============================== #
    ##  TERMINAL OPTIONS (OVERRIDE):
    # =============================== #
    if len(sys.argv) == 4:
        # USED TO GENERATE THE DATA BASIS USING WANNIER EQUATION
        m_eff_sub           = float(sys.argv[1])
        epsilon_sub         = float(sys.argv[2])
        r0_sub              = float(sys.argv[3])
        params['m_1']       = m_eff_sub
        params['epsilon']   = epsilon_sub
        params['r0']        = r0_sub
        preview             = False
        output_name = "results_excitons_wannier_meff_{}_eps_{}_r0_{}".format(m_eff_sub, epsilon_sub, r0_sub)
    elif len(sys.argv) == 2:
        # USED TO GENERATE THE DATA FOR THE 3-BANDS MODEL
        epsilon_sub         = float(sys.argv[1])
        params['epsilon']   = epsilon_sub
        preview             = False
        save                = True
        output_name = "results_excitons_3Bands_eps_{}".format(epsilon_sub)
    else:
        output_name = 'results_bse'

    # =============================== #
    ##      POP THE PARAMS OUT:
    # =============================== #
    # SINCE WE'RE VARING THE epsilon VALUES IN THIS CODE, WE WILL DISCARD THE
    # VALUE GIVEN IN THE INPUT FILE AND ADOPT THAT ONE FROM PARSER
    Ham, r_0, epsilon_file, exchange, d_chosen, L_k, n_mesh, n_sub, submesh_radius, n_rec_states = files.pop_out_model(params)

    # =============================== #
    ##    DEFINE THE K-SPACE GRID:
    # =============================== #
    Kx, Ky, dk2 = bse.define_grid_k(L_k, n_mesh)
    grid = (Kx, Ky, dk2)

    # =============================== #
    ##    DEFINE THE HAMILTONIAN:
    # =============================== #
    hamiltonian = Ham(**params)
    potential_obj = ham.Rytova_Keldysh(dk2=dk2, r_0=r_0, epsilon=epsilon)


    # =============================== #
    #       BUILD THE MATRIX:
    # =============================== #
    if hamiltonian.condBands != 0:
        MAIN_MATRIX = build_bse_matrix(hamiltonian, potential_obj, exchange, r_0, d_chosen, grid, n_sub, submesh_radius)
    else:
        MAIN_MATRIX = build_wannier_matrix(hamiltonian, potential_obj, r_0, grid, n_sub, submesh_radius)

    # =============================== #
    #       DIAGONALIZATION:
    # =============================== #
    values, vectors = diagonalize_bse(MAIN_MATRIX, n_rec_states, arpack=False)

    # =============================== #
    ## PLACE HOLDERS FOR BSE-RESULTS:
    # =============================== #
    eigvals_holder = values - hamiltonian.Egap
    eigvecs_holder = vectors

    # =============================== #
    #       SAVING THE RESULTS:
    # =============================== #
    output_name = output_file + "_eps_{}".format(epsilon)
    data_dic_to_save = dict(eigvals_holder=eigvals_holder, eigvecs_holder=eigvecs_holder)
    if save: files.output_file(output_name, data_dic_to_save)

    if preview:
        print("Non-extrapolated binding-energies:")
        print("\t[#]\t|\tValue [meV]")
        number_of_energies_to_show = min(15, n_rec_states)
        for val_index in range(number_of_energies_to_show):
            print("\t%i\t|\t%.2f" % (val_index, values[val_index]-hamiltonian.Egap))


if __name__ == '__main__':
    main()
