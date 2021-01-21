#!/home/marcos/anaconda3/envs/numba/bin/python

import numpy as np
import scipy.linalg as LA
import hamiltonians as ham
import wannier_coulomb_numba as wannier
import bethe_salpeter_equation as bse
import os, sys, subprocess, shutil, glob

print('\n**********************************')
print("Hallo, Pythonista! Wie geht's dir!")
print('**********************************\n')

'paz'.upper()
# ========================================================================= #
##                            'READ INPUT FILE
# ========================================================================= #
def open_template():
    path_src = '/home/marcos/Documents/DAAD_Research/excitons/excitons_python'
    template = 'template_infile.txt'
    path_dst = os.getcwd()
    specific = 'infile.txt'
    file_template = path_src + '/' + template
    file_specific = path_dst + '/' + specific
    shutil.copy(file_template, file_specific)
    subprocess.call(['vim', specific])
    return 0

def open_infile():
    path_dst = os.getcwd()
    specific = 'infile.txt'
    file_specific = path_dst + '/' + specific
    subprocess.call(['vim', specific])
    return 0

def read_file():
    current_path = os.getcwd()
    print("Você está em %s " % current_path)
    file  = open('infile.txt','r');
    lines = file.readlines()
    file.close()

    params = {}
    for line in lines:
        list_line = line.split()
        if len(list_line) == 0 or list_line[0]=='##': continue
        else: params[list_line[0]] = list_line[2]
    return params

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

    ## SAVING OPTIONS:
    n_rec_states    = int(n_rec_states)         # number of states to be recorded

    return Ham, gamma, Egap, r0, mc, mv, alpha_option, epsilon, L_k, n_mesh, n_sub, submesh_radius, n_rec_states


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
##                         WRITING THE OUTPUT FILE
# ========================================================================= #
def write_file(L_k, n_mesh, eigvals_holder, eigvecs_holder):
    current_path = os.getcwd()
    output_name = "/results_bse"
    complete_path = current_path + output_name
    print("\n\nSaving...")
    np.savez(complete_path, L_k=L_k, n_mesh=n_mesh, eigvals_holder=eigvals_holder, eigvecs_holder=eigvecs_holder)
    print("Done!")


# ========================================================================= #
##                            MAIN FUNCTION
# ========================================================================= #
def main():
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

    filelist = glob.glob('*.txt')
    if len(filelist) == 0 : open_template()
    else: open_infile()
    params = read_file()

    Ham, gamma, Egap, r_0, mc, mv, alpha_option, epsilon, L_k, n_mesh, n_sub, submesh_radius, n_rec_states = from_dic_to_var(**params)
    # print("Option adopted: %s" % alpha_option)
    if alpha_option == 'masses':
        alphac, alphav = 1/mc, 1/mv
    elif alpha_option == 'corrected':
        alphac = 1/mc + 1/hbar2_over2m * (gamma**2/Egap)
        alphav = 1/mv - 1/hbar2_over2m * (gamma**2/Egap)
    else:
        alphac, alphav = 0, 0
    if Egap == 0:
        # When we don't have a gap it possible to have an error
        # due to the current strategy using "split_values" function,
        # this artificial gap prevent this problem.
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

    #*********************************#
    answer = input("Do you want to proceed with calculations now? (Y/n)\n") or 'Y'
    if answer.upper() == 'N': exit()
    #*********************************#

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
    if save: write_file(L_k, n_mesh, eigvals_holder, eigvecs_holder)

    if preview:
        print("Non-extrapolated binding-energies:")
        print("\t[#]\t|\tValue [meV]")
        number_of_energies_to_show = 15
        for val_index in range(number_of_energies_to_show):
            print("\t%i\t|\t%.2f" % (val_index, values[val_index]-Egap))


if __name__ == '__main__':
    main()
    # open_template()
