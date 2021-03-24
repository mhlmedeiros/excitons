import os, sys
import numpy as np
import numpy.linalg as LA
import itertools as it
import physical_constants as const
import wannier_coulomb_numba as wannier
import hamiltonians as ham
import treat_files as files

class Gamma_Lorentz_E_var:
    def __init__(self, Gamma1, Gamma2, Gamma3, Egap):
        self.Gamma1 = Gamma1
        self.Gamma2 = Gamma2
        self.Gamma3 = Gamma3
        self.Egap   = Egap
    def __call__(self, E):
        Gamma1  = self.Gamma1
        Gamma2  = self.Gamma2
        Gamma3  = self.Gamma3
        Egap   = self.Egap
        return Gamma1 + Gamma2/(1+np.exp((Egap-E)/Gamma3))

def results_arrays(data_path):
    data = np.load(data_path)
    # data_content = [key for key in data.keys()]
    # print(data_content)
    # print(type(data_content[0]))
    return data['eigvals_holder'], data['eigvecs_holder']

#==============================================================================#
def pol_options(option):
    if      option == 'y'       : e_x, e_y = 0, 1;
    elif    option == 'c_plus'  : e_x, e_y = 1, 1;
    elif    option == 'c_minus' : e_x, e_y = 1,-1;
    else                        : e_x, e_y = 1, 0; # (x) default
    e_a = np.array([e_x, e_y])
    return 1/LA.norm(e_a) * e_a

def Gamma_Lorentz_options(option, Gammas_tuple):
    Gamma1, Gamma2, Gamma3, Egap = Gammas_tuple
    if option == 'V': function = Gamma_Lorentz_E_var(*Gammas_tuple)
    else: function = lambda x : Gamma1
    return function

def p_matrix(Hamiltonian, e_pol):
    Pix, Piy = Hamiltonian.Pi()
    P_MATRIX = e_pol[0] * Pix + e_pol[1] * Piy
    return P_MATRIX

def dipole_vector(pol_versor, eigenvalues, eigenvectors, P_matrix, Hamiltonian):
    """
    This function generate a 2D array that contains the elements of the
    dipole matrix for each combination of kx, ky, c (conduction band), and
    v (valence band).
    """
    N_x, N_y, N_states = eigenvalues.shape

    N_k = N_x * N_y

    cond_n = Hamiltonian.condBands # Number of conduction bands
    vale_n = Hamiltonian.valeBands # Number of valence bands

    # cond_inds = list(range(cond_n))                 # [ 0, 1, ... ,cond_n-1] # OLD VERSION
    # vale_inds = list(range(-1,-1*(vale_n+1),-1))    # [-1,-2, ... , -vale_n] # OLD VERSION
    cond_inds = list(range(-1,-1*(cond_n+1),-1))    # [-1,-2, ... , -vale_n] # NEW VERSION
    vale_inds = list(range(vale_n))                 # [ 0, 1, ... ,cond_n-1] # NEW VERSION
    k_inds = list(range(N_k))                       # [ 0, 1, ... , N_k ]

    # sum_components_pol_versor = pol_versor @ np.array([1,1j])
    # print("n = {},\nm = {}".format(N_x, N_y))
    # print(vale_inds)
    # print(cond_inds)
    p_a_nm = np.zeros(N_x * N_y * cond_n * vale_n, dtype=complex)

    count = 0
    for (i,j,v,c) in it.product(range(N_y), range(N_x), vale_inds, cond_inds):
        # THE ORDER OF THE FOR-LOOPS IS IMPORTANT: KX MUST RUN FASTER THAN KY
        # SINCE THAT IS THE ORDER EXPECTED FOR "A FLATTED" VERSION OF KX_MATRIX
        # AND KY_MATRIX:
        phi_valence = eigenvectors[i,j,:,v]
        phi_conduction = eigenvectors[i,j,:,c]
        p_a_nm[count] = phi_valence.conj() @ (P_matrix @ phi_conduction)
        count += 1

    # return sum_components_pol_versor * p_a_nm
    return p_a_nm

def absorption_raw(eigvals, eigvecs, dipole_matrix_elements, Egap, dk2):

    N_bse_energies = len(eigvals)
    A_p_sums = np.zeros(N_bse_energies)
    C_0 = dk2/(const.EPSILON_0 * const.HBAR)
    # print(4 * np.pi**2 * C_0 / dk2)
    # A = eigvecs[:,0,0]
    # print("A.shape = ", A.shape)
    # print("eigenvals.shape = ", eigvals.shape)
    energies_without_discount = eigvals + Egap

    for i in range(N_bse_energies):
        A = eigvecs[:,i]
        E = energies_without_discount[i]
        A_p_sums[i] = C_0/np.abs(E) * np.abs(A @ dipole_matrix_elements)**2
    return energies_without_discount, A_p_sums

def Lorentzian(x1, x0, Gamma):
    """
    Lorentzian function
    """
    L = 1/np.pi * (Gamma/2)/((x1-x0)**2 + (Gamma/2)**2)
    return L

def broadening(Energies, Deltas, Gamma_Lorentz, padding, N_points_broadening):
    E_array = np.linspace(Energies[0]-padding,Energies[-1]+padding, N_points_broadening)
    A_total = np.zeros(len(E_array))
    for E_ind in range(len(Energies)):
        A_total += Deltas[E_ind] * Lorentzian(E_array, Energies[E_ind], Gamma_Lorentz(Energies[E_ind]))
    return E_array, A_total

#==============================================================================#

def main():
    # print('\n******************************************************')
    # print("                     ABSORPTION                   ")
    # print('******************************************************\n')

    # =============================== #
    ##     READING THE INPUT FILES:
    # =============================== #
    #*********************************#
    # output_name = 'results_absorption'
    # if files.verify_output(output_name) == 'N': return 0
    #*********************************#
    # verify the existence of main input file: 'infile.txt'
    main_input_file =  'infile.txt'
    # files.verify_essential_file(main_input_file)

    # verify the existence of the absorption input file: 'absorption_infile.txt'
    absorption_input_file = 'absorption_infile.txt'
    # files.verify_file_or_template(absorption_input_file)

    # read both files
    # params_master = files.read_file(main_input_file)
    # params_abs    = files.read_file(absorption_input_file)
    params_master = files.read_params(main_input_file)
    params_abs    = files.read_params(absorption_input_file)

    # =============================== #
    ##  TERMINAL OPTIONS (OVERRIDE):
    # =============================== #
    if len(sys.argv) == 4:
        # USED TO GENERATE THE DATA BASIS USING WANNIER EQUATION
        return 0 # THERE IS NO ABSORPTION FOR WANNIER EQUATION RESULTS
    elif len(sys.argv) == 2:
        # USED TO GENERATE THE DATA FOR THE 3-BANDS MODEL
        # USED TO GENERATE THE DATA FOR THE 3-BANDS MODEL
        epsilon_sub         = float(sys.argv[1])
        params_master['epsilon']   = epsilon_sub
        output_name = "results_absorption_3Bands_eps_{}".format(epsilon_sub)
    else:
        output_name = 'results_bse'


    # Ham, gamma, Egap, r_0, mc, mv, alpha_option, epsilon, exchange, d_chosen, L_k, n_mesh, n_sub, submesh_radius, n_rec_states = files.from_dic_to_var(**params_master)
    # pol_option, p_matrix, Gamma_option, Gamma1, Gamma2, Gamma3, padding, N_points_broad = files.from_dic_to_var_abs(**params_absortion)
    #
    # if alpha_option == 'masses':
    #     alphac, alphav = 1/mc, 1/mv
    # elif alpha_option == 'corrected':
    #     alphac = 1/mc + 1/hbar2_over2m * (gamma**2/Egap)
    #     alphav = 1/mv - 1/hbar2_over2m * (gamma**2/Egap)
    # else:
    #     alphac, alphav = 0, 0

    # =============================== #
    ##    POP OUT ALL INFORMATION:
    # =============================== #
    Ham, r_0, epsilon, exchange, d_value, Lk, Nk, Nsub, Rsub, Nsaved = files.pop_out_model(params_master)
    pol_option      = params_abs['pol_option']
    # p_matrix        = eval('ham' + (params_abs['p_matrix']))
    Gamma_option    = params_abs['Gamma_option']
    Gamma1          = params_abs['Gamma1']
    Gamma2          = params_abs['Gamma2']
    Gamma3          = params_abs['Gamma3']
    padding         = params_abs['padding']
    N_points_broad  = params_abs['N_points_broad']

    # =============================== #
    ##    DEFINE THE K-SPACE GRID:
    # =============================== #
    Kx, Ky, dk2 = wannier.define_grid_k(Lk, Nk)
    grid = (Kx, Ky, dk2)

    # =============================== #
    ##    DEFINE THE HAMILTONIAN:
    # =============================== #
    hamiltonian      = Ham(**params_master)
    potential_obj    = ham.Rytova_Keldysh(dk2=dk2, r_0=r_0, epsilon=epsilon)
    VALORES, VETORES = ham.values_and_vectors(hamiltonian, Kx, Ky)

    # =============================== #
    ##      LIGHT POLARIZATION:
    # =============================== #
    ## DIPOLE MOMENTUM
    e_a = pol_options(pol_option)
    # p_matrix = eval(p_matrix)
    P_matrix = p_matrix(hamiltonian, e_a)
    dipole_matrix = dipole_vector(e_a, VALORES, VETORES, P_matrix, hamiltonian)

    # =============================== #
    ##     LOAD THE BSE-EIGENSTUFF
    # =============================== #
    results_file    = 'results_bse.npz'
    files.verify_essential_file(results_file)
    path = os.getcwd()
    data = path + '/' + results_file
    eigvals, eigvecs = results_arrays(data)


    # =============================== #
    ##     CALCULATE THE ABSORPTION
    # =============================== #
    E_raw, Abs_raw      = absorption_raw(eigvals=eigvals,
                        eigvecs=eigvecs,
                        dipole_matrix_elements=dipole_matrix,
                        Egap=hamiltonian.Egap,
                        dk2=dk2)

    Gammas_tuple        = (Gamma1, Gamma2, Gamma3, hamiltonian.Egap)
    Gamma_Lorentz       = Gamma_Lorentz_options(Gamma_option, Gammas_tuple)
    E_broad, Abs_broad  = broadening(E_raw, Abs_raw, Gamma_Lorentz, padding, N_points_broad)

    # =============================== #
    ##     SAVE THE RESULTS
    # =============================== #
    data_dic_to_save    = dict(E_raw=E_raw, Abs_raw=Abs_raw, E_broad=E_broad, Abs_broad=Abs_broad)
    files.output_file(output_name, data_dic_to_save)





if __name__ == '__main__':
    main()
