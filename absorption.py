import os
import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import itertools as it
import physical_constants as const
import wannier_coulomb_numba as wannier
import hamiltonians as ham
import treat_files as files

plt.rcParams.update({
    "text.usetex": True,
    "font.serif": ["Palatino"],
    })

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
    # Gamma1, Gamma2, Gamma3, Egap = Gammas_tuple
    if option == 'V': function = Gamma_Lorentz_E_var(*Gammas_tuple)
    else: function = lambda x : Gamma1
    return function

def P_2x2(gamma, e_pol):
    ex,ey = e_pol
    e_p = ex + 1j*ey
    e_m = ex - 1j*ey
    P_matrix = np.array([
        [0.0, gamma*e_p],
        [gamma*e_m, 0.0]])
    return P_matrix

# TODO: IMPLEMENT P_4X4

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

    cond_inds = list(range(cond_n))                 # [ 0, 1, ... ,cond_n-1]
    vale_inds = list(range(-1,-1*(vale_n+1),-1))    # [-1,-2, ... , -vale_n]
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

def plot_deltas_absorption(energies_without_discount, A_p_sums):
    fig, ax = plt.subplots(figsize=(10,8))
    ax.scatter(energies_without_discount, A_p_sums,s=100)
    ax.vlines(x = energies_without_discount, ymin = 0 * A_p_sums, ymax = A_p_sums,colors='k')
    ax.hlines(y = 0, xmin = 500, xmax = 1400, colors = 'k')
    ax.set_title(r"Absorption spectra", fontsize=22)
    ax.tick_params(axis='x', labelsize=22)
    ax.tick_params(axis='y', labelsize=22)
    ax.yaxis.offsetText.set_fontsize(20)
    deltax = -15
    deltay = 0.5
    for i in [0,3,8,15]:
        ax.text(energies_without_discount[i]+deltax,
                 A_p_sums[i]+deltay,
                 "%.1f" % (A_p_sums[i]), color='k', fontsize=20)
    ax.set_xlim([500,1310])
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    # ax.set_xlim([1110,1240])
    ax.set_xlabel(r"$\hbar \omega$ [meV]", fontsize=24)
    # ax.set_xlabel(r"$\hbar \omega - E_{gap}$ [meV]", fontsize=24)
    ax.set_ylabel(r"A($\omega$)", fontsize=24)
    ax.set_ylim([-0.02*A_p_sums[0] , A_p_sums[0] + 2*deltay])
    ax.grid()
    plt.show()

    return 0

def plot_absorption(energies_without_discount, absorption_conv, ax, Egap=None):
    ax.plot(energies_without_discount, absorption_conv, label="Marcos", linewidth=2.5)
    # ax.vlines(x=1230, ymin=-1e3, ymax=1.5e4, colors=['red'], linestyles=['--'], label='')
    ax.set_title(r"Absorption spectra", fontsize=22)
    ax.tick_params(axis='x', labelsize=22)
    ax.tick_params(axis='y', labelsize=22)
    ax.yaxis.offsetText.set_fontsize(20)
    ax.set_xlabel(r"$\hbar \omega$ [meV]", fontsize=24)
    # ax.set_xlabel(r"$\hbar \omega - E_{gap}$ [meV]", fontsize=24)
    ax.set_ylabel(r"A($\omega$)", fontsize=24)
    # ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    # ax.set_xlim([500,1500])
    # ax.set_ylim([-5e-1,9])
    if Egap:
        ax.vlines(x=Egap, ymin=-0.5, ymax=9, colors=['k'], linestyles=['--'], label=r'$E_{gap}$')
        plt.legend(fontsize=24, loc="upper center")
    # plt.show()
    return ax

def plot_absolute_wave(eigvecs):
    kx = np.linspace(-5,5,101)
    psi_1s = eigvecs[:,0,0].reshape((101,101))
    psi_2s = eigvecs[:,3,0].reshape((101,101))
    pico1 = np.abs(psi_1s[50,50])
    pico2 = np.abs(psi_2s[50,50])
    fig, ax = plt.subplots(figsize=(10,8))
    ax.plot(kx, np.abs(psi_1s[:,50]), marker='o',linewidth=2, label="1s")
    ax.plot(kx, np.abs(psi_2s[:,50]), marker='o',linewidth=2, label="2s")
    ax.yaxis.offsetText.set_fontsize(20)
    ax.xaxis.offsetText.set_fontsize(20)

    ax.tick_params(axis='x', labelsize=22)
    ax.tick_params(axis='y', labelsize=22)
    ax.set_xlabel(r"k [nm$^{-1}$]", fontsize=24)
    ax.set_ylabel(r"$|\psi(k_x,k_y=0)|$", fontsize=24)
    ax.text(1.5, 1.1*pico1, r"%.4f" % (pico1), color='black', fontsize=20) #
    ax.annotate("",xy=(0, pico1), xytext=(1.5, 1.1*pico1), arrowprops=dict(width=0.3,shrink=0.05,color='C0'))
    ax.text(1.5, 0.9*pico2, r"%.4f" % (pico2), color='black', fontsize=20) #
    ax.annotate("",xy=(0, pico2), xytext=(1.5, 0.9*pico2), arrowprops=dict(width=0.3,shrink=0.05,color='C1'))
    ax.legend(fontsize=22)
    plt.tight_layout()
    plt.show()

def plot_dat_together(file_path_name, ax):
    E, abs_pol_x = [],[]
    with open(file_path_name, 'r') as file:
        line = 0 # start the counter
        for l in file:
            line += 1
            if line == 1:
                continue
                # print(*l.split())
            else:
                E.append(float(l.split()[0]))
                abs_pol_x.append(float(l.split()[1]))
        E = 1E3 * np.array(E) # From eV to meV
        abs_pol_x = np.array(abs_pol_x)
        ax.plot(E, abs_pol_x, "o",
                            linewidth=2,
                            color='C3',
                            markevery=1,
                            markersize=5,
                            label="Paulo")
    return 0


def main():
    print('\n******************************************************')
    print("                     ABSORPTION                   ")
    print('******************************************************\n')

    # =============================== #
    ##     READING THE INPUT FILES:
    # =============================== #
    #*********************************#
    output_name = 'results_absorption'
    if files.verify_output(output_name) == 'N': return 0
    #*********************************#
    # verify the existence of main input file: 'infile.txt'
    main_input_file =  'infile.txt'
    files.verify_essential_file(main_input_file)

    # verify the existence of the absorption input file: 'absorption_infile.txt'
    absorption_input_file = 'absorption_infile.txt'
    files.verify_file_or_template(absorption_input_file)

    # read both files
    params_master = files.read_file(main_input_file)
    params_absortion = files.read_file(absorption_input_file)
    Ham, gamma, Egap, r_0, mc, mv, alpha_option, epsilon, L_k, n_mesh, n_sub, submesh_radius, n_rec_states = files.from_dic_to_var(**params_master)
    pol_option, p_matrix, Gamma_option, Gamma1, Gamma2, Gamma3, padding, N_points_broad = files.from_dic_to_var_abs(**params_absortion)

    if alpha_option == 'masses':
        alphac, alphav = 1/mc, 1/mv
    elif alpha_option == 'corrected':
        alphac = 1/mc + 1/hbar2_over2m * (gamma**2/Egap)
        alphav = 1/mv - 1/hbar2_over2m * (gamma**2/Egap)
    else:
        alphac, alphav = 0, 0

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
    VALORES, VETORES = ham.values_and_vectors(hamiltonian, Kx, Ky)

    # =============================== #
    ##      LIGHT POLARIZATION:
    # =============================== #
    ## DIPOLE MOMENTUM
    e_a = pol_options(pol_option)
    p_matrix = eval(p_matrix)
    P_matrix = p_matrix(gamma, e_a)
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
                        Egap=Egap,
                        dk2=dk2)

    Gammas_tuple        = (Gamma1, Gamma2, Gamma3, Egap)
    Gamma_Lorentz       = Gamma_Lorentz_options(Gamma_option, Gammas_tuple)
    E_broad, Abs_broad  = broadening(E_raw, Abs_raw, Gamma_Lorentz, padding, N_points_broad)

    # =============================== #
    ##     SAVE THE RESULTS
    # =============================== #
    data_dic_to_save    = dict(E_raw=E_raw, Abs_raw=Abs_raw, E_broad=E_broad, Abs_broad=Abs_broad)
    files.output_file(output_name, data_dic_to_save)


    # plot_absolute_wave(eigvecs)
    # plot_deltas_absorption(x, Abs_raw)
    fig, ax = plt.subplots(figsize=(10,8))
    plot_absorption(E_broad, Abs_broad, ax)
    # plot_dat_together(data_paulo, ax)
    # plt.legend(fontsize=24, loc="upper center")
    plt.savefig('absorption_broad.png')
    plt.show()


if __name__ == '__main__':
    main()
