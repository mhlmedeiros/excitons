import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import itertools as it
import physical_constants as const

plt.rcParams.update({
    "text.usetex": True,
    "font.serif": ["Palatino"],
    })

def hamiltonian2x2(kx, ky, E_gap=0.5, Gamma=1, Alpha_c=1, Alpha_v=-1):
    """
    Simple Hamiltonian to test the implementation:

    In its definition we have the effective masses "m_e" and "m_h",
    we also have the energy "gap". The model include one conduction-band
    and one valence-band, the couplings between these states are
    mediated by the named parameter "gamma".
    """
    k2 = kx**2 + ky**2
    H = np.array([[E_gap + const.hbar2_over2m * Alpha_c * k2, Gamma*(kx+1j*ky)],
                  [Gamma*(kx-1j*ky), const.hbar2_over2m * Alpha_v * k2]])
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

def results_arrays(data_path, info_path):
    data = np.load(data_path)
    info = np.load(info_path)
    info_content = [key for key in info.keys()]
    data_content = [key for key in data.keys()]
    # print("'data' contains: '%s' and '%s'" % (data_content[0], data_content[1]))
    # print("'info' contains: '%s' and '%s'" % (info_content[0], info_content[1]))
    # (# of eigenvals, # of discretizations, # system dimensions)
    # print("data['eigevals_holder'].shape = ", data['eigvals_holder'].shape)
    # (# of discretizations, )
    # print("info['n_points'].shape = ", info['n_points'].shape)
    # (# of system sizes)
    # print("info['L_values'].shape = ", info['L_values'].shape)
    # print("L_values = ", info['L_values'])
    # print("n_points = ", info['n_points'])
    return data['eigvals_holder'], data['eigvecs_holder'], info['n_points']

#==============================================================================#
def p_matrix(gamma,e_pol):
    ex,ey = e_pol
    e_p = ex + 1j*ey
    e_m = ex - 1j*ey
    P_matrix = np.array([
        [0.0, gamma*e_p],
        [gamma*e_m, 0.0]])
    return P_matrix

def dipole_vector(pol_versor, eigenvalues, eigenvectors, P_matrix):
    """
    This function generate a 2D array that contains the elements of the
    dipole matrix for each combination of kx, ky, c (conduction band), and
    v (valence band).
    """
    N_x, N_y, N_states = eigenvalues.shape

    N_k = N_x * N_y

    cond_vs, vale_vs = split_values(eigenvalues[0,0,:])
    cond_n = len(cond_vs) # Number of conduction bands
    vale_n = len(vale_vs) # Number of valence bands

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

    N_bse_energies, _, _ = eigvals.shape
    A_p_sums = np.zeros(N_bse_energies)
    C_0 = dk2/(const.EPSILON_0 * const.HBAR)
    # print(4 * np.pi**2 * C_0 / dk2)

    # A = eigvecs[:,0,0]
    # print("A.shape = ", A.shape)
    # print("eigenvals.shape = ", eigvals.shape)
    energies_without_discount = eigvals[:, 0, 0] + Egap

    for i in range(N_bse_energies):
        A = eigvecs[:,i,0]
        E = energies_without_discount[i]
        A_p_sums[i] = C_0/np.abs(E) * np.abs(A @ dipole_matrix_elements)**2
    return energies_without_discount, A_p_sums

def Lorentzian(x1, x0, Gamma):
    """
    Lorentzian function
    """
    L = 1/np.pi * (Gamma/2)/((x1-x0)**2 + (Gamma/2)**2)
    return L

def broadening(Energies, Deltas, Gamma_Lorentz):
    E_array = np.linspace(Energies[0]-100,Energies[-1]+100, 100000)
    A_total = np.zeros(len(E_array))
    for E_ind in range(len(Energies)):
        A_total += Deltas[E_ind] * Lorentzian(E_array, Energies[E_ind], Gamma_Lorentz(Energies[E_ind]))
    return E_array, A_total

def gamma_energy_dependent(E):
    Gamma1 = 10
    Gamma2 = 10
    Gamma3 = 10
    return Gamma1 + Gamma2/(1+np.exp((Egap-E)/Gamma3))

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

#==============================================================================#

def main():
    # PAULO-TEST:
    r0_chosen = 4.51 # nm (WSe2)
    epsilon_eff = 1
    gamma =  3.91504469e2# meV*nm ~ 2.6 eV*AA
    Egap = 1311.79 # meV ~ 2.4 eV
    alphac = 0
    alphav = 0

    ## SIMPLE-TEST:
    # r0_chosen = 5.0 # nm (WSe2)
    # epsilon_eff = 1
    # gamma =  0 # meV*nm = eV*AA
    # Egap = 1e3 # meV ~ 2.4 eV
    # mc = 0.2
    # mv = -0.4
    # alphac = 1/mc
    # alphav = 1/mv

    ## K-SPACE:
    k_array = np.linspace(-5,5,101)
    dk = k_array[1]-k_array[0]
    dk2 = dk**2
    Kx, Ky = np.meshgrid(k_array, k_array)

    ## CALCULATE THE HAMILTONIAN'S EIGENSTUFF:
    hamiltonian_params = dict(E_gap=Egap,
                            Alpha_c=alphac,
                            Alpha_v=alphav,
                            Gamma=gamma)
    VALORES, VETORES = values_and_vectors(hamiltonian2x2, Kx, Ky, **hamiltonian_params)

    ## LIGHT-POLARIZATION:
    e_x = 1
    e_y = 0
    e_a = np.array([e_x, e_y])
    # e_a = 1/LA.norm(e_a) * e_a

    ## DIPOLE MOMENTUM
    # gamma = 1e2 # meV*nm = eV*AA; gamma different from that one in the Hamiltonian for SIMPLE-TEST
    P_matrix = p_matrix(gamma, e_a)
    dipole_matrix = dipole_vector(e_a, VALORES, VETORES, P_matrix)


    # LOAD THE BSE-EIGENSTUFF
    ## DATA FOR SIMPLE-TEST:
    # data = "../Data/BSE_results/data_BSE_alphas_masses_gamma_0.0_eV_AA_Eg_1.0_eV_size_5_eps_1_discrete_101_sub_mesh_101_submesh_radius_1000.npz"
    # info = "../Data/BSE_results/info_BSE_alphas_masses_gamma_0.0_eV_AA_Eg_1.0_eV_size_5_eps_1_discrete_101_sub_mesh_101_submesh_radius_1000.npz"
    ## DATA FOR PAULO-TEST (JUPYTER)
    data = "../Data/BSE_results/data_BSE_alphas_zero_gamma_3.9150446899999998_eV_AA_Eg_1.31179_eV_size_5_eps_1_discrete_101_sub_mesh_101_with_smart_rytova_keldysh_with_potential_average_around_zero.npz"
    info = "../Data/BSE_results/info_BSE_alphas_zero_gamma_3.9150446899999998_eV_AA_Eg_1.31179_eV_size_5_eps_1_discrete_101_sub_mesh_101_with_smart_rytova_keldysh_with_potential_average_around_zero.npz"
    eigvals, eigvecs, n_points = results_arrays(data, info)

    x, abs_raw = absorption_raw(eigvals=eigvals,
                        eigvecs=eigvecs,
                        dipole_matrix_elements=dipole_matrix,
                        Egap=Egap,
                        dk2=dk2)
    # plot_absolute_wave(eigvecs)

    Energies = eigvals[:,0,0] + Egap
    E_array, absorption = broadening(Energies, abs_raw, lambda x: 10)
    ## PAULO-TEST:
    data_paulo = "../Paulo/absorption_tests/mesh_circ_massiveDirac_Nk50/lorentzian_absorption_exciton2D_eh001.dat"
    ## SIMPLE-TEST:
    # data_paulo = "../Paulo/absorption_tests/mesh_circ_effmass_Nk50/lorentzian_absorption_exciton2D_eh001.dat"

    fig, ax = plt.subplots(figsize=(10,8))
    # plot_deltas_absorption(x, abs_raw)
    plot_absorption(E_array, absorption, ax)
    plot_dat_together(data_paulo, ax)
    plt.legend(fontsize=24, loc="upper center")
    plt.show()


if __name__ == '__main__':
    main()
