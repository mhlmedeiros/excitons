import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import itertools as it

EPSILON_0 = 55.26349406             # e^2 GeV^{-1}fm^{-1} == e^2 (1e9 eV 1e-15 m)^{-1}
HBAR = 1.23984193/(2*np.pi)         # eV 1e-6 m/c
M_0  = 0.51099895000                # MeV/c^2
hbar2_over2m = HBAR**2/(2*M_0)*1e3  # meV nm^2

plt.rcParams.update({
    "text.usetex": True,
    "font.serif": ["Palatino"],
    })

def Lorentzian(x1, x0, Gamma):
    """
    Lorentzian function
    """
    L = 1/np.pi * (Gamma/2)/((x1-x0)**2 + (Gamma/2)**2)
    return L

def hamiltonian2x2(kx, ky, E_gap=0.5, Gamma=1, Alpha_c=1, Alpha_v=-1):
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

def polarization_vector(pol_versor, eigenvalues, eigenvectors, P_matrix):

    N_x, N_y, N_states = eigenvalues.shape

    N_k = N_x * N_y

    cond_vs, vale_vs = split_values(eigenvalues[0,0,:])
    cond_n = len(cond_vs)
    vale_n = len(vale_vs)

    cond_inds = list(range(cond_n))
    vale_inds = list(range(-1,-1*(vale_n+1),-1))
    k_inds = list(range(N_k))

    sum_components_pol_versor = pol_versor @ np.array([1,1j])

    print("n = {},\nm = {}".format(N_x, N_y))
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

    return sum_components_pol_versor * p_a_nm

def p_matrix(gamma,e_pol):
    ex,ey = e_pol
    e_p = ex + 1j*ey
    e_m = ex - 1j*ey
    P_matrix = np.array([
        [0.0, gamma*e_p],
        [gamma*e_m, 0.0]])

    return P_matrix

def results_arrays(data_path, info_path):
    data = np.load(data_path)
    info = np.load(info_path)

    info_content = [key for key in info.keys()]
    data_content = [key for key in data.keys()]
    print("'data' contains: '%s' and '%s'" % (data_content[0], data_content[1]))
    print("'info' contains: '%s' and '%s'" % (info_content[0], info_content[1]))

    # (# of eigenvals, # of discretizations, # system dimensions)
    print("data['eigevals_holder'].shape = ", data['eigvals_holder'].shape)
    # (# of discretizations, )
    print("info['n_points'].shape = ", info['n_points'].shape)
    # (# of system sizes)
    print("info['L_values'].shape = ", info['L_values'].shape)


    print("L_values = ", info['L_values'])
    print("n_points = ", info['n_points'])

    return data['eigvals_holder'], data['eigvecs_holder'], info['n_points']

def broadening(Energies, Deltas, Gamma_Lorentz):
    E_array = np.linspace(Energies[0]-100,Energies[-1]+100, 1000)
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

def absorption_raw(eigvals, eigvecs, dipole_matrix_elements, Egap, dk2):

    N_bse_energies, _, _ = eigvals.shape
    A_p_sums = np.zeros(N_bse_energies)
    C_0 = dk2/(EPSILON_0*HBAR)

    for i in range(N_bse_energies):
        A = eigvecs[:,i,0]
        E = eigvals[i,0,0] + Egap
        A_p_sums[i] = C_0/abs(E) * abs(np.sum(A*dipole_matrix_elements))**2
        # A_p_sums[i] = C_0 * abs(np.sum(A*dipole_matrix_elements))**2

    energies_without_discount = eigvals[:, 0, 0] + Egap

    return energies_without_discount, A_p_sums

def plot_deltas_absorption(energies_without_discount, A_p_sums):
    fig, ax = plt.subplots(figsize=(10,8))
    ax.scatter(energies_without_discount, A_p_sums,s=100)
    ax.vlines(x = energies_without_discount, ymin = 0 * A_p_sums, ymax = A_p_sums,colors='k')
    ax.hlines(y = 0, xmin = 800, xmax = 1400, colors = 'k')
    ax.set_title(r"Absorption spectra", fontsize=22)
    ax.tick_params(axis='x', labelsize=22)
    ax.tick_params(axis='y', labelsize=22)
    ax.yaxis.offsetText.set_fontsize(20)

    # deltax = -15
    # deltay = 5
    # for i in range(len(A_p_sums)):
    #     ax.text(energies_without_discount[i]+deltax,
    #              A_p_sums[i]+deltay,
    #              "%d" % (i), color='k', fontsize=20)

    # for i in [0,3,8,15]:
    #     ax.text(energies_without_discount[i]+deltax,
    #              A_p_sums[i]+deltay,
    #              "%.1f" % (A_p_sums[i]), color='k', fontsize=20)
    ax.set_xlim([800,1310])
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    # ax.set_xlim([1110,1240])
    ax.set_xlabel(r"$\hbar \omega$ [meV]", fontsize=24)
    # ax.set_xlabel(r"$\hbar \omega - E_{gap}$ [meV]", fontsize=24)
    ax.set_ylabel(r"A($\omega$)", fontsize=24)
    # ax.set_ylim([-1e1,1.5e2])
    ax.grid()
    plt.show()

    return 0


def main():
    ## PAULO'S TEST:
    r0_chosen = 4.51 # nm (WSe2)
    epsilon_eff = 1

    gamma =  3.91504469e2# meV*nm ~ 2.6 eV*AA
    Egap = 1311.79 # meV ~ 2.4 eV
    alphac = 0
    alphav = 0

    ## K-SPACE:
    k_array = np.linspace(-5,5,101)
    dk = k_array[1]-k_array[0]
    dk2 = dk**2
    Kx, Ky = np.meshgrid(k_array, k_array)

    ## CALCULATE THE HAMILTONIAN'S EIGENSTUFF:
    hamiltonian_params = dict(E_gap=Egap, Alpha_c=alphac, Alpha_v=alphav, Gamma=gamma)
    VALORES, VETORES = values_and_vectors(hamiltonian2x2, Kx, Ky, **hamiltonian_params)

    ## LIGHT-POLARIZATION:
    e_x = 1
    e_y = 0
    e_a = np.array([e_x, e_y])
    e_a = 1/LA.norm(e_a) * e_a

    ## DIPOLE MOMENTUM
    P_matrix = p_matrix(gamma, e_a)
    dipole_matrix = polarization_vector(e_a, VALORES, VETORES, P_matrix)

    # LOAD THE BSE-EIGENSTUFF
    data_5 = "../Data/BSE_results/data_BSE_alphas_zero_gamma_3.9150446899999998_eV_AA_Eg_1.31179_eV_size_5_eps_1_discrete_101_sub_mesh_101_with_smart_rytova_keldysh_with_potential_average_around_zero.npz"
    info_5 = "../Data/BSE_results/info_BSE_alphas_zero_gamma_3.9150446899999998_eV_AA_Eg_1.31179_eV_size_5_eps_1_discrete_101_sub_mesh_101_with_smart_rytova_keldysh_with_potential_average_around_zero.npz"
    eigvals_5, eigvecs_5, n_points_5 = results_arrays(data_5, info_5)

    x, y = absorption_raw(eigvals=eigvals_5, eigvecs=eigvecs_5, dipole_matrix_elements=dipole_matrix, Egap=Egap, dk2=dk2)
    plot_deltas_absorption(x,y)

    return 0


if __name__ == '__main__':
    main()
