
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm

plt.rcParams.update({
    "text.usetex": True,
    "font.serif": ["Palatino"],
    })


def plot_deltas_absorption(energies_without_discount, A_p_sums,ax=None):
    if ax == None: fig, ax = plt.subplots(figsize=(10,8))
    ax.scatter(energies_without_discount, A_p_sums,s=100)
    ax.vlines(x = energies_without_discount, ymin = 0 * A_p_sums, ymax = A_p_sums,colors='k')
    ax.hlines(y = 0, xmin = energies_without_discount[0],
                     xmax = energies_without_discount[-1], colors = 'k')
    ax.set_title(r"Absorption spectra", fontsize=22)
    ax.tick_params(axis='x', labelsize=22)
    ax.tick_params(axis='y', labelsize=22)
    ax.yaxis.offsetText.set_fontsize(20)
    deltax = -15
    deltay = 0.5
    # for i in [0,3,8,15]:
    #     ax.text(energies_without_discount[i]+deltax,
    #              A_p_sums[i]+deltay,
    #              "%.1f" % (A_p_sums[i]), color='k', fontsize=20)
    # ax.set_xlim([500,1310])
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    # ax.set_xlim([1110,1240])
    ax.set_xlabel(r"$\hbar \omega$ [meV]", fontsize=24)
    # ax.set_xlabel(r"$\hbar \omega - E_{gap}$ [meV]", fontsize=24)
    ax.set_ylabel(r"A($\omega$)", fontsize=24)
    # ax.set_ylim([-0.02*A_p_sums[0] , A_p_sums[0] + 2*deltay])
    ax.grid()
    # plt.show()

    return 0

def plot_absorption(energies_without_discount, absorption_conv, ax, Egap=None):
    ax.plot(energies_without_discount, absorption_conv, label="Absorption", linewidth=2.5)
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

#===============================================================================

def main():
    # READ THE DATA:
    data = np.load('../3BandsModel/results_coupling/cluster_results_01/results_absorption_3Bands_eps_2.0.npz')
    E_broad   = data['E_broad']
    Abs_broad = data['Abs_broad']
    E_raw = data['E_raw']
    Abs_raw = data['Abs_raw']

    # PLOT THE RESULTS:
    fig, ax = plt.subplots(ncols=1,nrows=2, figsize=(10,8))
    plot_absorption(E_broad, Abs_broad, ax[0])
    ax[0].vlines(x=[2.4e3],ymin=0, ymax=0.6, color='k', linestyle='--', label=r'$E_{gap}$')
    ax[0].set_xlim([2000,2600])
    ax[0].set_ylim([0,0.6])
    ax[0].legend(fontsize=20)
    # plt.savefig('absorption_broad_3Bands_eps_2_exchange.png', dpi=300)
    # plt.show()
    #
    # fig, ax = plt.subplots(figsize=(10,8))
    # plot_deltas_absorption(E_raw, Abs_raw, ax)
    # ax.vlines(x=[2.4e3],ymin=0, ymax=10, color='k', linestyle='--', label=r'$E_{gap}$')
    # ax.set_xlim([2100,2600])
    # ax.set_ylim([-0.5,10])
    # ax.legend(fontsize=20)
    # # plt.savefig('absorption_raw_3Bands_eps_2_exchange.png', dpi=300)
    # plt.show()


    data_bse = np.load('../3BandsModel/results_coupling/cluster_results_01/results_excitons_3Bands_eps_2.0.npz')

    wave_functions = data_bse['eigvecs_holder']
    N_entries, N_states = wave_functions.shape

    Ac2v = np.empty((N_states,))
    Ac1v = np.empty((N_states,))
    for i in range(N_states):
        Ac2v[i] = np.sum(np.abs(wave_functions[0::2, i])**2)
        Ac1v[i] = np.sum(np.abs(wave_functions[1::2, i])**2)

    # fig, ax = plt.subplots(ncols=1,nrows=1, figsize=[10,3])
    ax[1].scatter(E_raw, 0*Ac2v+0.1, s=4*(100*Ac2v), c='C0')
    ax[1].scatter(E_raw, 0*Ac1v, s=4*(100*Ac1v), c='C1')
    ax[1].vlines(x=2.4e3, ymin=-0.1, ymax=0.2, linestyle='--', color='k')
    ax[1].hlines(y=0.1, xmin=2e3, xmax=2.6e3, linestyle='-', color='C0', label=r'$|\mathcal{A}_{c_2v}|^2$')
    ax[1].hlines(y=0, xmin=2e3, xmax=2.6e3, linestyle='-', color='C1', label=r'$|\mathcal{A}_{c_1v}|^2$')
    ax[1].legend(fontsize=18)
    ax[1].set_xlim([2000,2600])
    ax[1].set_ylim([-0.2,0.3])
    ax[1].axis('off')
    plt.tight_layout()
    plt.show()

if __name__=='__main__':
    main()
