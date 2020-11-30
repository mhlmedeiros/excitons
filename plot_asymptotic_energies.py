import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
import matplotlib as mpl
from matplotlib import cm
from scipy import stats

plt.rcParams.update({
    "text.usetex": True,
    "font.serif": ["Palatino"],})


def extrapolation(x_total, y_total, n_final_points):
    indexes = list(range(-1, -n_final_points-1, -1))
    x_limited = x_total[indexes]
    y_limited = y_total[indexes]
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_limited, y_limited)
    return slope, intercept


def take_2s_energies(array_with_eigenvalues):
    n_of_discretizations = array_with_eigenvalues.shape[1]
    n_of_different_sizes = array_with_eigenvalues.shape[2]
    eigvals_2s = np.zeros((n_of_discretizations, n_of_different_sizes))
    for i in range(n_of_different_sizes):
        for j in range(n_of_discretizations):
            if round(array_with_eigenvalues[1,j,i],1) == round(array_with_eigenvalues[2,j,i],1):
                eigvals_2s[j,i] = array_with_eigenvalues[3,j,i]
            else:
                eigvals_2s[j,i] = array_with_eigenvalues[1,j,i]
    return eigvals_2s


file_name_info = "../Data/info_wannier_rytova_keldysh_WSe2_many_sizes_eps_1_from_11_through_111.npz"
file_name_data = "../Data/data_wannier_rytova_keldysh_WSe2_many_sizes_eps_1_from_11_through_111.npz"

data = np.load(file_name_data)
info = np.load(file_name_info)

# for key in data.keys():
#     print(key)

eigvals_1s = data['eigvals_holder'][0,:,:]

##==================================================================================##
##                                   1S - STATE                                     ##
##==================================================================================##

## EXTRAPOLATION:
x_to_extrp = 100/info["n_points"]
y_to_extrp_1s = eigvals_1s

n_points_to_extrap = 3

alpha_1_1s, beta_1_1s = extrapolation(x_to_extrp, y_to_extrp_1s[:,0], n_points_to_extrap)
alpha_2_1s, beta_2_1s = extrapolation(x_to_extrp, y_to_extrp_1s[:,1], n_points_to_extrap)
alpha_3_1s, beta_3_1s = extrapolation(x_to_extrp, y_to_extrp_1s[:,2], n_points_to_extrap)
alpha_4_1s, beta_4_1s = extrapolation(x_to_extrp, y_to_extrp_1s[:,3], n_points_to_extrap)
alpha_5_1s, beta_5_1s = extrapolation(x_to_extrp, y_to_extrp_1s[:,4], n_points_to_extrap)

x_extrapolated = np.linspace(0,1,10)
y_extrapolated_1 = alpha_1_1s*x_extrapolated + beta_1_1s
y_extrapolated_2 = alpha_2_1s*x_extrapolated + beta_2_1s
y_extrapolated_3 = alpha_3_1s*x_extrapolated + beta_3_1s
y_extrapolated_4 = alpha_4_1s*x_extrapolated + beta_4_1s
y_extrapolated_5 = alpha_5_1s*x_extrapolated + beta_5_1s

##==================================================================================##
##                               TABLE OF RESULTS                                   ##
##==================================================================================##
list_of_extrapolated_values = [beta_1_1s, beta_2_1s, beta_3_1s, beta_4_1s, beta_5_1s]
list_of_titles = [r' 1/nm',r' 2/nm',r' 3/nm',r' 4/nm',' 5/nm']
for title in list_of_titles:
    print(title, end='\t|\t')
print('\n------------------------------------------------------------------------')
for variable in list_of_extrapolated_values:
    print(round(variable,1),end='\t|\t')

##==================================================================================##
##                                    PLOTTING                                      ##
##==================================================================================##
# Eb_1s = -87.071

fig, ax = plt.subplots(nrows=1, ncols=2,figsize=(16,8))
ax[0].plot(info["n_points"], eigvals_1s, marker='s')
# ax[0].hlines(Eb_1s,0,135,linestyle="dashed",colors='k')
ax[0].legend((r'1 nm$^{-1}$', r'2 nm$^{-1}$', r'3 nm$^{-1}$', r'4 nm$^{-1}$', r'5 nm$^{-1}$','E$_{Analyt.}$'), fontsize=22)
ax[0].tick_params(axis='x', labelsize=20)
ax[0].tick_params(axis='y', labelsize=20)
ax[0].set_xlabel('Disc. N$_k$', fontsize=20)
ax[0].set_ylabel(r'E$_B$[meV]', fontsize=20)
# ax[0].text(10,-85,r"E$_B$ = -87.071 meV", fontsize=20)
# ax[0].set_xlim((5,135))
ax[1].plot(100/info["n_points"], eigvals_1s, marker='s')
# ax[1].hlines(Eb_1s,-0.5,11,linestyle="dashed",colors='k')
ax[1].legend((r'1 nm$^{-1}$', r'2 nm$^{-1}$', r'3 nm$^{-1}$', r'4 nm$^{-1}$', r'5 nm$^{-1}$','E$_{Analyt.}$'), fontsize=22)
ax[1].plot(x_extrapolated, y_extrapolated_1, '.',color='C0')
ax[1].plot(x_extrapolated, y_extrapolated_2, '.',color='C1')
ax[1].plot(x_extrapolated, y_extrapolated_3, '.',color='C2')
ax[1].plot(x_extrapolated, y_extrapolated_4, '.',color='C3')
ax[1].plot(x_extrapolated, y_extrapolated_5, '.',color='C4')
ax[1].tick_params(axis='x', labelsize=20)
ax[1].tick_params(axis='y', labelsize=20)
ax[1].set_xlabel('Disc. 1/N$_k$ [x$10^{-2}$]', fontsize=20)
ax[1].set_ylabel(r'E$_B$[meV]', fontsize=20)
# ax[1].text(5,-85,r"E$_B$ = -87.071 meV", fontsize=20)
# ax[1].vlines(0,-90,-20,linestyle="dashed",colors='k')
# ax[1].set_xlim((-0.1,10))
plt.tight_layout()
plt.show()

##==================================================================================##
##                               2S - STATES                                        ##
##==================================================================================##
eigvals_2s = take_2s_energies(data['eigvals_holder'])


## EXTRAPOLATION:
x_to_extrp = 100/info["n_points"]
y_to_extrp = eigvals_2s

n_points_to_extrap = 3

alpha_1_2s, beta_1_2s = extrapolation(x_to_extrp, y_to_extrp[:,0], n_points_to_extrap)
alpha_2_2s, beta_2_2s = extrapolation(x_to_extrp, y_to_extrp[:,1], n_points_to_extrap)
alpha_3_2s, beta_3_2s = extrapolation(x_to_extrp, y_to_extrp[:,2], n_points_to_extrap)
alpha_4_2s, beta_4_2s = extrapolation(x_to_extrp, y_to_extrp[:,3], n_points_to_extrap)
alpha_5_2s, beta_5_2s = extrapolation(x_to_extrp, y_to_extrp[:,4], n_points_to_extrap)

x_extrapolated = np.linspace(0,1,10)
y_extrapolated_1_2s = alpha_1_2s*x_extrapolated + beta_1_2s
y_extrapolated_2_2s = alpha_2_2s*x_extrapolated + beta_2_2s
y_extrapolated_3_2s = alpha_3_2s*x_extrapolated + beta_3_2s
y_extrapolated_4_2s = alpha_4_2s*x_extrapolated + beta_4_2s
y_extrapolated_5_2s = alpha_5_2s*x_extrapolated + beta_5_2s

##==================================================================================##
##                               TABLE OF RESULTS                                   ##
##==================================================================================##
list_of_extrapolated_values = [beta_1_2s, beta_2_2s, beta_3_2s, beta_4_2s, beta_5_2s]
list_of_titles = [r' 1/nm',r' 2/nm',r' 3/nm',r' 4/nm',' 5/nm']
print('\n')
for title in list_of_titles:
    print(title, end='\t|\t')
print('\n------------------------------------------------------------------------')
for variable in list_of_extrapolated_values:
    print(round(variable,1),end='\t|\t')
print('\n')

##==================================================================================##
##                                    PLOTTING                                      ##
##==================================================================================##
# Eb_2s = -9.675
fig, ax = plt.subplots(nrows=1, ncols=2,figsize=(16,8))
ax[0].plot(info["n_points"][1:], eigvals_2s[1:],marker='s')
# ax[0].hlines(Eb_2s,10,135,linestyle="dashed",colors='k')
ax[0].legend((r'1 nm$^{-1}$', r'2 nm$^{-1}$', r'3 nm$^{-1}$', r'4 nm$^{-1}$', r'5 nm$^{-1}$','E$_{Analyt.}$'), fontsize=22)
ax[0].tick_params(axis='x', labelsize=20)
ax[0].tick_params(axis='y', labelsize=20)
ax[0].set_xlabel('Disc. N$_k$', fontsize=20)
ax[0].set_ylabel(r'E$_B$[meV]', fontsize=20)
# ax[0].text(20,-9,r"E$_B$ = {:.3} meV".format(Eb_2s), fontsize=20)
# ax[0].set_xlim((15,135))
# ax[0].set_ylim((-12,15))
##
ax[1].plot(100/info["n_points"][1:], eigvals_2s[1:],marker='s')
# ax[1].hlines(Eb_2s,-0.5,6,linestyle="dashed",colors='k')
ax[1].legend((r'1 nm$^{-1}$', r'2 nm$^{-1}$', r'3 nm$^{-1}$', r'4 nm$^{-1}$', r'5 nm$^{-1}$','E$_{Analyt.}$'), fontsize=22)
ax[1].plot(x_extrapolated, y_extrapolated_1_2s, '.',color='C0')
ax[1].plot(x_extrapolated, y_extrapolated_2_2s, '.',color='C1')
ax[1].plot(x_extrapolated, y_extrapolated_3_2s, '.',color='C2')
ax[1].plot(x_extrapolated, y_extrapolated_4_2s, '.',color='C3')
ax[1].plot(x_extrapolated, y_extrapolated_5_2s, '.',color='C4')
ax[1].tick_params(axis='x', labelsize=20)
ax[1].tick_params(axis='y', labelsize=20)
ax[1].set_xlabel('Disc. 1/N$_k$ [x$10^{-2}$]', fontsize=20)
ax[1].set_ylabel(r'E$_B$[meV]', fontsize=20)
# ax[1].text(3,-9,r"E$_B$ = {:.3} meV".format(Eb_2s), fontsize=20)
# ax[1].vlines(0,-11,15,linestyle="dashed",colors='k')
# ax[1].set_xlim((-0.1,5))
# ax[1].set_ylim((-12,17))
plt.tight_layout()
plt.show()


################################################################################

################################################################################

ind = 1
N_points = 111
nk_wfunc = int(info['n_points'][-1])
wannier_coulomb_1s = data['eigvecs_holder'][:,0,ind].reshape(N_points, N_points)
wannier_coulomb_2s = data['eigvecs_holder'][:,3,ind].reshape(N_points, N_points)

# wannier_coulomb_1s_1 = wannier_1s[:,ind].reshape(nk_wfunc, nk_wfunc)
# wannier_coulomb_2s_1 = wannier_2s[:,ind].reshape(nk_wfunc, nk_wfunc)
#
L = info['L_values'][ind]
kx = np.linspace(-L,L,nk_wfunc)
ky = np.linspace(-L,L,nk_wfunc)
#
Kx, Ky = np.meshgrid(kx,ky)
#
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16,8))
ax[0].pcolormesh(Kx,Ky,abs(wannier_coulomb_1s), shading='auto', cmap=cm.inferno)
ax[0].tick_params(axis='x', labelsize=20)
ax[0].tick_params(axis='y', labelsize=20)
ax[0].set_xlabel(r"k$_x$[nm$^{-1}$]",fontsize=22)
ax[0].set_ylabel(r"k$_y$[nm$^{-1}$]",fontsize=22)
ax[0].set_title("1s-state Rytova-Keldysh",fontsize=24)
ax[1].pcolormesh(Kx,Ky,abs(wannier_coulomb_2s), shading='auto', cmap=cm.inferno)
ax[1].tick_params(axis='x', labelsize=20)
ax[1].tick_params(axis='y', labelsize=20)
ax[1].set_xlabel(r"k$_x$[nm$^{-1}$]",fontsize=22)
ax[1].set_ylabel(r"k$_y$[nm$^{-1}$]",fontsize=22)
ax[1].set_title("2s-state Rytova-Keldysh",fontsize=24)
plt.tight_layout()
plt.show()
