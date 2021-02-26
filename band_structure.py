import numpy as np
import matplotlib.pyplot as plt
import treat_files as files

def split_states(Values, Vectors):
    # NEW HOLDERS THAT WILL BE ORDERED
    if len(Values.shape) == 1:
        Values  = Values.reshape((1,)+Values.shape)   # placeholder following the basis order
        Vectors = Vectors.reshape((1,)+Vectors.shape) # placeholder following the basis order
    Values_New  = np.empty(Values.shape)  # placeholder following the basis order
    Vectors_New = np.empty(Vectors.shape,dtype=complex) # placeholder following the basis order
    print(Values.shape)
    print(Vectors.shape)
    Nk, Nstates = Values_New.shape
    for i in range(Nk):
        for j in range(Nstates):
            vec = np.abs(Vectors[i,:,j])**2
            ind_new = np.where(vec==vec.max())[0][0]
            Values_New[i,ind_new]    = Values[i,j]
            Vectors_New[i,:,ind_new] = Vectors[i,:,j]
    return Values_New, Vectors_New

def plot_kormanyos_fabian_bands(kx, Values):
    fig, ax =  plt.subplots(nrows=2, ncols=2,figsize=(16,16))
    ax[0,0].plot(kx, Values[:,2], '-',linewidth=2, color='C2',label=r'$|c\uparrow\rangle$')
    ax[0,0].plot(kx, Values[:,3], '-',linewidth=2, color='C3',label=r'$|c\downarrow\rangle$')
    ax[0,0].hlines([2.8e3],xmin=-5, xmax=5, linestyle='--')
    ax[0,0].set_xlim([-2.0,2.0])
    ax[0,0].set_ylim([2750,3000])
    ax[0,0].legend(fontsize=22,loc=0)
    ax[0,0].grid()
    ax[1,0].plot(kx, Values[:,0], '-', linewidth=2, color='C2',label=r'$|v\uparrow\rangle$')
    ax[1,0].plot(kx, Values[:,1], '-', linewidth=2, color='C3',label=r'$|v\downarrow\rangle$')
    ax[1,0].set_ylim([-750,50])
    ax[1,0].hlines([0],xmin=-5, xmax=5, linestyle='--')
    ax[1,0].set_xlim([-4.5,4.5])
    ax[1,0].set_ylim([-750, 50])
    ax[1,0].legend(fontsize=22,loc=0)
    ax[1,0].grid()
    ax[0,1].plot(kx, Values[:,-2], '-',linewidth=2, color='C2')
    ax[0,1].plot(kx, Values[:,-1], '-',linewidth=2, color='C3')
    ax[0,1].hlines([2.8e3],xmin=-5, xmax=5, linestyle='--')
    ax[0,1].set_xlim([-2.0,2.0])
    ax[0,1].set_ylim([2750,3000])
    ax[0,1].grid()
    ax[1,1].plot(kx, Values[:,0], '-', linewidth=2, color='C2')
    ax[1,1].plot(kx, Values[:,1], '-', linewidth=2, color='C3')
    ax[1,1].set_ylim([-750,50])
    ax[1,1].hlines([0],xmin=-5, xmax=5, linestyle='--')
    ax[1,1].set_xlim([-4.5,4.5])
    ax[1,1].set_ylim([-750, 50])
    ax[1,1].grid()
    plt.show()
    return 0


def main():
    # READ THE "infile.txt"
    params = files.read_params("infile.txt")
    Ham, r_0, epsilon, exchange, d_chosen, Lk, n_mesh, n_sub, submesh_radius, n_rec_states = files.pop_out_model(params)
    hamiltonian      = Ham(**params)

    # K-SPACE DEFINITION:
    Lk = 5   # 1/nm
    Nk = 1001
    kx = np.linspace(-Lk, Lk, Nk) # 1/AA
    Nstates = hamiltonian.condBands + hamiltonian.valeBands
    Values  = np.empty((Nk, Nstates), dtype=float)
    Vectors = np.empty((Nk, Nstates, Nstates), dtype=complex)

    # CALCULATION OF THE EIGENVALUES AND THE EIGENVECTORS
    for i in range(len(kx)):
        Values[i,:], Vectors[i,:,:] = LA.eigh(hamiltonian.call(kx[i],0))

    # REORGANIZE THE VECTORS AND THE VALUES
    Values_New, Vectors_New = split_states(Values, Vectors)

    # PLOT THE BANDS
    plot_kormanyos_fabian_bands(kx, Values_New)



if __name__=='__main__':
    main()
