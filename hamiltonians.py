
import numpy as np
import numpy.linalg as LA
import physical_constants as const
from numba import jit, njit, int32, float32
from numba.experimental import jitclass


#===============================================================================
# NOTE THAT WE CANNOT INHERITATE FROM A "jitclass".
# SO, AN WORKAROUND IS TO DEFINE A PYTHON CLASS THAT BEHAVES LIKE A
# ABSTRACT CLASS. SUCH CLASS IS NOT ALLOWED TO INSTANTIATE OBJECTS
# BUT IT CAN SERVE AS PARENT OF 'jitclasses'.
#===============================================================================

class Hamiltonian:
    """
    WORK IN PROGRESS: A ABSTRACT PARENT CLASS WITH ALL FUNCTIONATIES
    RELATED TO THE MODEL HAMILTONIAN ADOPTED.

    IN THE CURRENT VERSION, THE HAMILTONIANS DO NOT HAVE A COMMON
    PARENT AND THEY HAVE ONLY THE FOLLOWING METHODS:

     * __init__
     * call()

    """

    def __init__(self):
        """Need to be overwritten by children's definitions"""
        self.condBands = 4
        self.valeBands = 4
        pass

    def call(self):
        """Need to be overwritten by children's definitions"""
        pass

    def split(self, vectors):
        ## Revert the order (of valence bands: index [0] -> closer to the gap)
        valence_vectors = vectors[:,:self.valeBands]
        conduction_vectors = vectors[:,::-1]
        ## All the remaining vectors are conduction states
        conduct_vectors = vectors[:,self.valeBands:]
        return valence_vectors, conduct_vectors

    def values_and_vectors(self, kx_matrix, ky_matrix):
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
        n, m = kx_matrix.shape # WE'RE ASSUMING A SQUARE GRID EQUALLY SPACED
        l = self.condBands + self.valeBands
        W = np.zeros((n,m,l))
        V = 1j * np.zeros((n,m,l,l))
        W, V = eig_vals_vects(hamiltonian, W, V, kx_matrix, ky_matrix)
        return W, V

    def delta(self, c1, v1, c2, v2, k1=(0,0), k2=(0,0)):
        _, vectors_1 = LA.eigh(self.call(*k1))
        _, vectors_2 = LA.eigh(self.call(*k2))
        vectors_1_v, vectors_1_c = self.split(vectors_1)
        vectors_2_v, vectors_2_c = self.split(vectors_2)
        cond_1 = vectors_1_c[:,c1].conjugate()
        cond_2 = vectors_2_c[:,c2]
        vale_1 = vectors_1_v[:,v1]
        vale_2 = vectors_2_v[:,v2].conjugate()
        # The operators "@" and "np.dot()" return a warning
        return np.sum(cond_1*cond_2) * np.sum(vale_2*vale_1)

class H4x4_general:
    """
    The instances of this class cannot be passed to numba-compiled functions.
    But, since it is a Python class it can be parent of other classes, it
    includes 'jitclasses'.
    """
    def __init__(self, alphac_up, alphav_up, gap_up, gamma_up, alphac_down, alphav_down, gap_down, gamma_down):
        ## SPIN-UP:
        self.alphac_up = alphac_up
        self.alphav_up = alphav_up
        self.gap_up = gap_up
        self.gamma_up = gamma_up
        ## SPIN-DOWN:
        self.alphac_down = alphac_down
        self.alphav_down = alphav_down
        self.gap_down = gap_down
        self.gamma_down = gamma_down
        ## HAMILTONIAN:
        self.valeBands = 2
        self.condBands = 2

    def call(self, kx, ky):
            """
            To be used as a parent class for "jit-classes" we cannot define
            the parent class with a '__call__' method. Instead, we define a
            simple method named 'call' that needs to be called explicitly.
            """
            Eg_up, Eg_down = self.gap_up, self.gap_down
            alpha_c_up, alpha_c_down = self.alphac_up, self.alphac_down
            alpha_v_up, alpha_v_down = self.alphav_up, self.alphav_down
            gamma_up, gamma_down = self.gamma_up, self.gamma_down
            k2 = kx**2 + ky**2
            H = np.array([
            [Eg_up + const.hbar2_over2m * alpha_c_up * k2, gamma_up*(kx+1j*ky), 0 , 0],
            [gamma_up*(kx-1j*ky), const.hbar2_over2m * alpha_v_up * k2, 0, 0],
            [0, 0, Eg_down + const.hbar2_over2m * alpha_c_down * k2, gamma_down*(kx+1j*ky)],
            [0, 0, gamma_down*(kx-1j*ky), const.hbar2_over2m * alpha_v_down * k2]])
            return H

#===============================================================================
fields2x2 = [
    ('alphac', float32),
    ('alphav', float32),
    ('gap', float32),
    ('gamma', float32),
    ('condBands', int32),
    ('valeBands', int32),
]

@jitclass(fields2x2)
class H2x2:
    def __init__(self, alphac, alphav, gap, gamma):
        self.alphac = alphac
        self.alphav = alphav
        self.gap = gap
        self.gamma = gamma
        self.valeBands = 1
        self.condBands = 1

    def call(self, kx, ky):
        E_gap = self.gap
        Alpha_c = self.alphac
        Alpha_v = self.alphav
        Gamma = self.gamma
        k2 = kx**2 + ky**2
        H = np.array([
            [E_gap + const.hbar2_over2m * Alpha_c * k2, Gamma*(kx+1j*ky)],
            [Gamma*(kx-1j*ky), const.hbar2_over2m * Alpha_v * k2]])
        return H

#===============================================================================
fields4x4 = [
    ('alphac_up', float32),
    ('alphav_up', float32),
    ('gap_up', float32),
    ('gamma_up', float32),
    ('alphac_down', float32),
    ('alphav_down', float32),
    ('gap_down', float32),
    ('gamma_down', float32),
    ('condBands', int32),
    ('valeBands', int32),
]

@jitclass(fields4x4)
class H4x4(H4x4_general):
    """
    This is the instantiable version of "H4x4_general".
    Like happens in Julia: this is the leaf of the inheritance tree.
    """
    pass

@jitclass(fields4x4)
class H4x4_equal(H4x4_general):
    def __init__(self, alphac, alphav, gap, gamma):
        ## SPIN-UP:
        self.alphac_up = alphac
        self.alphav_up = alphav
        self.gap_up = gap
        self.gamma_up = gamma
        ## SPIN-DOWN:
        self.alphac_down = alphac
        self.alphav_down = alphav
        self.gap_down = gap
        self.gamma_down = gamma
        ## HAMILTONIAN:
        self.valeBands = 2
        self.condBands = 2

#===============================================================================
fieldsRytova = [
    ('dk2', float32),
    ('r_0', float32),
    ('epsilon', float32),
]

@jitclass(fieldsRytova)
class Rytova_Keldysh:
    def __init__(self, dk2, r_0, epsilon):
        self.dk2 = dk2
        self.r_0 = r_0
        self.epsilon = epsilon

    def call(self, q):
        """
        The "pontual" version of the function in Wannier script.
        Instead of return the whole matrix this function returns
        only the value asked.
        """
        dk2, epsilon, r_0 = self.dk2, self.epsilon, self.r_0
        Vkk_const = 1e6/(2*const.EPSILON_0)
        V =  1/(epsilon*q + r_0*q**2)
        return - Vkk_const * dk2/(2*np.pi)**2 * V

@njit
def potential_average(V, k_vec_diff, N_submesh, submesh_radius):
    """
    As we've been using a square lattice, we can use
    * w_x_array == w_y_array -> w_array
    * with limits:  -dw/2, +dw/2
    * where: dw = sqrt(dk2)
    """
    k_diff_norm = np.sqrt(k_vec_diff[0]**2 + k_vec_diff[1]**2)
    dk = np.sqrt(V.dk2)
    threshold = submesh_radius * dk

    # print('threshold: ', threshold)
    # print('k_diff: ', k_diff_norm)

    if N_submesh==None or k_diff_norm > threshold:
        Potential_value = V.call(k_diff_norm)
    else:
        # THIS BLOCK WILL RUN ONLY IF "k_diff_norm" IS EQUAL OR SMALLER
        # THAN A LIMIT, DENOTED HERE BY "threshold":
        w_array = np.linspace(-dk/2, dk/2, N_submesh)
        Potential_value = 0
        number_of_sing_points = 0
        for wx in w_array:
            for wy in w_array:
                w_vec = np.array([wx, wy])
                q_vec = k_vec_diff + w_vec
                q = np.linalg.norm(q_vec)
                if q == 0: number_of_sing_points += 1; continue; # skip singularities
                Potential_value += V.call(q)
        if number_of_sing_points != 0 :
            print("\t\t\tFor k-k' = ", k_vec_diff ," the number of singular points was ", number_of_sing_points)
        Potential_value = Potential_value/(N_submesh**2 - number_of_sing_points)
    return Potential_value

@njit
def smart_potential_matrix(V, kx_flat, ky_flat, N_submesh, submesh_radius):
    """
    CONSIDERING A SQUARE K-SPACE GRID:

    This function explore the regularity in the meshgrid that defines the k-space
    to build the potential-matrix [V(k-k')].

    Note that it is exclusive for the Rytova-Keldysh potential.

    # TODO: to make this function more general in the sense that any other potential
    function could be adopted: Define a Potential-class and  pass instances of such
    class instead of pass attributes of Rytova-Keldysh potential.

    """
    n_all_k_space = len(kx_flat)
    n_first_row_k = int(np.sqrt(n_all_k_space)) # number of points in the first row of the grid
    M_first_rows = np.zeros((n_first_row_k, n_all_k_space))
    M_complete = np.zeros((n_all_k_space, n_all_k_space))
    print("\t\tCalculating the first rows (it may take a while)...")
    for k1_ind in range(n_first_row_k):
        for k2_ind in range(k1_ind+1, n_all_k_space):
            k1_vec = np.array([kx_flat[k1_ind], ky_flat[k1_ind]])
            k2_vec = np.array([kx_flat[k2_ind], ky_flat[k2_ind]])
            k_diff = k1_vec - k2_vec
            M_first_rows[k1_ind, k2_ind] = potential_average(V, k_diff, N_submesh, submesh_radius)

    print("\t\tOrganizing the the calculated values...")
    M_complete[:n_first_row_k,:] = M_first_rows
    for row in range(1, n_first_row_k):
        ni, nf = row * n_first_row_k, (row+1) * n_first_row_k
        mi, mf = ni, -ni
        M_complete[ni:nf, mi:] = M_first_rows[:, :mf]

    M_complete += M_complete.T
    # plt.imshow(M_complete)
    return M_complete


def potential_matrix(V, kx_matrix, ky_matrix, N_submesh, submesh_radius=0):
    """
    This function generates a square matrix that contains the values of
    the potential for each pair of vectors k & k'.

    Dimensions = Nk x Nk
    where Nk = (Nk_x * Nk_y)
    """
    kx_flat = kx_matrix.flatten()
    ky_flat = ky_matrix.flatten()

    # OUT OF DIAGONAL: SMART SCHEME
    # N_submesh_off = N_submesh if submesh_off_diag == True else None
    V_main = smart_potential_matrix(V, kx_flat, ky_flat, N_submesh, submesh_radius)

    # DIAGONAL VALUE: EQUAL FOR EVERY POINT (WHEN USING SUBMESH)
    if N_submesh != None:
        print("\t\tCalculating the potential around zero...")
        k_0 = np.array([0,0])
        V_0 = potential_average(V, k_0, N_submesh, submesh_radius)
        np.fill_diagonal(V_main, V_0) # PUT ALL TOGETHER

    return V_main



#===============================================================================
@njit
def values_and_vectors(hamiltonian, kx_matrix, ky_matrix):
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
    n, m = kx_matrix.shape # WE'RE ASSUMING A SQUARE GRID EQUALLY SPACED
    l = hamiltonian.condBands + hamiltonian.valeBands
    W = np.zeros((n,m,l))
    V = 1j * np.zeros((n,m,l,l))
    W, V = eig_vals_vects(hamiltonian, W, V, kx_matrix, ky_matrix)
    return W, V

@njit
def eig_vals_vects(H, W, V, Kx, Ky):
    n, m = Kx.shape # WE'RE ASSUMING A SQUARE GRID EQUALLY SPACED
    for i in range(n):
        for j in range(m):
            W[i,j,:], V[i,j,:,:] = LA.eigh(H.call(Kx[i,j], Ky[i,j]))
    return W,V


def main():
    k = np.linspace(-1,1,10)
    dk2 = (k[1] - k[0])**2
    Kx, Ky = np.meshgrid(k,k)

    # eigenvalues, eigenvectors = values_and_vectors(H,Kx,Ky)
    # print(eig_vals(H))
    # print(eigenvalues)
    # alphac = 0
    # alphav = 0
    # E_gap = 2.4e3 # meV ~ 2.4 eV
    # gamma = 2.6e2 # meV*nm ~ 2.6 eV*AA
    # H4 = H4x4(alphac, alphav, E_gap, gamma, alphac, alphav, E_gap, gamma)
    # H2 = H2x2(alphac, alphav, E_gap, gamma)
    # print(H2.call(0,0))








if __name__=='__main__':
    main()
