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

#===============================================================================
fieldsWannier=[
    ('Egap',     float32),
    ('m_eff',    float32),
    ('condBands', int32),
    ('valeBands', int32),
]

class H_Wannier:

    def __init__(self,m_1,m_2=None,Egap=0):
        """
        """
        if m_2 != None: self.m_eff = (m_1*m_2)/(m_1+m_2)
        else: self.m_eff = m_1
        self.condBands = 0
        self.valeBands = 0
        self.Egap      = Egap

    def call(self,kx,ky):
        m_eff = self.m_eff
        return (const.hbar2_over2m/m_eff)*(kx**2 + ky**2)

#===============================================================================
fields2x2 = [
    ('alphac', float32),
    ('alphav', float32),
    ('Egap', float32),
    ('gamma', float32),
    ('valeBands', int32),
    ('condBands', int32),
]

@jitclass(fields2x2)
class H2x2:
    def __init__(self, alphac, alphav, Egap, gamma):
        self.alphac = alphac
        self.alphav = alphav
        self.Egap = Egap
        self.gamma = gamma
        self.valeBands = 1
        self.condBands = 1

    def Pi(self):
        gamma = self.gamma
        Pix = gamma * np.array([
            [0,1],
            [1,0]])
        Piy = gamma * np.array([
            [  0, 1j],
            [-1j,  0]])
        return Pix, Piy

    def H_0(self):
        E_gap   = self.Egap
        H = np.array([
            [E_gap, 0],
            [    0, 0*E_gap]])
        return H

    def H_kp1(self, kx, ky):
        # gamma = self.gamma
        Pix, Piy = self.Pi()
        return kx*Pix + ky*Piy

    def H_k2(self, kx, ky):
        a_c = self.alphac
        a_v = self.alphav
        k2 = kx**2 + ky**2
        H = const.hbar2_over2m * k2 * np.array([
            [ a_c,   0],
            [   0, a_v]])
        return H

    def call(self, kx, ky):
        return self.H_kp1(kx,ky) + self.H_k2(kx,ky) + self.H_0()

#===============================================================================

class H4x4_general:
    """
    The instances of this class cannot be passed to numba-compiled functions.
    But, as a Python class it can be parent of other classes, which includes
    'jitclasses'.
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

    def Pi(self):
        g_up = self.gamma_up
        g_dn = self.gamma_down
        Pix = (1. + 0.j) * np.array([
            [   0,g_up,   0,   0],
            [g_up,   0,   0,   0],
            [   0,   0,   0,g_dn],
            [   0,   0,g_dn,   0]])
        Piy = 1j * np.array([
            [    0, g_up,     0,    0],
            [-g_up,    0,     0,    0],
            [    0,    0,     0, g_dn],
            [    0,    0, -g_dn,    0]])
        return Pix, Piy

    def H_0(self):
        Eg_up = self.gap_up
        Eg_dn = self.gap_down
        H = np.array([
        [Eg_up,   0,    0,   0],
        [    0,   0,    0,   0],
        [    0,   0,Eg_dn,   0],
        [    0,   0,    0,   0]])
        return H

    def H_kp1(self, kx, ky):
        Pix, Piy = self.Pi()
        return kx*Pix + ky*Piy

    def H_k2(self,kx,ky):
        k2 = kx**2 + ky**2
        a_c_up, a_c_down = self.alphac_up, self.alphac_down
        a_v_up, a_v_down = self.alphav_up, self.alphav_down
        H = const.hbar2_over2m * k2 * np.array([
        [a_c_up, 0, 0, 0],
        [0, a_v_up, 0, 0],
        [0, 0, a_c_down, 0],
        [0, 0, 0, a_v_down]])
        return H

    def call(self, kx, ky):
        """
        To be used as a parent class for "jit-classes" we cannot define
        the parent class with a '__call__' method. Instead, we define a
        simple method named 'call' that needs to be called explicitly.
        """
        # H = np.array([
        # [Eg_up + const.hbar2_over2m * alpha_c_up * k2, gamma_up*(kx+1j*ky), 0 , 0],
        # [gamma_up*(kx-1j*ky), const.hbar2_over2m * alpha_v_up * k2, 0, 0],
        # [0, 0, Eg_down + const.hbar2_over2m * alpha_c_down * k2, gamma_down*(kx+1j*ky)],
        # [0, 0, gamma_down*(kx-1j*ky), const.hbar2_over2m * alpha_v_down * k2]])
        return self.H_0() + self.H_kp1(kx,ky) + self.H_k2(kx,ky)

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
    """
    Concrete class: instantiable version of 'H4x4_general'.
    The difference between this class and the 'H4x4' is that
    here the spin-up and spin-down blocks are identical.
    """
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
fieldsKormanyosFabian = [
    ('Egap', float32),
    ('E_c', float32),
    ('E_v', float32),
    ('alpha_up', float32),
    ('alpha_dn', float32),
    ('beta_up', float32),
    ('beta_dn', float32),
    ('gamma', float32),
    ('delta_c', float32),
    ('delta_v', float32),
    ('kappa_up', float32),
    ('kappa_dn', float32),
    ('valey', int32),
    ('condBands', int32),
    ('valeBands', int32),
]

@jitclass(fieldsKormanyosFabian)
class H4x4_Kormanyos_Fabian:
    """
    """
    def __init__(self, E_c, E_v, alpha_up, alpha_dn, beta_up, beta_dn, gamma, delta_c, delta_v, kappa_up, kappa_dn, valey):
        # PARAMS OF THE HAMILTONIAN
        self.Egap= E_c
        self.E_c = E_c + np.abs(delta_c)
        self.E_v = E_v - np.abs(delta_v)
        self.alpha_up = alpha_up
        self.alpha_dn = alpha_dn
        self.beta_up = beta_up
        self.beta_dn = beta_dn
        self.gamma = gamma
        self.delta_c = delta_c
        self.delta_v = delta_v
        self.kappa_up = kappa_up
        self.kappa_dn = kappa_dn
        ## META DATA OF THE HAMILTONIAN:
        self.valey = valey        # K == 1, K' == -1
        self.condBands = 2
        self.valeBands = 2

    def basis(self):
        """
        This method is only used in the wavefunction analysis. The cluster
        version of the code doesn't need to have this method.
        It returns two lists:
            * the first list contains the conduction basis vectors
            * the second list has the valence basis vectors
        """
        cond_up = np.array([0,0,1,0])
        cond_dn = np.array([0,0,0,1])
        vale_up = np.array([1,0,0,0])
        vale_dn = np.array([0,1,0,0])
        return [cond_up, cond_dn], [vale_up, vale_dn]

    def Pi(self):
        gamma = self.gamma
        valey = self.valey
        Pix = (1.+0j) * valey * gamma * np.array([
        [0,0,1,0],
        [0,0,0,1],
        [1,0,0,0],
        [0,1,0,0]])
        Piy = gamma * np.array([
        [  0,  0,-1j,  0],
        [  0,  0,  0,-1j],
        [+1j,  0,  0,  0],
        [  0,+1j,  0,  0]])
        return Pix, Piy

    def H_0(self):
        E_v, E_c = self.E_v, self.E_c
        H0 = np.array([
        [E_v, 0, 0, 0],
        [0, E_v, 0, 0],
        [0, 0, E_c, 0],
        [0, 0, 0, E_c]])
        return H0

    def H_SO(self):
        d_v, d_c = self.delta_v, self.delta_c
        HSO = np.array([
        [d_v, 0, 0, 0],
        [0,-d_v, 0, 0],
        [0, 0, d_c, 0],
        [0, 0, 0,-d_c]])
        return self.valey * HSO

    def H_kp1(self, kx, ky):
        Pix, Piy = self.Pi()
        return kx*Pix + ky*Piy

    def H_kp2(self, kx, ky):
        k2  = kx**2 + ky**2
        k_p = kx + self.valey * 1j * ky
        k_m = kx - self.valey * 1j * ky
        if self.valey == 1:
            a1, a2 = self.alpha_up, self.alpha_dn
            b1, b2 = self.beta_up, self.beta_dn
            kau, kad = self.kappa_up, self.kappa_dn
        else:
            a1, a2 = self.alpha_dn, self.alpha_up
            b1, b2 = self.beta_dn, self.beta_up
            kau, kad = self.kappa_dn, self.kappa_up
        Hkp2 = np.array([
        [a1*k2     ,         0,kau*k_p**2,         0],
        [         0,     a2*k2,         0,kad*k_p**2],
        [kau*k_m**2,         0,     b1*k2,         0],
        [         0,kad*k_m**2,         0,     b2*k2]])
        return Hkp2

    def call(self, kx,ky):
        return self.H_0() + self.H_SO() + self.H_kp1(kx, ky) + self.H_kp2(kx, ky)

fieldsKormanyosFabianRashba = [
    ('Egap', float32),
    ('E_c', float32),
    ('E_v', float32),
    ('alpha_up', float32),
    ('alpha_dn', float32),
    ('beta_up', float32),
    ('beta_dn', float32),
    ('gamma', float32),
    ('gamma_z', float32),
    ('delta_c', float32),
    ('delta_v', float32),
    ('kappa_up', float32),
    ('kappa_dn', float32),
    ('alpha_Rashba_c', float32),
    ('alpha_Rashba_v', float32),
    ('valey', int32),
    ('condBands', int32),
    ('valeBands', int32),
]

@jitclass(fieldsKormanyosFabianRashba)
class H4x4_Kormanyos_Fabian_Rashba:
    """
    Version of the Kormanyos Hamiltonian including the Rashba SOC:
    """
    def __init__(self, E_c, E_v, alpha_up, alpha_dn, beta_up, beta_dn, gamma, gamma_z, delta_c, delta_v, kappa_up, kappa_dn, alpha_Rashba_c, alpha_Rashba_v, valey):
        # PARAMS OF THE HAMILTONIAN
        self.Egap= E_c
        self.E_c = E_c + np.abs(delta_c)
        self.E_v = E_v - np.abs(delta_v)
        self.alpha_up = alpha_up
        self.alpha_dn = alpha_dn
        self.beta_up = beta_up
        self.beta_dn = beta_dn
        self.gamma = gamma
        self.gamma_z = gamma_z
        self.delta_c = delta_c
        self.delta_v = delta_v
        self.kappa_up = kappa_up
        self.kappa_dn = kappa_dn
        self.alpha_Rashba_c = alpha_Rashba_c
        self.alpha_Rashba_v = alpha_Rashba_v
        ## META DATA OF THE HAMILTONIAN:
        self.valey = valey        # K == 1, K' == -1
        self.condBands = 2
        self.valeBands = 2

    def basis(self):
        """
        This method is only used in the wavefunction analysis. The cluster
        version of the code doesn't need to have this method.
        It returns two lists:
            * the first list contains the conduction basis vectors
            * the second list has the valence basis vectors
        """
        cond_up = np.array([0,0,1,0])
        cond_dn = np.array([0,0,0,1])
        vale_up = np.array([1,0,0,0])
        vale_dn = np.array([0,1,0,0])
        return [cond_up, cond_dn], [vale_up, vale_dn]

    def Pi(self):
        """
        Differently from the majority of the models implemented here,
        for this particular case we have implemented the matrix 'Piz'.
        However, this matrix is NOT supposed to be used in the Hamiltonian
        definition, instead it is only used to calculate the absorption spectra
        when the polarization z is required.
        """
        gamma = self.gamma
        gamma_z = self.gamma_z
        valey = self.valey
        Pix = (1.+0j) * valey * gamma * np.array([
        [ 0, 0, 1, 0],
        [ 0, 0, 0, 1],
        [ 1, 0, 0, 0],
        [ 0, 1, 0, 0]])
        Piy = gamma * np.array([
        [  0,  0,-1j,  0],
        [  0,  0,  0,-1j],
        [+1j,  0,  0,  0],
        [  0,+1j,  0,  0]])
        Piz = (1.+0j) * gamma_z * np.array([
        [ 0, 0, 0, 1],
        [ 0, 0, 1, 0],
        [ 0, 1, 0, 0],
        [ 1, 0, 0, 0]])
        return Pix, Piy #, Piz

    def H_0(self):
        E_v, E_c = self.E_v, self.E_c
        H0 = np.array([
        [E_v, 0, 0, 0],
        [0, E_v, 0, 0],
        [0, 0, E_c, 0],
        [0, 0, 0, E_c]])
        return H0

    def H_SO(self):
        d_v, d_c = self.delta_v, self.delta_c
        HSO = np.array([
        [d_v, 0, 0, 0],
        [0,-d_v, 0, 0],
        [0, 0, d_c, 0],
        [0, 0, 0,-d_c]])
        return self.valey * HSO

    def H_kp1(self, kx, ky):
        # Pix, Piy,_ = self.Pi()
        Pix, Piy = self.Pi()
        return kx*Pix + ky*Piy

    def H_kp2(self, kx, ky):
        k2  = kx**2 + ky**2
        k_p = kx + self.valey * 1j * ky
        k_m = kx - self.valey * 1j * ky
        if self.valey == 1:
            a1, a2 = self.alpha_up, self.alpha_dn
            b1, b2 = self.beta_up, self.beta_dn
            kau, kad = self.kappa_up, self.kappa_dn
        else:
            a1, a2 = self.alpha_dn, self.alpha_up
            b1, b2 = self.beta_dn, self.beta_up
            kau, kad = self.kappa_dn, self.kappa_up
        Hkp2 = np.array([
            [a1*k2     ,         0,kau*k_p**2,         0],
            [         0,     a2*k2,         0,kad*k_p**2],
            [kau*k_m**2,         0,     b1*k2,         0],
            [         0,kad*k_m**2,         0,     b2*k2]
        ])
        return Hkp2

    def H_Rashba(self, kx, ky):
        a_C = self.alpha_Rashba_c
        a_V = self.alpha_Rashba_v
        k_p = kx + 1j * ky
        k_m = kx - 1j * ky
        H_R = 1j * np.array([
            [       0,-a_V*k_m,       0,       0],
            [ a_V*k_p,       0,       0,       0],
            [       0,       0,       0,-a_C*k_m],
            [       0,       0, a_C*k_p,       0]
        ])
        return H_R

    def call(self, kx,ky):
        return self.H_0() + self.H_SO() + self.H_kp1(kx, ky) + self.H_kp2(kx, ky) + self.H_Rashba(kx, ky)



#===============================================================================
fields3Bands = [
    ('Egap', float32),
    ('E0', float32),
    ('E1', float32),
    ('E2', float32),
    ('m0', float32),
    ('m1', float32),
    ('m2', float32),
    ('P10', float32),
    ('P20', float32),
    ('P21', float32),
    ('P20_sign', int32),
    ('P21_sign', int32),
    ('condBands', int32),
    ('valeBands', int32),
]

@jitclass(fields3Bands)
class H3x3:
    """
    3-Bands model adopted to investigate the influence of the conduction-band
    couplings in the excitons absorption spectra.
    """
    def __init__(self, E0, E1, E2, m0, m1, m2, P10, P20, P21, P20_sign, P21_sign):
        # PARAMS OF THE HAMILTONIAN
        self.E0 = E0
        self.E1 = E1
        self.E2 = E2
        self.m0 = m0
        self.m1 = m1
        self.m2 = m2
        self.P10 = P10
        self.P20 = P20
        self.P21 = P21
        self.P20_sign = P20_sign  # {+1,-1}
        self.P21_sign = P21_sign  # {+1,-1}
        self.Egap = min((E1-E0), (E2-E0))
        ## META DATA OF THE HAMILTONIAN:
        self.condBands = 2
        self.valeBands = 1

    def basis(self):
        """
        This method is only used in the wavefunction analysis. The cluster
        version of the code doesn't need to have this method.
        It returns two lists:
            * the first list contains the conduction basis vectors
            * the second list has the valence basis vectors
        """
        cond_0 = np.array([1,0,0])
        cond_1 = np.array([0,1,0])
        vale   = np.array([0,0,1])
        return [cond_0, cond_1], [vale]

    def Pi(self):
        """
        Notice that this is the important Pi-operator with which
        one can calculate the absorption. This method is also adopted
        inside this class by being called by self.H_k1(), which by its
        turn is called by self.call().

        The operators Pix and Piy have to be Hemitian, and
        Pix * k_x + Piy * k_y = [[           0, P21*k_(+/-), P20*k_(+/-)],
                                 [ P21*k_(-/+),           0,   P10*k_(+)],
                                 [ P20*k_(-/+),   P10*k_(-),           0]]
        """
        P10 = self.P10
        P20 = self.P20
        P21 = self.P21

        s0 = self.P20_sign
        s1 = self.P21_sign

        ## k_x COEFICIENTS:
        ## ALL POSITIVE, AND REAL
        ## DO NOT DEPEND ON 'CHIRALITY'
        ## PARAMETERS.
        Pix = (1.+ 0j) * np.array([
        [   0, P21, P20],
        [ P21,   0, P10],
        [ P20, P10,   0]])

        ## k_y COEFICIENTS: ALL IMAGINARY
        ## HERE THE SIGN WILL DEPEND ON
        ## ATTRIBUTES THAT CHANGE 'CHIRALITY':
        ## 'self.P20_sign', and 'self.P21_sign'.
        Piy = 1j * np.array([
        [      0, s1*P21,  s0*P20],
        [-s1*P21,      0,     P10],
        [-s0*P20,   -P10,       0]])

        return Pix, Piy

    def H_0(self):
        E0, E1, E2 = self.E0, self.E1, self.E2
        H0 = np.array([
        [E2, 0, 0],
        [0, E1, 0],
        [0, 0, E0]])
        return H0

    def H_k1(self, kx, ky):
        Pix, Piy = self.Pi()
        return kx*Pix + ky*Piy

    def H_k2(self, kx, ky):
        k2  = kx**2 + ky**2
        P10, P20, P21 = self.P10, self.P20, self.P21
        E0, E1, E2 = self.E0, self.E1, self.E2
        m0, m1, m2 = self.m0, self.m1, self.m2
        F0 =  1/m0 - (1/const.hbar2_over2m)*(P20**2/(E0-E2) + P10**2/(E0-E1))
        F1 =  1/m1 - (1/const.hbar2_over2m)*(P21**2/(E1-E2) + P10**2/(E1-E0))
        F2 =  1/m2 - (1/const.hbar2_over2m)*(P21**2/(E2-E1) + P20**2/(E2-E0))
        Hk2 = const.hbar2_over2m * k2 * np.array([
        [ F2,  0,  0],
        [  0, F1,  0],
        [  0,  0, F0]])
        return Hk2

    def call(self, kx, ky):
        return self.H_0() + self.H_k2(kx, ky) + self.H_k1(kx, ky)

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
    """
    Remember that the arguments Kx and Ky are outputs
    of "np.meshgrid". Here we have adopted the default
    order where

    Kx[i,0], Kx[i,-1] == min_kx, max_kx
    Ky[0,j], Ky[-1,j] == min_ky, max_ky

    In this way, the nested for-loop below will sweep
    all the kx-values for a fixed ky-value, i.e. the inner-loop
    while the ky-value will change following the outer-loop.
    """
    ny, nx = Kx.shape # WE'RE ASSUMING A SQUARE GRID EQUALLY SPACED
    for i in range(ny):
        for j in range(nx):
            W[i,j,:], V[i,j,:,:] = LA.eigh(H.call(Kx[i,j], Ky[i,j]))
    return W,V


def kinetic_wannier(Ham, kx_matrix, ky_matrix):
    kx_flat   = kx_matrix.flatten()
    ky_flat   = ky_matrix.flatten()
    n_points  = len(kx_flat)
    K_Wannier = np.empty(n_points)
    for ind in range(n_points):
        K_Wannier[ind] = Ham.call(kx_flat[ind], ky_flat[ind])
    return np.diagflat(K_Wannier)


def main():
    # k = np.linspace(-1,1,10)
    # dk2 = (k[1] - k[0])**2
    # Kx, Ky = np.meshgrid(k,k)
    #
    # Ham_params = dict(
    # E_c         = 2.42e3,                    # float : H_0   [meV]
    # E_v         = 0.0,                       # float : H_0   [meV]
    # alpha_up    = -2.16e1,                   # float : H_2kp [meV nm²]
    # alpha_dn    = -3.77e1,                  # float : H_2kp [meV nm²]
    # beta_up     = 6.81e1,                    # float : H_2kp [meV nm²]
    # beta_dn     = 4.89e1,                    # float : H_2kp [meV nm²]
    # gamma       = 2.60e2,                    # float : H_1kp [meV nm]
    # delta_c     = 0.0185e3,                  # float : H_SO  [meV]
    # delta_v     = 0.2310e3,                  # float : H_SO  [meV]
    # kappa_up    = -1.36e1,                   # float : H_2kp [meV nm²]
    # kappa_dn    = -1.14e1,                   # float : H_2kp [meV nm²]
    # valey       = 1,                         # int   : 1 == K-point; -1 == K'-point
    # )
    # HKF = H4x4_Kormanyos_Fabian(**Ham_params)
    # # H4 = H4x4(alphac, alphav, E_gap, gamma, alphac, alphav, E_gap, gamma)
    # H4 = H4x4_equal(alphac, alphav, E_gap, gamma)
    # print(H4.Pi().shape)
    params_test = dict(
        E0 = 0,
        E1 = 2.4,
        E2 = 2.5,
        m0 =-0.6,
        m1 = 0.4,
        m2 = 0.6,
        P10 = 2.0,
        P20 = 0.5,
        P21 = 0.5,
        P20_sign = +1,
        P21_sign = +1,
    )
    H = H3x3(**params_test)
    print(H.call(0,0))

    print(H.basis())


if __name__=='__main__':
    main()
