import numpy as np

EPSILON_0 = 55.26349406             # e^2 GeV^{-1}fm^{-1} == e^2 (1e9 eV 1e-15 m)^{-1}

class Rytova:
    def __init__(self, epsilon, r0, dk2):
        self.epsilon = epsilon
        self.r0 = r0
        self.dk2 = dk2

    def __call__(self, q):
        epsilon = self.epsilon
        r0 = self.r0
        dk2 = self.dk2
        Vkk_const = 1e6/(2*EPSILON_0)
        V =  1/(epsilon*q + r0*q**2)
        return - Vkk_const * dk2/(2*np.pi)**2 * V
