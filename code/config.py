import numpy as np

class opt_parameters:
    def __init__(self):
        self.N_sites = 20


        ### initialize random starting G ###
        self.G = np.random.uniform(-1, 1, size=(2 * self.N_sites, 2 * self.N_sites)) + \
          1.0j * np.random.uniform(-1, 1, size=(2 * self.N_sites, 2 * self.N_sites))
        self.G = self.G + self.G.conj().T

        ### initialize random starting Omega ###
        self.O = np.random.uniform(-1, 1, size=(2 * self.N_sites, 2 * self.N_sites)) / 10.
        self.O = self.O + self.O.T

        self.O[np.arange(2 * self.N_sites), np.arange(2 * self.N_sites)] = 0.



        ### nearest-neighbor hopping with PBC ###
        self.H = np.zeros((2 * self.N_sites, 2 * self.N_sites))
        self.H[np.arange(self.N_sites), (np.arange(self.N_sites) + 1) % self.N_sites] = 1.0

        self.H[np.arange(self.N_sites) + self.N_sites, self.N_sites + ((np.arange(self.N_sites) + 1) % self.N_sites)] = 1.0

        self.H = self.H + self.H.T



        self.n_opt = 1000
        self.lr = 1e-2
        self.U = 1.

        return