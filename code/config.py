import numpy as np

class opt_parameters:
    def __init__(self):
        self.N_sites = 5
        self.flux = 0.0

        ### initialize random starting G ###
        self.G = np.random.uniform(-1, 1, size = (2 * self.N_sites, 2 * self.N_sites)) + \
          1.0j * np.random.uniform(-1, 1, size = (2 * self.N_sites, 2 * self.N_sites))
        self.G = (self.G + self.G.conj().T)

        ### initialize random starting Omega ###
        self.O = np.random.uniform(-1, 1, size=(2 * self.N_sites, 2 * self.N_sites)) / 10.
        self.O = self.O + self.O.T

        self.O[np.arange(2 * self.N_sites), np.arange(2 * self.N_sites)] = 0.


        ### nearest-neighbor hopping with PBC ###
        self.H = np.zeros((2 * self.N_sites, 2 * self.N_sites), dtype=np.complex128)
        self.H[np.arange(self.N_sites), (np.arange(self.N_sites) + 1) % self.N_sites] = 1.0
        self.H[np.arange(self.N_sites), (np.arange(self.N_sites) + 2) % self.N_sites] = np.exp(2.0j * np.pi * self.flux)

        self.H[np.arange(self.N_sites) + self.N_sites, self.N_sites + ((np.arange(self.N_sites) + 1) % self.N_sites)] = 1.0
        self.H[np.arange(self.N_sites) + self.N_sites, self.N_sites + ((np.arange(self.N_sites) + 2) % self.N_sites)] = np.exp(2.0j * np.pi * self.flux)

        self.H = self.H + self.H.T.conj()


        free_energies = np.linalg.eigh(self.H)[0]
        print(free_energies)
        print(np.sum(free_energies[free_energies < 0.]), "FREE ENERGY")

        self.n_opt = 100000
        self.lr = 0.03
        self.U = 3.
        self.density = self.N_sites


        self.trans = np.concatenate([(np.arange(self.N_sites) + 1) % self.N_sites, self.N_sites + ((np.arange(self.N_sites) + 1) % self.N_sites)], axis=0)

        return
