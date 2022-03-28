import numpy as np

class VarState:
    def __init__(self, opt_config):
        self.nsites = opt_config.N_sites
        self.H = opt_config.H
        self.G = opt_config.G.copy()
        self.O = opt_config.O.copy()
        self.U = opt_config.U

    def gradient(self):
        raise NotImplementedError

    def energy(self):
        S = self.nsites  # just for brevity

        alpha = -np.subtract.outer(self.O, self.O)
        alpha = np.diagonal(alpha.transpose((2, 0, 1, 3)), axis1=-1, axis2=-2)

        Aexp = np.zeros((2 * S, 2 * S, 2 * S, 2 * S), dtype=np.complex128)
        Aexp[..., np.arange(2 * S), np.arange(2 * S)] = np.exp(1.0j * alpha)  # abpq


        d4_id = np.tile(np.eye(2 * S)[np.newaxis, np.newaxis, ...], (2 * S, 2 * S, 1, 1))
        d4_Gamma = np.tile(self.G[np.newaxis, np.newaxis, ...], (2 * S, 2 * S, 1, 1))


        dets = np.linalg.det(d4_id - d4_Gamma @ (d4_id - Aexp))  # abpq -> ab


        Phis = Aexp @ d4_Gamma @ np.linalg.inv(d4_id - (d4_id - Aexp) @ d4_Gamma)
        Phis = np.diagonal(Phis.transpose((0, 2, 1, 3)), axis1=0, axis2=1)
        Phis = np.diagonal(Phis, axis1=0, axis2=1).T  # abpq -> ab


        exp_small = np.diagonal(Aexp.conj(), axis1=-1, axis2=-2)
        exp_small = np.diagonal(exp_small.transpose((2, 0, 1)), axis1=0, axis2=1).T


        kin_energy = np.sum(self.H * exp_small * dets * Phis)


        pot_energy = np.sum(self.G[np.arange(S), np.arange(S)] * self.G[np.arange(S) + S, np.arange(S) + S] - \
                            self.G[np.arange(S), np.arange(S) + S] * self.G[np.arange(S) + S, np.arange(S)])

        return kin_energy + self.U * pot_energy