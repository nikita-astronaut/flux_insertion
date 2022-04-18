from Helper import Helper
from abc import ABC, abstractmethod

class AbstractExpectationValue(ABC):

    def __init__(self,
                 gamma,
                 omega,
                 coulomb,
                 n_sites,
                 hopping_matrix):
        self.gamma = gamma
        self.omega = omega
        self.coulomb = coulomb
        self.n_sites = n_sites
        self.hopping_matrix = hopping_matrix
        self.helper = Helper(gamma, omega, n_sites)

    @abstractmethod
    def get_energy(self):
        pass


class PotentialExpectationValue(AbstractExpectationValue):

    def get_energy(self):
        energy = 0
        for i in range(self.n_sites):
            i0 = self.helper.get_index(i, 0)
            i1 = self.helper.get_index(i, 1)
            energy += self.gamma[i0][i0] * self.gamma[i1][i1] - self.gamma[i0][i1] * self.gamma[i1][i0]
        energy *= self.coulomb
        return energy.real


class KineticExpectationValue(AbstractExpectationValue):

    def get_energy(self):
        energy = 0
        check = 0
        for i in range(2 * self.n_sites):
            for j in range(2 * self.n_sites):
                a, a_exp = self.helper.get_a(i, j)
                d, m_d = self.helper.get_determinant(i, j)
                phi = self.helper.get_phi(i, j)[i][j]
                energy += self.hopping_matrix[i][j] * a_exp[i][i].conjugate() * d * phi
        return energy.real