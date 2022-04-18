import numpy as np
from Helper import Helper
from scipy.linalg import expm, inv, det

class Derivatives():

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
        self.helper = Helper(gamma, omega, n_sites)
        self.delta = 10 ** (-6)
        self.hopping_matrix = hopping_matrix

    def get_gamma_num_derivs(self, energy_class):
        num_derivs = np.zeros((2 * self.n_sites, 2 * self.n_sites))
        energy = energy_class(self.gamma, self.omega, self.coulomb, self.n_sites,
                              self.hopping_matrix).get_energy()
        for i in range(2 * self.n_sites):
            for j in range(2 * self.n_sites):
                gamma_plus = self.gamma.copy()
                gamma_plus[i][j] = self.gamma[i][j] + self.delta
                energy_plus = energy_class(gamma_plus, self.omega, self.coulomb, self.n_sites,
                                           self.hopping_matrix).get_energy()
                num_derivs[i][j] = (energy_plus - energy) / self.delta
        return num_derivs

    def get_gamma_potential_derivative(self):
        h_U_matrix = np.zeros((2 * self.n_sites, 2 * self.n_sites))
        for i in range(self.n_sites):
            i0 = self.helper.get_index(i, 0)
            i1 = self.helper.get_index(i, 1)
            h_U_matrix[i0, i0] = self.coulomb * self.gamma[i1, i1]
            h_U_matrix[i1, i1] = self.coulomb * self.gamma[i0, i0]
            h_U_matrix[i0, i1] = - self.coulomb * self.gamma[i1, i0]
            h_U_matrix[i1, i0] = - self.coulomb * self.gamma[i0, i1]
        return h_U_matrix

    def get_h1(self):
        deriv_matrix = np.zeros((2 * self.n_sites, 2 * self.n_sites))
        for i in range(2 * self.n_sites):
            for j in range(2 * self.n_sites):
                a, a_exp = self.helper.get_a(i, j)
                phi = self.helper.get_phi(i, j)[i][j]
                det, mdet = self.helper.get_determinant(i, j)
                inv_mdet = inv(mdet)
                exp_m = self.helper.identity - a_exp
                deriv_matrix += np.real(
                    -self.hopping_matrix[i][j] * a_exp[i][i].conjugate() * phi * det * (exp_m @ inv_mdet))
        return deriv_matrix

    def get_h2(self):
        deriv_matrix = np.zeros((2 * self.n_sites, 2 * self.n_sites))
        for a in range(2 * self.n_sites):
            for b in range(2 * self.n_sites):
                deriv_matrix[b, a] = self.get_h2_el(a, b)
        return deriv_matrix

    def get_h2_el(self, a, b):
        deriv = 0
        for i in range(2 * self.n_sites):
            for j in range(2 * self.n_sites):
                a_m, a_exp = self.helper.get_a(i, j)
                det, mdet = self.helper.get_determinant(i, j)
                inv_a_exp = self.helper.get_inv_matrix(a_exp)
                exp_m = self.helper.identity - a_exp
                t1 = a_exp[i, a] * inv_a_exp[b, j] + (a_exp @ self.gamma @ inv_a_exp @ exp_m)[i, a] * inv_a_exp[b, j]
                deriv += self.hopping_matrix[i][j] * a_exp[i][i].conjugate() * det * t1
        return deriv.real
