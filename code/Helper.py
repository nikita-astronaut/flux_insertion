import numpy as np
from scipy.linalg import expm, inv, det

class Helper():

    def __init__(self,
                 gamma,
                 omega,
                 n_sites):
        self.n_sites = n_sites
        self.gamma = gamma
        self.omega = omega
        self.identity = np.eye(2 * self.n_sites)

    def get_index(self, i, a):
        return i % self.n_sites + a * self.n_sites

    def get_element(self, matrix, i, j, a, b):
        return matrix[self.get_index(i, a), self.get_index(j, b)]

    def get_a(self, i, j):
        a = 2 * (self.omega[j, :] - self.omega[i, :])
        return np.diag(a), np.diag(np.exp(1j * a))

    def get_inv_matrix(self, a_exp):
        return np.array(inv(self.identity - (self.identity - a_exp) @ self.gamma))

    def get_phi(self, i, j):
        a, a_exp = self.get_a(i, j)
        phi = a_exp @ self.gamma @ self.get_inv_matrix(a_exp)
        return phi

    def get_determinant(self, i, j):
        a, a_exp = self.get_a(i, j)
        matrix = self.identity - self.gamma @ (self.identity - a_exp)
        return det(matrix), matrix