import numpy as np

class VarState:
    def __init__(self, opt_config):
        self.nsites = opt_config.N_sites
        self.H = opt_config.H
        self.G = opt_config.G.copy()
        self.O = opt_config.O.copy()
        self.U = opt_config.U


        # for on-site interaction
        self.U_asmatrix = np.zeros(self.H.shape, dtype=np.complex128)
        self.U_asmatrix[np.arange(self.H.shape[0] // 2), np.arange(self.H.shape[0] // 2) + self.H.shape[0] // 2] = self.U  

        self.restore_idempotent_form()

    def gradient(self, E, h_matrix):
        domega = self.O * 0.#self.omega_natural_grad(E, h_matrix)

        dGamma = h_matrix#self.Gamma_natural_grad(domega, h_matrix)

        return dGamma, domega

    def energy_derivative(self):
        S = self.nsites  # just for brevity

        alpha = np.subtract.outer(self.O, self.O)
        alpha = np.diagonal(alpha.transpose((2, 0, 1, 3)), axis1=-1, axis2=-2)

        Aexp = np.zeros((2 * S, 2 * S, 2 * S, 2 * S), dtype=np.complex128)
        Aexp[..., np.arange(2 * S), np.arange(2 * S)] = np.exp(2.0j * alpha)  # abpq


        d4_id = np.tile(np.eye(2 * S)[np.newaxis, np.newaxis, ...], (2 * S, 2 * S, 1, 1))
        d4_Gamma = np.tile(self.G[np.newaxis, np.newaxis, ...], (2 * S, 2 * S, 1, 1))


        dets = np.linalg.det(d4_id - d4_Gamma @ (d4_id - Aexp))  # abpq -> ab

        invs = np.linalg.inv(d4_id - (d4_id - Aexp) @ d4_Gamma)
        invs_det = np.linalg.inv(d4_id - d4_Gamma @ (d4_id - Aexp))
        Phis = Aexp @ d4_Gamma @ invs
        Phis = np.diagonal(Phis.transpose((0, 2, 1, 3)), axis1=0, axis2=1)
        Phis = np.diagonal(Phis, axis1=0, axis2=1)  # abpq -> ab


        exp_small = np.diagonal(Aexp.conj(), axis1=-1, axis2=-2)
        exp_small = np.diagonal(exp_small.transpose((2, 0, 1)), axis1=0, axis2=1).T


        kin_energy = np.sum(self.H * exp_small * dets * Phis)
        pot_energy = -np.einsum('ab,ab,ba', self.U_asmatrix, self.G, self.G) + \
                      np.einsum('ab,aa,bb', self.U_asmatrix, self.G, self.G)

        Heff_kin = self.H * exp_small * dets * Phis

        #First there should be inverse of the derivative (invs_det instead invs). In second and third terms I corrected the output to
        # h_kin = np.einsum('ab,abxy->xy', Heff_kin, -(d4_id - Aexp) @ invs_det, optimize='optimal') + \
        #         np.einsum('ab,abax,abyb->yx', self.H * exp_small * dets, Aexp, invs, optimize='optimal') + \
        #         np.einsum('ab,abax,abyb->yx', self.H * exp_small * dets, Aexp @ d4_Gamma @ invs @ (d4_id - Aexp), invs, optimize='optimal')
        h_kin = np.einsum('ab,abxy->xy', Heff_kin, -(d4_id - Aexp) @ invs_det, optimize='optimal') + \
                np.einsum('ab,abax,abyb->xy', self.H * exp_small * dets, Aexp, invs, optimize='optimal') + \
                np.einsum('ab,abax,abyb->xy', self.H * exp_small * dets, Aexp @ d4_Gamma @ invs @ (d4_id - Aexp), invs, optimize='optimal')

        #Second one np.einsum('xy,yx->xy', self.G, self.U_asmatrix) change. Instead h_pot.conjugate().T != h_pot.
        h_pot = -np.einsum('xy,yx->xy', self.U_asmatrix, self.G) - np.einsum('xy,xy->yx', self.U_asmatrix, self.G, optimize='optimal') + \
                 np.einsum('xy,xb,bb->xy', np.eye(2 * S), self.U_asmatrix, self.G, optimize='optimal') + \
                 np.einsum('xy,ax,aa->xy', np.eye(2 * S), self.U_asmatrix, self.G, optimize='optimal')

        return kin_energy + pot_energy, h_kin + h_pot

    def restore_idempotent_form(self):
        # for an idempotent matrix, in the SVD decomposition V = U^* and singular values are only 0 and 1
        u, s0, v = np.linalg.svd(self.G)
        s = s0.copy()
        s[s < 0.5] = 0.0
        s[s > 0.5] = 1.0

        print('idempotentmachung: ||v - u^dag||', np.linalg.norm(v - u.conj().T))
        print('|s - s0|:', np.linalg.norm(s - s0))


        self.G = u @ np.diag(s) @ u.conj().T

        assert np.allclose(self.G, self.G @ self.G)
        return


    def omega_natural_grad(self, E, h):
        VklGS_values = VklGS(self.G)
        rhs = E * VklGS_values
        assert np.isclose(np.linalg.norm(rhs.imag), 0.0) # <GS|n_i n_j |GS> can only be real-valued for any i-j

        rhs -= VklHpotGS(self.G, self.U_asmatrix)
        assert np.isclose(np.linalg.norm(rhs.imag), 0.0) # <GS|n_i n_j H_pot|GS> -- only density density terms

        rhs -= VklHkinGS(self.G, self.O, self.H)
        print(rhs)
        #exit(-1)
        VklcjicjGS_values = VklcdicjGS(self.G)


        '''
        print('VklcjicjGS_values norm', np.sum(np.abs(VklcjicjGS_values)))

        VklcjicjGS_values_transpose = VklcjicjGS_values.conj().transpose((1, 0, 2, 3))

        for i in range(VklcjicjGS_values.shape[0]):
            for j in range(VklcjicjGS_values.shape[0]):
                for k in range(VklcjicjGS_values.shape[0]):
                    for l in range(VklcjicjGS_values.shape[0]):
                        if i == k or i == l or j == k or j == l:
                            continue

                        #if np.abs(VklcjicjGS_values[i, j, k, l]) < 1e-10:
                        #    continue
                        #print('ACHTUNG', i, j, k, l)
                        #print(VklcjicjGS_values[i, j, k, l], VklcjicjGS_values_transpose[i, j, k, l])
                        assert np.isclose(VklcjicjGS_values[i, j, k, l], VklcjicjGS_values_transpose[i, j, k, l])
        '''
        rhs -= np.einsum('ijkl,ij->kl', VklcjicjGS_values, self.G.T @ h.T - h.T @ self.G.T, optimize='optimal')



        M = 1.0j / 2. * VklVij(self.G)
        # Vijkl (delta_ij \sum_m i \delta_t \omega_jm Gamma_mm - i \delta_t \omega_ji Gamma_ji)

        # first term Viijk \sum_m i \delta_t \omega_im Gamma_mm
        # \omega_ij A_ij^kl = \sum_i \sum_j V_iikl i \omega_ij \Gamma_jj
        M += 1.0j * np.einsum('jikl,ij->ijkl', VklcjicjGS_values, self.G)  # check here (transposition)
        M -= 1.0j * np.einsum('iikl,jj->ijkl', VklcjicjGS_values, self.G)


        # VklGS (1/2) \sum_ij i \delta_t \omega_ij \Gamma_jj - V klGS i  \delta_t \sum_ij \delta_ij \omega_ij \Gamma_ij
        M += np.einsum('kl,ij->ijkl', VklGS_values,  \
            1.0j / 2. * np.tile(np.diag(self.G)[np.newaxis, :], (self.G.shape[0], 1)))  # check me
        M -= np.einsum('kl,ij->ijkl', \
            VklGS_values, 1.0j / 2. * np.diag(np.diag(self.G)))


        mat = M.reshape((self.G.shape[0] ** 2, -1))
        print('is mat rehm?', np.linalg.norm(mat - mat.conj().T))

        return (np.linalg.inv(M.reshape((self.G.shape[0] ** 2, -1))) @ rhs.flatten()).reshape((self.G.shape[0], -1))


    def Gamma_natural_grad(self, dtauomega, h):
        O = np.einsum('ij,ik,kk->ij', np.eye(h.shape[0]), 1.0j * dtauomega, self.G) - 1.0j * dtauomega * self.G
        return 2 * self.G @ h @ self.G - h @ self.G - self.G @ h - self.G @ O + O @ self.G


        






# ni nj nk nl = cdi ci cdj cj cdk ck cdl cl = (i != j, k != l) = cdi cdj ci cj cdk cdl ck cl
# ci cj cdk cdl = ci (Djk - cdk cj) cdl = Djk (Dli - cdl ci) - (Dki - cdk ci) (Dlj - cdl cj)  = 
# = Djk Dli - Dki Dlj - Djk cdl ci + Dlj cdk ci + Dki cdl cj - cdk (Dli - cdl ci) cj = 
# = Djk Dli - Dki Dlj - Djk cdl ci + Dlj cdk ci + Dki cdl cj - Dli cdk cj + cdk cdl ci cj

# -> cdi cdj [[Djk Dli - Dki Dlj - Djk cdl ci + Dlj cdk ci + Dki cdl cj - Dli cdk cj + cdk cdl ci cj]] ck cl =
# = -Djk Dli <cdi cdj ci cj>OK - Dki Dlj <cdi cdj ci cj>OK - Djk <cdi cdj cdl ci cj cl>OK - Dlj <cdi cdj cdk ci cj ck>OK
# - Dki <cdi cdj cdl ci cj cl>OK - Dli <cdi cdj cdk ci cj ck>OK + <cdi cdj cdk cdl ci cj ck cl>OK

def twoidxijij(G):
    # <cdi cdj ci cj> = -Gii Gjj + Gij Gji
    return -np.einsum('ii,jj->ij', G, G) + np.einsum('ij,ji->ij', G, G)


def twoidxijkl(G):
    # <cdi cdj ck cl> = -Gik Gjl + Gil Gjk
    return -np.einsum('ik,jl->ijkl', G, G) + np.einsum('il,jk->ijkl', G, G)


def threeidx(G):
    # <cdi cdj cdk ci cj ck> = Gii <cdj cdk cj ck> - Gij <cdj cdk ci ck> + Gik <cdj cdk ci cj> =
    # Gii (-Gjj Gkk + Gjk Gkj) - Gij <-Gji Gkk + Gjk Gki> + Gik <-Gji Gkj + Gjj Gki>

    return np.einsum('ii,jk->ijk', G, twoidxijij(G)) - \
           np.einsum('ij,jkik->ijk', G, twoidxijkl(G)) + \
           np.einsum('ik,jkij->ijk', G, twoidxijkl(G))

def fouridx(G):
    # <cdi cdj cdk cdl ci cj ck cl> = -Gii <cdj cdk cdl cj ck cl> + Gij <cdj cdk cdl ci ck cl> - \
    #                                 -Gik <cdj cdk cdl ci cj cl> + Gil <cdj cdk cdl ci cj ck> = 
    # -Gii <cdj cdk cdl cj ck cl> + 
    # + Gij [Gji <cdk cdl ck cl> - Gjk <cdk cdl ci cl> + Gjl <cdk cdl ci ck>] -
    # - Gik [Gji <cdk cdl cj cl> - Gjj <cdk cdl ci cl> + Gjl <cdk cdl ci cj>] +
    # + Gil [Gji <cdk cdl cj ck> - Gjj <cdk cdl ci ck> + Gjk <cdk cdl ci cj>]

    two_ijkl = twoidxijkl(G)

    return -np.einsum('ii,jkl->ijkl', G, threeidx(G)) + \
            np.einsum('ij,ji,kl->ijkl', G, G, twoidxijij(G)) - np.einsum('ij,jk,klil->ijkl', G, G, two_ijkl) + np.einsum('ij,jl,klik->ijkl', G, G, two_ijkl) + \
            np.einsum('ik,ji,kljl->ijkl', -G, G, two_ijkl) - np.einsum('ik,jj,klil->ijkl', -G, G, two_ijkl) + np.einsum('ik,jl,klij->ijkl', -G, G, two_ijkl) + \
            np.einsum('il,ji,kljk->ijkl', G, G, two_ijkl) - np.einsum('il,jj,klik->ijkl', G, G, two_ijkl) + np.einsum('il,jk,klij->ijkl', G, G, two_ijkl)

def VklVij(G):
    # -Djk Dli <cdi cdj ci cj> - Dki Dlj <cdi cdj ci cj> - Djk <cdi cdj cdl ci cj cl> - Dlj <cdi cdj cdk ci cj ck>
    # - Dki <cdi cdj cdl ci cj cl> - Dli <cdi cdj cdk ci cj ck> + <cdi cdj cdk cdl ci cj ck cl> = \
    # <cdi cdj ci cj> (-Djk Dli - Dki Dlj) + <cdi cdj cdl ci cj cl> (-Djk - Dki) + <cdi cdj cdk ci cj ck> (-Dlj - Dli) + <cdi cdj cdk cdl ci cj ck cl>

    two_ijkl = twoidxijkl(G)

    three_ijk = threeidx(G)
    assert np.isclose(np.sum(np.abs(three_ijk.imag)), 0.0)
    four_ijkl = fouridx(G)
    assert np.isclose(np.sum(np.abs(four_ijkl.imag)), 0.0)


    i = np.eye(G.shape[0])
    return -np.einsum('jk,il,ijij->ijkl', i, i, two_ijkl) + np.einsum('ik,jl,ijji->ijkl', i, i, two_ijkl) - \
            np.einsum('ijl,jk->ijkl', three_ijk, i) - np.einsum('ijl,ik->ijkl', three_ijk, i) - \
            np.einsum('ijk,jl->ijkl', three_ijk, i) - np.einsum('ijk,il->ijkl', three_ijk, i) + \
            fouridx(G)

def VklcdicjGS(G):
    two_ijkl = twoidxijkl(G)
    three_ijk = threeidx(G)
    i = np.eye(G.shape[0])
    return np.einsum('ik,illj->ijkl', i, two_ijkl) - np.einsum('il,kikj->ijkl', i, two_ijkl) + \
           np.einsum('kl,likj->ijkl', G, two_ijkl) - np.einsum('kk,lilj->ijkl', G, two_ijkl) + np.einsum('kj,lilk->ijkl', G, two_ijkl)
    
def VklGS(G):
    return -twoidxijij(G)

def VklHpotGS(G, U):
    VklVij_matrix = VklVij(G)
    assert np.isclose(np.sum(np.abs(VklVij_matrix.imag)), 0.0)  # <ni nj nk nl> -- ONLY real-valued

    return np.einsum('ij,ijkl->kl', U, VklVij_matrix)


def Vklcdacb_exp(G, expalpha, a, b):  # <-- here alpha is a diagonal part of ab-dependent alpha
    # -<cdk cdl ck cl cda cb> = <cdk cdl ck cda cl cb> - Dal <cdk cdl ck cb> = 
    # Dak <cdk cdl cl cb> - Dal <cdk cdl ck cb> - <cdk cdl cda ck cl cb> = 
    # Dak <cda cdl cl cb> + Dal <cda cdk ck cb> + <cda cdk cdl cl ck cb>
    def cdicdjcjck_exp(G, expalpha):
        # < cdi cdj cj ck exp(i \sum_c n_c) >
        i = np.eye(G.shape[0])

        det = np.linalg.det(i - G @ (i - expalpha))
        Phi = np.linalg.inv(i - G @ (i - expalpha)) @ G @ expalpha

        return det * (np.einsum('ik,jj->ijk', Phi, Phi) - np.einsum('ij,jk->ijk', Phi, Phi))


    def cdicdjcdkckcjcl_exp(G, expalpha):
        # < cdi cdj cdk ck cj cl exp(i \sum_c n_c) >
        i = np.eye(G.shape[0])

        det = np.linalg.det(i - G @ (i - expalpha))
        Phi = np.linalg.inv(i - G @ (i - expalpha)) @ G @ expalpha

        two_ijkl = twoidxijkl(Phi)
        return det * (np.einsum('il,jkkj->ijkl', Phi, two_ijkl) - \
                      np.einsum('ij,jkkl->ijkl', Phi, two_ijkl) + \
                      np.einsum('ik,jkjl->ijkl', Phi, two_ijkl))

    i = np.eye(G.shape[0])
    threeop = cdicdjcjck_exp(G, expalpha)
    fourop = cdicdjcdkckcjcl_exp(G, expalpha)

    return np.einsum('ak,alb->aklb', i, threeop)[a, :, :, b] + np.einsum('al,akb->aklb', i, threeop)[a, :, :, b] + \
           fourop[a, :, :, b]



def VklHkinGS(G, Omega, K):
    S = Omega.shape[0] // 2
    alpha = np.subtract.outer(Omega, Omega)
    alpha = np.diagonal(alpha.transpose((2, 0, 1, 3)), axis1=-1, axis2=-2)

    Aexp = np.zeros((2 * S, 2 * S, 2 * S, 2 * S), dtype=np.complex128)
    Aexp[..., np.arange(2 * S), np.arange(2 * S)] = np.exp(2.0j * alpha)  # abpq

    exp_small = np.diagonal(Aexp.conj(), axis1=-1, axis2=-2)
    exp_small = np.diagonal(exp_small.transpose((2, 0, 1)), axis1=0, axis2=1).T

    a_vals, b_vals = np.nonzero(K)

    res = np.zeros(G.shape, dtype=np.complex128)
    for a, b in zip(a_vals, b_vals):  # it feels like w.o. this for loop we will get out of memory at 20-30 sites
        res += Vklcdacb_exp(G, Aexp[a, b, ...], a, b) * K[a, b] * exp_small[a, b]
    #np.allclose(res, )
    return res