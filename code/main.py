import varstate
import numpy as np


n_opt = 1000
lr = 1e-2


N_sites = 20
U = 1.


### initialize random starting G ###
G = np.random.uniform(-1, 1, size=(2 * N_sites, 2 * N_sites)) + 1.0j * np.random.uniform(-1, 1, size=(2 * N_sites, 2 * N_sites))
G = G + G.conj().T

### initialize random starting Omega ###
O = np.random.uniform(-1, 1, size=(2 * N_sites, 2 * N_sites)) / 10.
O = O + O.T



### nearest-neighbor hopping with PBC ###
H = np.zeros((2 * N_sites, 2 * N_sites))
H[np.arange(N_sites), (np.arange(N_sites) + 1) % N_sites] = 1.0

H[np.arange(N_sites) + N_sites, N_sites + ((np.arange(N_sites) + 1) % N_sites)] = 1.0

H = H + H.T

state = varstate.VarState(N_sites, H, U, G, O)

for n_iter in range(n_opt):
	energy = state.energy()

	print('energy = {:.5f} + i ({:.3f})'.format(energy.real, energy.imag))
	grad_G, grad_O = state.gradient()

	G -= lr * grad_G
	O -= lr * grad_O