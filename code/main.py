import varstate
import sys
import os
import config as cv_module
#from code.utils import *
import numpy as np


def import_config(filename: str):
    import importlib

    module_name, extension = os.path.splitext(os.path.basename(filename))
    module_dir = os.path.dirname(filename)
    if extension != ".py":
        raise ValueError(
            "Could not import the module from {!r}: not a Python source file.".format(
                filename
            )
        )
    if not os.path.exists(filename):
        raise ValueError(
            "Could not import the module from {!r}: no such file or directory".format(
                filename
            )
        )
    sys.path.insert(0, module_dir)
    module = importlib.import_module(module_name)
    sys.path.pop(0)
    return module

#np.random.seed(240)
config_file = import_config(sys.argv[1])
config_import = config_file.opt_parameters()

opt_config = cv_module.opt_parameters()
opt_config.__dict__ = config_import.__dict__.copy()



n_opt = opt_config.n_opt
lr = opt_config.lr

N_sites = opt_config.N_sites
U = opt_config.U


state = varstate.VarState(opt_config)

#energy, h_matrix = state.energy_derivative()
'''
energy_check = PotentialExpectationValue(state.G, state.O, U, N_sites, state.H).get_energy() + KineticExpectationValue(state.G, state.O, U, N_sites, state.H).get_energy()
print('energy', np.allclose(energy, energy_check))
derivatives = Derivatives(state.G,
                 state.O,
                 U,
                 N_sites,
                 state.H)
print('h', np.allclose(h_matrix, derivatives.get_gamma_num_derivs(PotentialExpectationValue) + derivatives.get_gamma_num_derivs(KineticExpectationValue).T))
'''

for n_iter in range(n_opt):
    energy, h_matrix, o_matrix = state.energy_derivative()

    '''    
    for i in range(opt_config.N_sites * 2):
        for j in range(opt_config.N_sites * 2):
            state.G[i, j] += 1e-7
            state.G[j, i] += 1e-7

            energy_der, _ = state.energy_derivative()
            print('REAL', h_matrix[i, j].real, (energy_der - energy).real / 1e-7 / 2.)

            state.G[i, j] -= 1e-7
            state.G[j, i] -= 1e-7

            state.G[i, j] += 1e-7j
            state.G[j, i] -= 1e-7j

            energy_der, _ = state.energy_derivative()
            print('IMAG', h_matrix[i, j].imag, -(energy_der - energy).real / 1e-7 / 2.)

            state.G[i, j] -= 1e-7j
            state.G[j, i] += 1e-7j


    ### END DEBUG ###
    
    '''
    print('energy = {:.5f} + i ({:.3f})'.format(energy.real, energy.imag))
    grad_G, grad_O = state.gradient(energy, h_matrix, o_matrix)

    print(np.linalg.norm(grad_G - grad_G.conj().T))
    print(np.linalg.norm(grad_O.imag))

    print('current density', np.trace(state.G))

    state.G -= lr * grad_G
    state.O -= lr * grad_O
    state.restore_idempotent_form()