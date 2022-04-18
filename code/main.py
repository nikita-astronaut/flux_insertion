import varstate
import numpy as np
import sys
import os
import config as cv_module
from ExpectationValue import *
from Derivatives import *

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


config_file = import_config(sys.argv[1])
config_import = config_file.opt_parameters()

opt_config = cv_module.opt_parameters()
opt_config.__dict__ = config_import.__dict__.copy()

n_opt = opt_config.n_opt
lr = opt_config.lr

N_sites = opt_config.N_sites
U = opt_config.U

state = varstate.VarState(opt_config)

for n_iter in range(n_opt):
    energy = state.energy()

    pot_en = PotentialExpectationValue(state.G, state.O, state.U, state.nsites, state.H).get_energy()
    kin_en = KineticExpectationValue(state.G, state.O, state.U, state.nsites, state.H).get_energy()
    if not np.isclose(energy, pot_en + kin_en):
        raise ValueError('Energies do not match')
    print('energy = {:.5f} + i ({:.3f})'.format(energy.real, energy.imag))

    # grad_G, grad_O = state.gradient()
    grad_G = state.gradient()

    derivatives = Derivatives(state.G, state.O, state.U, state.nsites, state.H)
    grad_num_G = derivatives.get_gamma_num_derivs(PotentialExpectationValue)
    grad_num_G += derivatives.get_gamma_num_derivs(KineticExpectationValue)
    if not np.allclose(grad_G, grad_num_G):
        raise ValueError('Gradient is not correct')

    state.G -= lr * grad_G

    if not np.allclose(state.G @ state.G, state.G):
        raise ValueError('Gamma is not idempotent')
    # G -= lr * grad_G
# O -= lr * grad_O
