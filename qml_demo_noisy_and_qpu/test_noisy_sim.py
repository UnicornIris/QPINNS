# %%
import os
from math import pi
import numpy as np
import random
import itertools
import scipy.optimize as opt
import matplotlib.pyplot as plt
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as AerSampler
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime.fake_provider import FakeBrisbane
from qiskit_ibm_runtime import Session
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService

# %%
shots = 1000

# %%
backend = FakeBrisbane()
sampler = Sampler(backend)

# %%
pm = generate_preset_pass_manager(backend=backend)

# %%
theta_vec = ParameterVector(r"$\theta$", length=1)

ansatz = QuantumCircuit(3)
ansatz.h(0)
ansatz.cx(0, 1)
ansatz.rxx(theta_vec[0], 1, 2)
ansatz.measure_all()  # don't forget this line
# ansatz.draw("mpl")

# %%
ansatz_isa = pm.run(ansatz)
# ansatz_isa.draw("mpl", idle_wires=False)

# %%
qc_isa = ansatz_isa.assign_parameters({theta_vec[0]: 2*pi/5})

# %%
result = sampler.run([qc_isa], shots = shots).result()
counts = result[0].data.meas.get_counts()
print(counts)

# %%



