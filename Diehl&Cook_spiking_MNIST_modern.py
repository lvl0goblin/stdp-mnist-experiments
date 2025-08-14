import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
from tqdm import tqdm
from brian2 import *

#-------------------------------------------------------------------------------
# Paths and data
#-------------------------------------------------------------------------------
MNIST_data_path = Path('data')
weight_path = Path('weights')
data_path = Path('./activity')
data_path.mkdir(exist_ok=True, parents=True)

#-------------------------------------------------------------------------------
# Functions
#-------------------------------------------------------------------------------
def load_mnist(filename, bTrain=True):
    """Load MNIST data from .pickle or raw MNIST files."""
    file_path = MNIST_data_path / (filename + '.pickle')
    if file_path.exists():
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
    else:
        if bTrain:
            images_file = MNIST_data_path / 'train-images-idx3-ubyte'
            labels_file = MNIST_data_path / 'train-labels-idx1-ubyte'
        else:
            images_file = MNIST_data_path / 't10k-images-idx3-ubyte'
            labels_file = MNIST_data_path / 't10k-labels-idx1-ubyte'
        
        with open(images_file, 'rb') as f:
            f.read(4)
            N = int.from_bytes(f.read(4), 'big')
            rows = int.from_bytes(f.read(4), 'big')
            cols = int.from_bytes(f.read(4), 'big')
            x = np.frombuffer(f.read(), dtype=np.uint8).reshape((N, rows, cols))
        with open(labels_file, 'rb') as f:
            f.read(4)
            y = np.frombuffer(f.read(), dtype=np.uint8).reshape((N, 1))
        data = {'x': x, 'y': y, 'rows': rows, 'cols': cols}
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
    return data

#-------------------------------------------------------------------------------
# Load MNIST
#-------------------------------------------------------------------------------
training = load_mnist('training')
testing = load_mnist('testing', bTrain=False)

#-------------------------------------------------------------------------------
# Parameters
#-------------------------------------------------------------------------------
n_input = 784
n_e = 400
n_i = n_e

single_example_time = 0.35*second
resting_time = 0.15*second
num_examples = 10000

v_rest_e, v_rest_i = -65*mV, -60*mV
v_reset_e, v_reset_i = -65*mV, -45*mV
v_thresh_e, v_thresh_i = -52*mV, -40*mV
refrac_e, refrac_i = 5*ms, 2*ms
input_intensity = 2.0

#-------------------------------------------------------------------------------
# Neuron equations
#-------------------------------------------------------------------------------
eqs_e = '''
dv/dt = ((v_rest_e - v) + (I_synE+I_synI)/nS)/(100*ms) : volt
I_synE = ge*nS*(-v) : amp
I_synI = gi*nS*(-100*mV - v) : amp
dge/dt = -ge/(1*ms) : 1
dgi/dt = -gi/(2*ms) : 1
theta : volt
'''

eqs_i = '''
dv/dt = ((v_rest_i - v) + (I_synE+I_synI)/nS)/(10*ms) : volt
I_synE = ge*nS*(-v) : amp
I_synI = gi*nS*(-85*mV - v) : amp
dge/dt = -ge/(1*ms) : 1
dgi/dt = -gi/(2*ms) : 1
'''

#-------------------------------------------------------------------------------
# Neuron groups
#-------------------------------------------------------------------------------
neurons_e = NeuronGroup(n_e, eqs_e, threshold='v>v_thresh_e', reset='v=v_reset_e',
                        refractory=refrac_e, method='euler')
neurons_i = NeuronGroup(n_i, eqs_i, threshold='v>v_thresh_i', reset='v=v_reset_i',
                        refractory=refrac_i, method='euler')

neurons_e.v = v_rest_e - 40*mV
neurons_i.v = v_rest_i - 40*mV

# Load pretrained theta
theta_file = weight_path / 'theta_A.npy'
if theta_file.exists():
    neurons_e.theta = np.load(theta_file) * mV
else:
    neurons_e.theta = 20*mV

#-------------------------------------------------------------------------------
# Input Poisson group
#-------------------------------------------------------------------------------
input_group = PoissonGroup(n_input, rates=0*Hz)

#-------------------------------------------------------------------------------
# Synapses (pretrained)
#-------------------------------------------------------------------------------
weight_file = weight_path / 'XeAe.npy'
W = np.load(weight_file)  # shape (313600, 3)
# Use second column and reshape to (n_input, n_e)
W = W[:, 2].reshape((n_input, n_e))

syn_input = Synapses(input_group, neurons_e, model='w:1', on_pre='ge += w')
syn_input.connect()  # all-to-all
syn_input.w[:] = W.flatten()

#-------------------------------------------------------------------------------
# Monitors
#-------------------------------------------------------------------------------
rate_mon_e = PopulationRateMonitor(neurons_e)

#-------------------------------------------------------------------------------
# Simulation loop
#-------------------------------------------------------------------------------
result_monitor = np.zeros((num_examples, n_e))
input_numbers = np.zeros(num_examples, dtype=int)

for j in tqdm(range(num_examples), desc="Simulating MNIST"):
    spike_mon_e = SpikeMonitor(neurons_e)  # reset for each example
    rates = testing['x'][j % 10000].reshape(-1)/8.0 * input_intensity
    input_group.rates = rates*Hz
    run(single_example_time)
    result_monitor[j, :] = spike_mon_e.count[:]
    input_numbers[j] = testing['y'][j % 10000, 0]
    input_group.rates = 0*Hz
    run(resting_time)

#-------------------------------------------------------------------------------
# Save activity
#-------------------------------------------------------------------------------
np.save(data_path / f'resultPopVecs_{num_examples}.npy', result_monitor)
np.save(data_path / f'inputNumbers_{num_examples}.npy', input_numbers)

#-------------------------------------------------------------------------------
# Plot population firing rate
#-------------------------------------------------------------------------------
plt.figure(figsize=(6,4))
plt.plot(rate_mon_e.t/second, rate_mon_e.smooth_rate(window='flat', width=10*ms))
plt.xlabel('Time (s)')
plt.ylabel('Population firing rate')
plt.show()
