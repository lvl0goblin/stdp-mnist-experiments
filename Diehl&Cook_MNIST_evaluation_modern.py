import numpy as np
from pathlib import Path

#-------------------------------------------------------------------------------
# Paths
#-------------------------------------------------------------------------------
data_path = Path('/workspaces/stdp-mnist-experiments/activity')
training_ending = '10000'
testing_ending = '10000'

#-------------------------------------------------------------------------------
# Network parameters
#-------------------------------------------------------------------------------
n_e = 400

#-------------------------------------------------------------------------------
# Functions
#-------------------------------------------------------------------------------
def get_recognized_number_ranking(assignments, spike_rates):
    summed_rates = np.zeros(10)
    num_assignments = np.zeros(10)
    for i in range(10):
        num_assignments[i] = np.sum(assignments == i)
        if num_assignments[i] > 0:
            summed_rates[i] = np.sum(spike_rates[assignments == i]) / num_assignments[i]
    return np.argsort(summed_rates)[::-1]

def get_new_assignments(result_monitor, input_numbers):
    print(result_monitor.shape)
    assignments = -1 * np.ones(n_e, dtype=int)
    input_nums = np.asarray(input_numbers)
    maximum_rate = np.zeros(n_e)
    for j in range(10):
        num_inputs = np.sum(input_nums == j)
        if num_inputs > 0:
            rate = np.sum(result_monitor[input_nums == j], axis=0) / num_inputs
            for i in range(n_e):
                if rate[i] > maximum_rate[i]:
                    maximum_rate[i] = rate[i]
                    assignments[i] = j
    return assignments

#-------------------------------------------------------------------------------
# Load results
#-------------------------------------------------------------------------------
training_result_monitor = np.load(data_path / f'resultPopVecs_{training_ending}.npy')
training_input_numbers = np.load(data_path / f'inputNumbers_{training_ending}.npy')
testing_result_monitor = np.load(data_path / f'resultPopVecs_{testing_ending}.npy')
testing_input_numbers = np.load(data_path / f'inputNumbers_{testing_ending}.npy')

print(training_result_monitor.shape)

#-------------------------------------------------------------------------------
# Compute neuron assignments
#-------------------------------------------------------------------------------
assignments = get_new_assignments(training_result_monitor, training_input_numbers)
print("Neuron assignments:", assignments)

#-------------------------------------------------------------------------------
# Evaluate accuracy on testing set
#-------------------------------------------------------------------------------
num_tests = testing_result_monitor.shape[0] // 10000
sum_accuracy = np.zeros(num_tests)

for counter in range(num_tests):
    start_idx = counter * 10000
    end_idx = min(testing_result_monitor.shape[0], (counter+1) * 10000)
    
    test_results = np.zeros((10, end_idx - start_idx), dtype=int)
    
    print(f"Calculating accuracy for batch {counter+1}/{num_tests}")
    for i in range(end_idx - start_idx):
        test_results[:, i] = get_recognized_number_ranking(assignments,
                                                           testing_result_monitor[start_idx + i, :])
    
    difference = test_results[0, :] - testing_input_numbers[start_idx:end_idx]
    correct = np.sum(difference == 0)
    incorrect = np.where(difference != 0)[0]
    
    sum_accuracy[counter] = correct / float(end_idx - start_idx) * 100
    print(f"Batch {counter+1} - Sum response accuracy: {sum_accuracy[counter]:.2f}%  Num incorrect: {len(incorrect)}")

print(f"Overall sum response accuracy --> mean: {np.mean(sum_accuracy):.2f}%, std: {np.std(sum_accuracy):.2f}%")
