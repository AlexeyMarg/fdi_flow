import matplotlib.pyplot as plt
import numpy as np

def plot_simulation_results(result):
    """
    Plots the inputs, outputs, and state trajectories from a simulation result dictionary.

    Parameters:
        result (dict): Dictionary with keys 'time', 'inputs', 'outputs', 'states'
    """
    time = result['time']
    inputs = np.array(result['inputs'])
    outputs = np.array(result['outputs'])
    states = np.array(result['states'])

    n_inputs = inputs.shape[1]
    n_outputs = outputs.shape[1]
    n_states = states.shape[1]

    plt.figure(figsize=(16, 10))

    # Plot inputs
    plt.subplot(3, 1, 1)
    for i in range(n_inputs):
        plt.plot(time, inputs[:, i], label=f'u{i+1}')
    plt.title('Control Inputs')
    plt.xlabel('Time [s]')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)

    # Plot outputs
    plt.subplot(3, 1, 2)
    for i in range(n_outputs):
        plt.plot(time, outputs[:, i], label=f'y{i+1}')
    plt.title('System Outputs')
    plt.xlabel('Time [s]')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)

    # Plot states
    plt.subplot(3, 1, 3)
    for i in range(n_states):
        plt.plot(time, states[:, i], label=f'x{i+1}')
    plt.title('State Variables')
    plt.xlabel('Time [s]')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
