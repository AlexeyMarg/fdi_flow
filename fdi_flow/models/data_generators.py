import numpy as np

class DynamicSystemDataGenerator:
    """
    A data generator for dynamic system models (linear or nonlinear, continuous or discrete).
    
    This generator uses a system model that implements a `.simulate()` method with parameters (u, x0, t_final, dt),
    and produces datasets for different input signals and initial conditions.

    Parameters:
        model (object): A dynamic system model instance with a `.simulate()` method.
        x0_bounds (tuple): (low, high) bounds for random initial conditions.
    """
    def __init__(self, model, x0_bounds):
        self.model = model
        self.x0_bounds = x0_bounds

    def generate(self, num_simulations, signal_type='constant', 
                 signal_params=None, t_final=10.0, dt=0.01):
        """
        Generate multiple simulation results using randomized initial conditions and input signals.

        Parameters:
            num_simulations (int): Number of simulations to generate.
            signal_type (str): Type of control input signal: 'constant' or 'sine'.
            signal_params (dict): Parameters for signal generation:
                For 'constant':
                    - min_val, max_val (float): Value range
                    - distribution (str): 'uniform' or 'normal'
                For 'sine':
                    - amplitude_range (tuple): Min/max amplitude
                    - offset_range (tuple): Min/max offset
                    - phase_range (tuple): Min/max phase
                    - frequency (float): Sine wave frequency
            t_final (float): Total duration of simulation.
            dt (float): Time step for simulation (integration or discrete step).

        Returns:
            List[dict]: A list of simulation results (each as a dict with keys 'time', 'states', 'outputs', 'inputs').
        """
        results = []
        model = self.model
        x0_low, x0_high = np.array(self.x0_bounds[0]), np.array(self.x0_bounds[1])
        state_dim = len(x0_low)

        for _ in range(num_simulations):
            # Generate random initial condition
            x0 = np.random.uniform(x0_low, x0_high)

            # Determine number of time steps
            n_steps = int(t_final / dt)

            # Generate input signal
            if signal_type == 'constant':
                dist = signal_params.get('distribution', 'uniform')
                min_val = signal_params.get('min_val', -1.0)
                max_val = signal_params.get('max_val', 1.0)
                m = model.m if hasattr(model, 'm') else signal_params.get('input_dim', 1)

                if dist == 'uniform':
                    u_val = np.random.uniform(min_val, max_val, size=(m,))
                else:  # 'normal'
                    u_val = np.random.normal(loc=(min_val + max_val) / 2,
                                             scale=(max_val - min_val) / 6,
                                             size=(m,))
                u = np.tile(u_val, (n_steps, 1))

            elif signal_type == 'sine':
                amplitude_range = signal_params.get('amplitude_range', (0.5, 1.0))
                offset_range = signal_params.get('offset_range', (-1.0, 1.0))
                phase_range = signal_params.get('phase_range', (0, 2 * np.pi))
                frequency = signal_params.get('frequency', 1.0)
                m = model.m if hasattr(model, 'm') else signal_params.get('input_dim', 1)
                time = np.linspace(0, t_final, n_steps)

                u = np.zeros((n_steps, m))
                for j in range(m):
                    amp = np.random.uniform(*amplitude_range)
                    off = np.random.uniform(*offset_range)
                    phase = np.random.uniform(*phase_range)
                    u[:, j] = off + amp * np.sin(2 * np.pi * frequency * time + phase)

            else:
                raise ValueError(f"Unsupported signal type: {signal_type}")

            # Run the simulation with generated input and initial state
            result = model.simulate(u=u, x0=x0, t_final=t_final, dt=dt)
            results.append(result)

        return results
