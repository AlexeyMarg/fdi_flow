import numpy as np
from scipy.integrate import solve_ivp


class NonlinearStateSpaceModel:
    """
    Simulates a nonlinear continuous-time MIMO system of the form:
        dx/dt = f(x, u, t)
        y = g(x, u, t)

    The user provides the functions f and g as callables.

    Parameters:
        f (callable): Function f(x, u, t) → dx/dt
        g (callable): Function g(x, u, t) → output vector
    """
    def __init__(self, f, g):
        self.f = f  # динамика системы
        self.g = g  # выходная функция

    def simulate(self, u, x0, t_final, dt, method='RK45'):
        """
        Simulate the nonlinear system.

        Parameters:
            u (ndarray or list): Control input signals, shape (n_steps, m)
            x0 (ndarray): Initial state vector, shape (n,)
            t_final (float): Final simulation time
            dt (float): Time step
            method (str): Integration method (e.g., 'RK45', 'Radau', etc.)

        Returns:
            dict: {
                "time": ndarray of time values,
                "states": ndarray of states over time,
                "outputs": ndarray of outputs over time,
                "inputs": ndarray of inputs over time
            }
        """
        u = np.array(u)
        x0 = np.array(x0)
        n_steps = int(t_final / dt)
        time = np.linspace(0, t_final, n_steps)

        def u_interp(t):
            i = min(int(t / dt), len(u) - 1)
            return u[i]

        def ode(t, x):
            return self.f(x, u_interp(t), t)

        sol = solve_ivp(ode, (0, t_final), x0, t_eval=time, method=method)
        states = sol.y.T
        inputs = np.array([u_interp(t) for t in time])
        outputs = np.array([self.g(x, u_interp(t), t) for x, t in zip(states, time)])

        return {
            "time": time,
            "states": states,
            "outputs": outputs,
            "inputs": inputs
        }
        
        
class NonlinearDiscreteStateSpaceModel:
    """
    Simulates a discrete-time nonlinear MIMO system:
        x[k+1] = f(x[k], u[k], k)
        y[k] = g(x[k], u[k], k)

    Parameters:
        f (callable): State transition function f(x, u, k) -> x_next
        g (callable): Output function g(x, u, k) -> y
    """
    def __init__(self, f, g):
        self.f = f  
        self.g = g  
        
    def simulate(self, u, x0, dt=1.0):
        """
        Simulate the system response.

        Parameters:
            u (ndarray): Input signal, shape (n_steps, m)
            x0 (ndarray): Initial state, shape (n,)
            dt (float): Time step (default = 1.0)

        Returns:
            dict: {
                "time": ndarray of shape (n_steps,),
                "states": ndarray of shape (n_steps, n),
                "outputs": ndarray of shape (n_steps, p),
                "inputs": ndarray of shape (n_steps, m)
            }
        """
        u = np.array(u)
        x0 = np.array(x0)
        n_steps = u.shape[0]
        m = u.shape[1]

        x = x0
        y_sample = self.g(x, u[0], 0)
        n = len(x)
        p = len(y_sample)

        states = np.zeros((n_steps, n))
        outputs = np.zeros((n_steps, p))
        inputs = np.zeros((n_steps, m))
        time = np.arange(n_steps) * dt

        for k in range(n_steps):
            uk = u[k]
            yk = self.g(x, uk, k)
            x_next = self.f(x, uk, k)

            states[k] = x
            outputs[k] = yk
            inputs[k] = uk
            x = x_next

        return {
            "time": time,
            "states": states,
            "outputs": outputs,
            "inputs": inputs
        }