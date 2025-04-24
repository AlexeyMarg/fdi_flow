import numpy as np
from scipy.integrate import solve_ivp

class LinearContinuousStateSpaceModel:
    """
    Simulates a linear continuous-time MIMO dynamic system using either Euler method or scipy ODE solvers.

    State-space form:
        dx/dt = A x + B u(t)
        y = C x + D u(t)
    """
    def __init__(self, A, B, C, D):
        self.A = np.array(A)
        self.B = np.array(B)
        self.C = np.array(C)
        self.D = np.array(D)

        self.n = self.A.shape[0]
        self.m = self.B.shape[1]
        self.p = self.C.shape[0]

    def simulate(self, u, x0, t_final, dt, method='euler'):
        """
        Simulate the system dynamics.

        Parameters:
            u (ndarray or list): Control inputs of shape (n_steps, m)
            x0 (ndarray): Initial state vector of shape (n,)
            t_final (float): Simulation time
            dt (float): Time step
            method (str): 'euler' or scipy solver name like 'RK45', 'Radau'

        Returns:
            dict: {
                "time": ndarray,
                "states": ndarray,
                "outputs": ndarray,
                "inputs": ndarray
            }
        """
        u = np.array(u)
        x0 = np.array(x0)
        n_steps = int(t_final / dt)
        time = np.linspace(0, t_final, n_steps)

        def u_interp(t):
            i = min(int(t / dt), len(u) - 1)
            return u[i]

        def rhs(t, x):
            ui = u_interp(t)
            return self.A @ x + self.B @ ui

        if method == 'euler':
            x = x0
            states = np.zeros((n_steps, self.n))
            outputs = np.zeros((n_steps, self.p))
            inputs = np.zeros((n_steps, self.m))

            for i in range(n_steps):
                ui = u[i] if i < len(u) else u[-1]
                dx = self.A @ x + self.B @ ui
                x = x + dx * dt
                y = self.C @ x + self.D @ ui

                states[i] = x
                outputs[i] = y
                inputs[i] = ui

        else:
            # Решение с помощью solve_ivp
            sol = solve_ivp(rhs, (0, t_final), x0, t_eval=time, method=method)
            states = sol.y.T
            inputs = np.array([u_interp(t) for t in time])
            outputs = np.array([self.C @ x + self.D @ u for x, u in zip(states, inputs)])

        return {
            "time": time,
            "states": states,
            "outputs": outputs,
            "inputs": inputs
        }

class LinearDiscreteStateSpaceModel:
    """
    Simulates a discrete-time linear MIMO system:
        x[k+1] = A x[k] + B u[k]
        y[k] = C x[k] + D u[k]

    Parameters:
        A (ndarray): State matrix
        B (ndarray): Input matrix
        C (ndarray): Output matrix
        D (ndarray): Feedthrough matrix
    """
    def __init__(self, A, B, C, D):
        self.A = np.array(A)
        self.B = np.array(B)
        self.C = np.array(C)
        self.D = np.array(D)

        self.n = self.A.shape[0]
        self.m = self.B.shape[1]
        self.p = self.C.shape[0]

    def simulate(self, u, x0, dt=1.0):
        """
        Simulate the system response to a control input sequence.

        Parameters:
            u (ndarray): Input signal, shape (n_steps, m)
            x0 (ndarray): Initial state vector, shape (n,)
            dt (float): Discrete time step (default = 1.0)

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

        states = np.zeros((n_steps, self.n))
        outputs = np.zeros((n_steps, self.p))
        inputs = np.zeros((n_steps, self.m))
        time = np.arange(n_steps) * dt

        x = x0
        for k in range(n_steps):
            uk = u[k]
            yk = self.C @ x + self.D @ uk
            x = self.A @ x + self.B @ uk

            states[k] = x
            outputs[k] = yk
            inputs[k] = uk

        return {
            "time": time,
            "states": states,
            "outputs": outputs,
            "inputs": inputs
        }
