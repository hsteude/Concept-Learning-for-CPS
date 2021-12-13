import numpy as np
from scipy.integrate import odeint
import solution3.constants as const
from sklearn.preprocessing import PolynomialFeatures


np.random.seed(62654)


class ThreeTankDataGenerator():

    def __init__(self, number_initial_states=2,
                 number_timesteps=const.NUMBER_TIMESTEPS, t_max=const.T_MAX,
                 q1=const.Q1, q3=const.Q3, A=const.A, g=const.G, latent_dim=const.LATENT_DIM,
                 derivatives=False):
        self.t_max = t_max
        self.number_timesteps = number_timesteps
        self.q1 = q1
        self.q3 = q3
        self.C = np.sqrt(2*g)/A
        self.t = np.linspace(0, self.t_max, self.number_timesteps)
        self.latent_dim = latent_dim
        self.number_initial_states = number_initial_states
        self.initial_states = self.get_random_inition_states(
            N=number_initial_states)
        self.derivatives = derivatives
        self.z_scaling_factor = 100

    def system_dynamics_function(self, x, t):
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        dh1_dt = self.C * self.q1 - self.C * \
            np.sign(x1 - x2) * np.sqrt(np.abs(x1 - x2))
        dh2_dt = self.C * np.sign(x1 - x2) * np.sqrt(np.abs(x1 - x2))\
            - self.C * np.sign(x2 - x3) * np.sqrt(np.abs(x2 - x3))
        dh3_dt = self.C * self.q3 + self.C * \
            np.sign(x2 - x3) * np.sqrt(np.abs(x2 - x3))
        return dh1_dt, dh2_dt, dh3_dt

    def get_random_inition_states(self, N):
        initial_states = np.array(np.random.uniform(low=const.INITIAL_LEVEL_MIN,
                                                    high=const.INITIAL_LEVEL_MAX,
                                                    size=N*self.latent_dim))
        return initial_states.reshape(N, self.latent_dim)

    def solve_ode(self, initial_state):
        return odeint(self.system_dynamics_function, initial_state, self.t)

    def compute_derivatives(self, x, dt):
        """
        First order forward difference (forward difference)
        TODO: Find out how the pysindy authors, came up with the formula for the start and end points
        """
        # Uniform timestep (assume t contains dt)
        x_dot = np.full_like(x, fill_value=np.nan)
        if np.isscalar(dt):
            x_dot[:-1, :] = (x[1:, :] - x[:-1, :]) / dt
            x_dot[-1, :] = (3 * x[-1, :] / 2 - 2 *
                            x[-2, :] + x[-3, :] / 2) / dt
        return x_dot


    def generate_x_space_data(self):
        z = np.zeros((self.number_timesteps *
                     self.number_initial_states, self.latent_dim))
        z_dot = np.zeros(
            (self.number_timesteps * self.number_initial_states, self.latent_dim))
        time = np.array(list(self.t)*self.number_initial_states)
        uid_initial_state = np.array(
            [[i]*self.number_timesteps for i in range(self.number_initial_states)]).ravel()
        poly = PolynomialFeatures(const.POLY_ORDER)
        poly.fit(np.identity(const.LATENT_DIM))
        x = np.zeros((self.number_timesteps *
                     self.number_initial_states, poly.n_output_features_))
        x_dot = np.zeros(
            (self.number_timesteps * self.number_initial_states, poly.n_output_features_))
        for i in range(self.number_initial_states):
            # compute time (in z space) series for initial state i
            z_i = self.solve_ode(self.initial_states[i, :])/ self.z_scaling_factor
            z_dot_i = self.compute_derivatives(
                z_i, dt=self.t_max / (self.number_timesteps - 1))
            x_i = poly.fit_transform(z_i)
            x_dot_i = self.compute_derivatives(
                x_i, dt=self.t_max / (self.number_timesteps - 1))
            start_idx = i*self.number_timesteps
            end_idx = i*self.number_timesteps+self.number_timesteps
            z[start_idx:end_idx, :] = z_i
            x[start_idx:end_idx, :] = x_i
            x_dot[start_idx:end_idx, :] = x_dot_i
            z_dot[start_idx:end_idx, :] = z_dot_i
        return x, x_dot, z, z_dot, time, uid_initial_state
