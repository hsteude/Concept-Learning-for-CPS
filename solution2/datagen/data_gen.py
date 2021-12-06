import numpy as np
from scipy.integrate import odeint
import solution2.constants as const

np.random.seed(62654)

class ThreeTankDataGenerator():

    def __init__(self, sample_size=const.NUMBER_OF_SAMPLES,
                 number_timesteps=const.NUMBER_TIMESTEPS, t_max=const.T_MAX,
                 A=const.A, g=const.G):
        self.sample_size = sample_size
        self.t_max = t_max
        self.number_timesteps = number_timesteps
        self.C = np.sqrt(2*g)/A
        self.t = np.linspace(0, self.t_max, self.number_timesteps)
        self.initial_states = np.array(const.INITIAL_STATES)
        self.number_signals = 3

    def system_dynamics_function(self, x, t, q1, q3, kv12, kv23):
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        dh1_dt = self.C * q1 - kv12 * self.C * \
            np.sign(x1 - x2) * np.sqrt(np.abs(x1 - x2))
        dh2_dt = kv12 * self.C * np.sign(x1 - x2) * np.sqrt(np.abs(x1 - x2))\
            - kv23 * self.C * np.sign(x2 - x3) * np.sqrt(np.abs(x2 - x3))
        dh3_dt = kv23 * self.C * q3 + self.C * \
            np.sign(x2 - x3) * np.sqrt(np.abs(x2 - x3))
        return dh1_dt, dh2_dt, dh3_dt

    def get_random_samples_for_underl_factors(self, N):
        q1_samples = np.array(np.random.uniform(
            low=const.Q1_MIN, high=const.Q1_MAX, size=N))
        q3_samples = np.array(np.random.uniform(
            low=const.Q3_MIN, high=const.Q3_MAX, size=N))
        kv12_samples = np.array(np.random.uniform(
            low=const.KV12_MIN, high=const.KV12_MAX, size=N))
        kv23_samples = np.array(np.random.uniform(
            low=const.KV23_MIN, high=const.KV23_MAX, size=N))
        return q1_samples, q3_samples, kv12_samples, kv23_samples

    def compute_answers(self, q1, q3, kv12, kv23):
        ans_question_1 = const.TANK_MAX_FILL / q1
        ans_question_2 = const.TANK_MAX_FILL / q3
        ans_question_3 = const.TANK_MAX_FILL / kv12
        ans_question_4 = const.TANK_MAX_FILL / kv23
        return ans_question_1, ans_question_2, ans_question_3, ans_question_4

    def solve_ode(self, initial_state, q1, q3, kv12, kv23):
        return odeint(self.system_dynamics_function, initial_state, self.t, (q1, q3, kv12, kv23))

    def generate(self):
        q1, q3, kv12, kv23 = self.get_random_samples_for_underl_factors(
            self.sample_size)
        ans_question_1, ans_question_2, ans_question_3, ans_question_4 = \
            self.compute_answers(q1, q3, kv12, kv23)
        x = np.zeros((self.number_timesteps *
                     self.sample_size, self.number_signals))
        time = np.array(list(self.t)*self.sample_size)
        uid_ts_sample = np.array(
            [[i]*self.number_timesteps for i in range(self.sample_size)]).ravel()
        for i in range(self.sample_size):
            x_i = self.solve_ode(self.initial_states,
                                 q1[i], q3[i], kv12[i], kv23[i])
            start_idx = i*self.number_timesteps
            end_idx = i*self.number_timesteps+self.number_timesteps
            x[start_idx:end_idx, :] = x_i
        return x, time, uid_ts_sample, q1, q3, kv12, kv23, \
            ans_question_1, ans_question_2, ans_question_3, ans_question_4


