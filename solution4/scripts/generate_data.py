import numpy as np
import pandas as pd
from scipy.integrate import odeint

from solution4.scripts.create_figures import plot_simulation


class ThreeTankSimulation:
    def __init__(self, tank_1_lvl=0, tank_2_lvl=0, tank_3_lvl=0):
        self.tank_levels = np.array([tank_1_lvl, tank_2_lvl, tank_3_lvl])
        self.state_df = pd.DataFrame(columns=["q1", "q3", "kv1", "kv2", "kv3", "duration"])

    def add_state(self, q1: float, q3: float, kv1: float, kv2: float, kv3: float, duration: int, name=None) -> None:
        """Add a state to the state dataframe.
        A state consists of specific settings to the system's parameters.
        Args:
            q1 (float): inflow tank 1
            q3 (float): inflow tank 3
            kv1 (float): coefficient of the valve between tank 1 and 2
            kv2 (float): coefficient of the valve between tank 2 and 3
            kv3 (float): coefficient of the outgoing valve on tank 3
            duration (int): number of time steps of the state
            name (string): the name of the state
        """
        duration = int(duration)
        if name is not None:
            self.state_df.loc[name] = [q1, q3, kv1, kv2, kv3, duration]
        else:
            self.state_df.append(dict(q1=q1, q3=q3, kv1=kv1, kv2=kv2, kv3=kv3, duration=duration),
                                 ignore_index=True)

    @staticmethod
    def _system_dynamics_function(x, t, q1, q3, kv1, kv2, kv3):
        # set constants
        A = 5
        g = 9.81
        C = np.sqrt(2 * g) / A
        x1, x2, x3 = x * (x > 0)  # ensure non-negative tank levels

        # ODE
        dh1_dt = C * q1 - kv1 * C * np.sign(x1 - x2) * np.sqrt(np.abs(x1 - x2))
        dh2_dt = kv1 * C * np.sign(x1 - x2) * np.sqrt(np.abs(x1 - x2)) \
                 - kv2 * C * np.sign(x2 - x3) * np.sqrt(np.abs(x2 - x3))
        dh3_dt = C * q3 + kv2 * C * np.sign(x2 - x3) * np.sqrt(np.abs(x2 - x3)) - kv3 * np.sqrt(x3)

        return dh1_dt, dh2_dt, dh3_dt

    def _compute_section(self, duration: int = 10, x0: np.array = np.array([30, 10, 50]),
                         kv1: float = 1, kv2: float = 1, kv3: float = 1,
                         q1: float = 1, q3: float = 1):
        t = np.array(range(duration))
        y = odeint(self._system_dynamics_function, x0, t, (q1, q3, kv1, kv2, kv3))
        # non-negativity
        y = y * (y > 0)
        y_stop = y[-1, :]
        return y, y_stop

    def _configuration_seq(self, cycle: list, nb_of_cycles: int,
                           sd_q: float, sd_kv: float, sd_dur: float, leaky: bool):
        seq = list()
        for i in range(nb_of_cycles):
            for state in cycle:
                if type(state) is str:
                    seq.append(self.state_df.loc[state])
                else:
                    seq.append((self.state_df.iloc[state, :]))
        seq_df = pd.concat(seq, axis=1).T.astype({"duration": int})
        seq_df_noise = seq_df.copy()
        seq_len = seq_df.shape[0]
        if sd_q is not None:
            q_noise = np.random.normal(0, sd_q, 2 * seq_len)
            seq_df_noise["q1"] = seq_df["q1"] + q_noise[:seq_len]
            seq_df_noise["q3"] = seq_df["q3"] + q_noise[seq_len:]
            if not leaky:
                seq_df_noise["q1"].where(seq_df["q1"] > 0, other=0, inplace=True)  # no leaky inflow
                seq_df_noise["q3"].where(seq_df["q3"] > 0, other=0, inplace=True)
        if sd_kv is not None:
            kv_noise = np.random.normal(0, sd_kv, 3 * seq_len)
            seq_df_noise["kv1"] = seq_df["kv1"] + kv_noise[:seq_len]
            seq_df_noise["kv2"] = seq_df["kv2"] + kv_noise[seq_len:2*seq_len]
            seq_df_noise["kv3"] = seq_df["kv3"] + kv_noise[2*seq_len:]
            if not leaky:
                seq_df_noise["kv1"].where(seq_df["kv1"] > 0, other=0, inplace=True)  # no leaky valve
                seq_df_noise["kv2"].where(seq_df["kv2"] > 0, other=0, inplace=True)
                seq_df_noise["kv3"].where(seq_df["kv3"] > 0, other=0, inplace=True)
        if sd_dur is not None:
            dur_noise = np.random.normal(0, sd_dur, seq_len)
            seq_df_noise["duration"] = round(seq_df["duration"] + dur_noise).astype(int)
        seq_df_noise = seq_df_noise.where(seq_df_noise >= 0, 0)  # no negative inflow etc.
        return seq_df, seq_df_noise

    def simulate(self, cycle: list, nb_of_cycles: int = 10,
                 sd_q: float = None, sd_kv: float = None, sd_dur: float = None, leaky: bool = False,
                 export_path: str = None) -> np.array:
        """Simulates the dynamics in the three-tank system
        Args:
            cycle (list): sequence of states that compose a typical cycle.
                          Either list of integers or list of state names.
            nb_of_cycles (int): number of successive cycles to simulate
            sd_q (float): if set, white noise with this standard deviation is added to the inflow
            sd_kv (float): if set, white noise with this standard deviation is added to the valve coefficients
            sd_dur (float): if set, white noise with this standard deviation is added to the duration
            leaky (bool): if true, add noise on closed valves and stopped inflow
            export_path (str): if set, save simulation datagen at export path
        """
        seq_denoised, seq = self._configuration_seq(cycle, nb_of_cycles, sd_q, sd_kv, sd_dur, leaky)

        y_ls = []
        y_stop = self.tank_levels
        for config in seq.itertuples():
            y, y_stop = self._compute_section(duration=config.duration, x0=y_stop,
                                              kv1=config.kv1, kv2=config.kv2, kv3=config.kv3,
                                              q1=config.q1, q3=config.q3)
            y_ls.append(y)
        y_out = np.concatenate(y_ls)
        if export_path is not None:
            y_df = pd.DataFrame(y_out, columns=['h1', 'h2', 'h3'])
            y_df.to_csv(export_path, index=False)
            # seq_denoised.to_csv(export_path[:-4] + "_config.csv", index=False)
            # seq.to_csv(export_path[:-4] + "_config_with_noise.csv")
        return y_out


def run():
    system = ThreeTankSimulation(tank_1_lvl=10, tank_2_lvl=20, tank_3_lvl=10)
    system.add_state(q1=0.2, q3=0.5, kv1=0.05, kv2=0.1, kv3=0,   duration=50,   name="fill")
    system.add_state(q1=0,   q3=0,   kv1=0,    kv2=0,   kv3=0,   duration=50,   name="rest")
    system.add_state(q1=0,   q3=0,   kv1=0.05, kv2=0.1, kv3=0.1, duration=50,   name="mix")
    system.add_state(q1=0.1, q3=0.1, kv1=1,    kv2=1,   kv3=0.1, duration=1000, name="empty")

    y = system.simulate(cycle=["fill", "rest", "mix"] * 3 + ["empty"],
                        nb_of_cycles=100,
                        export_path="../../data/solution_4_dataset.csv")
    plot_simulation(y)


if __name__ == "__main__":
    run()
