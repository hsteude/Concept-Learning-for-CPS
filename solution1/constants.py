# data generation related
NUMBER_TIMESTEPS = 50
NUMBER_OF_SAMPLES = 10_000
T_MAX = 10
A = 10
G = 9.81

INITIAL_STATES = [10, 95, 33]
Q1_MIN = 0
Q1_MAX = 10
Q3_MIN = 0
Q3_MAX = 10
KV12_MIN = .1
KV12_MAX = 1
KV23_MIN = .1
KV23_MAX = 1
SCALING_CONST = 100

STATE_COL_NAMES = ['h1', 'h2', 'h3']
TIME_COL_NAME = 'time'
Q1_COL_NAME = 'q1'
Q3_COL_NAME = 'q3'
KV12_COL_NAME = 'kv12'
KV23_COL_NAME = 'kv23'
LABEL_COLS = [Q1_COL_NAME, Q3_COL_NAME, KV12_COL_NAME, KV23_COL_NAME]
UID_SAMPLE_COL_NAME = 'uid_ts_sample'

# Path
DATA_PATH = 'data/solution_1_dataset.parquet'
SEED = 12354

FIGURE_PATH_TIME_SERIES = 'figures/solution_1_example_ts.pdf'
FIGURE_PATH_RESULTS = 'figures/solution_1_results.pdf'
LOGDIR = "logs"
MODEL_NAME = "SEQ2SEQ"


