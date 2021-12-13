from sklearn.preprocessing import PolynomialFeatures
import numpy as np

# data generation related
NUMBER_TIMESTEPS = 50
NUMBER_INITIAL_STATES = 1000
T_MAX = 20
Q1 = 0
Q3 = 0
A = 10
G = 9.81
LATENT_DIM = 3

POLY_ORDER = 5
PICTURE_SIZE = 100
H1_X_RANGE = (0, 32)
H2_X_RANGE = (34, 66)
H3_X_RANGE = (68, 100)
BLUR_FULTER_SIGMA = 2

INITIAL_LEVEL_MIN = 10
INITIAL_LEVEL_MAX = 90

Z_COL_NAMES = ['h0', 'h1', 'h2']
Z_DOT_COL_NAMES = ['h0_dot', 'h1_dot', 'h2_dot']
UID_INITIAL_STATE_COL_NAME = 'uid_initial_state'
TIME_COL_NAME = 'time'

#path
poly = PolynomialFeatures(POLY_ORDER)
X_COL_NAMES = [x.replace('x', 'h') for x in poly.fit(np.identity(LATENT_DIM)).get_feature_names_out()]
XDOT_COL_NAMES = [f'df_dt_{i}' for i in X_COL_NAMES]


DATA_PATH = 'data/solution_3_dataset.parquet'
FIGURE_PATH_RESULTS = 'figures/solution_3_results.pdf'
MODEL_NAME = "AE-SINDY"
LOGDIR = "logs"
SEED = 12354
