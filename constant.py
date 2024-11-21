import enum
import os


class DatasetType(enum.Enum):
    BITCOIN_ALPHA = 0,
    BITCOIN_OTC = 1,
    WIKI_GL = 2,
    WIKI_EO = 3,
    DIGG = 4,


BETA = 1
SEED = 2024
VAL_RATE = 0.1
TEST_RATE = 0.2

DATA_DIR_PATH = os.path.join(os.path.abspath("."), 'data')

# bitcoin alpha setting
BITCOIN_ALPHA_PATH = os.path.join(DATA_DIR_PATH, 'bitcoin_alpha')
BITCOIN_ALPHA_NAME = 'bitcoin_alpha.mat'
BITCOIN_ALPHA_TS = 60
BITCOIN_ALPHA_LR = 0.005
BITCOIN_ALPHA_LAM = 1e-5
BITCOIN_ALPHA_TAU = 0.2

# bitcoin otc setting
BITCOIN_OTC_PATH = os.path.join(DATA_DIR_PATH, 'bitcoin_otc')
BITCOIN_OTC_NAME = 'bitcoin_otc.mat'
BITCOIN_OTC_TS = 60
BITCOIN_OTC_LR = 0.005
BITCOIN_OTC_LAM = 1e-2
BITCOIN_OTC_TAU = 0.1

# WIKI setting
WIKI_GL_PATH = os.path.join(DATA_DIR_PATH, 'wiki_gl')
WIKI_GL_NAME = 'wiki_gl.mat'
WIKI_GL_TS = 60
WIKI_GL_LR = 0.01
WIKI_GL_LAM = 5e-2
WIKI_GL_TAU = 0.3

# WIKI_EO setting
WIKI_EO_PATH = os.path.join(DATA_DIR_PATH, 'wiki_eo')
WIKI_EO_NAME = 'wiki_eo.mat'
WIKI_EO_TS = 60
WIKI_EO_LR = 0.01
WIKI_EO_LAM = 1e-2
WIKI_EO_TAU = 0.7

# DIGG setting
DIGG_PATH = os.path.join(DATA_DIR_PATH, 'digg')
DIGG_NAME = 'digg.mat'
DIGG_TS = 50
DIGG_LR = 0.0001
DIGG_LAM = 1e-05
DIGG_TAU = 0.8
