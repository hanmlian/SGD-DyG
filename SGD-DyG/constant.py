import enum
import os


class DatasetType(enum.Enum):
    BITCOIN_ALPHA = 0,
    BITCOIN_OTC = 1,
    WIKI_GL = 2,
    WIKI_EO = 3,
    DIGG = 4,
    PPIN = 5,
    DBLP = 6,
    LAST_FM = 7,


BETA = 1
SEED = 2024
VAL_RATE = 0.1
TEST_RATE = 0.2

DATA_DIR_PATH = os.path.join(os.path.abspath(".."), '../data')

# bitcoin alpha setting
BITCOIN_ALPHA_PATH = os.path.join(DATA_DIR_PATH, 'bitcoin_alpha')
BITCOIN_ALPHA_NAME = 'bitcoin_alpha.mat'
BITCOIN_ALPHA_TS = 60

# bitcoin otc setting
BITCOIN_OTC_PATH = os.path.join(DATA_DIR_PATH, 'bitcoin_otc')
BITCOIN_OTC_NAME = 'bitcoin_otc.mat'
BITCOIN_OTC_TS = 60

# WIKI setting
WIKI_GL_PATH = os.path.join(DATA_DIR_PATH, 'wiki_gl')
WIKI_GL_NAME = 'wiki_gl.mat'
WIKI_GL_TS = 60

# WIKI_EO setting
WIKI_EO_PATH = os.path.join(DATA_DIR_PATH, 'wiki_eo')
WIKI_EO_NAME = 'wiki_eo.mat'
WIKI_EO_TS = 60

# DIGG setting
DIGG_PATH = os.path.join(DATA_DIR_PATH, 'digg')
DIGG_NAME = 'digg.mat'
DIGG_TS = 50

# PPIN setting
PPIN_PATH = os.path.join(DATA_DIR_PATH, 'ppin')
PPIN_NAME = 'ppin.mat'
PPIN_TS = 36

# DBLP setting
DBLP_PATH = os.path.join(DATA_DIR_PATH, 'dblp')
DBLP_NAME = 'dblp.mat'
DBLP_TS = 45

# LAST_FM setting
LAST_FM_PATH = os.path.join(DATA_DIR_PATH, 'last_fm')
LAST_FM_NAME = 'last_fm.mat'
LAST_FM_TS = 53
