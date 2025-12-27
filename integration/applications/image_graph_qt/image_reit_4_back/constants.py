import  os

# Constants
DEFAULT_JOB_NAME = "rating_competition"
INITIAL_RATING_MIN = 1
INITIAL_RATING_MAX = 10
DEFAULT_RATING = 5.0
DEFAULT_MU = 25.0
MODEL_SIGMA = 8.333
ACESSIBLE_QUALITY = 0.1
AUTO_LOAD_ROUND = 100000
AUTO_WIN_PAIRS = True
ANTI_STACK_COUNT = 3
ADD_RAND_COUNT = 100

# Cache configuration
CACHE_DIR = r"D:\data\rc"
PAIRS_CACHE_PATH = os.path.join(CACHE_DIR, "pairs_cache")
TRUESKILL_CACHE_PATH = os.path.join(CACHE_DIR, "trueskill_cache")
ITEMS_CACHE_PATH = os.path.join(CACHE_DIR, "items_cache")