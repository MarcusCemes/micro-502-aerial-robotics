from typing import Final

# == Crazyflie configuration == #

CACHE_DIR: Final = "./cache"
RANGE_LOG_PERIOD_MS: Final = 50
STAB_LOG_PERIOD_MS: Final = 50
URI: Final = "radio://0/10/2M/E7E7E7E711"

INITIAL_POSITION: Final = (0.75, 1.5)


# == Environment == #

BOX_LIMIT: Final = 0.5
PAD_HEIGHT: Final = 0.1
PAD_WIDTH: Final = 0.37
LINE_TARGET_SEARCH: Final = 3.5


# == Flight == #

CRUISING_ALTITUDE: Final = 0.4

ALTITUDE_ERROR: Final = 0.15
POSITION_ERROR: Final = 0.2
POSITION_ERROR_PAD: Final = 0.1
HOME_PAD_ERROR: Final = 0.6

PAD_THRESHOLD: Final = 0.1
MAX_SLOPE: Final = 0.01
MIN_DIFF: Final = 45

VELOCITY_LIMIT: Final = 0.3
VELOCITY_LIMIT_SLOW: Final = 0.2
VELOCITY_LIMIT_FAST: Final = 0.4
VERTICAL_VELOCITY_LIMIT: Final = 0.1
ANGULAR_VELOCITY_LIMIT_DEG: Final = 70.0
ANGULAR_SCAN_VELOCITY_DEG: Final = 55.0
PROBABILITY_THRESHOLD: Final = 110


# == Navigation == #

MAP_SIZE: Final = (5.0, 3.0)
OPTIMISE_PATH: Final = True
RANGE_THRESHOLD: Final = 2.0
MAP_PX_PER_M: Final = 25
SEARCHING_PX_PER_M: Final = 40
LATERAL_MOVEMENT: Final = 0.5


# == Miscellaneous == #

DEBUG_FILES: Final = False
SERVER_ENABLED: Final = False
SERVER_PORT: Final = 8080
