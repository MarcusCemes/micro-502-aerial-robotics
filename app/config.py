from typing import Final

# == Crazyflie configuration == #

CACHE_DIR: Final = "./cache"
LOG_PERIOD_MS: Final = 50
URI: Final = "radio://0/10/2M/E7E7E7E711"

INITIAL_POSITION: Final = (3.0, 1.0)

# == Environment == #

BOX_LIMIT: Final = 0.5
PAD_HEIGHT: Final = 0.1


# == Flight == #

CRUISING_ALTITUDE: Final = 0.5

ALTITUDE_ERROR: Final = 0.05
POSITION_ERROR: Final = 0.2

PAD_THRESHOLD: Final = 0.1
MAX_SLOPE = 0.02

VELOCITY_LIMIT: Final = 0.2
VERTICAL_VELOCITY_LIMIT: Final = 0.1
ANGULAR_VELOCITY_LIMIT_DEG: Final = 90.0
ANGULAR_SCAN_VELOCITY_DEG: Final = 45.0


# == Navigation == #

MAP_SIZE: Final = (5.0, 3.0)
OPTIMISE_PATH: Final = True
RANGE_THRESHOLD: Final = 2.0
MAP_PX_PER_M: Final = 25

# == Miscellaneous == #

DEBUG_FILES: Final = True
SERVER_PORT = 8080
