# __init__.py
from .sint_canonization import compute_shifts_dictionary
from .ibp_generation import pre_build_ibp_relations
from .ibp_system_handling import *
from .global_precomputed_functions import *
from .ibp_sols_handling import *
from .reduction import *
from .seed_generation import *
from .sint_utils import *
from .in_out_utils import *
# from . import main

import logging
DETAILED_INFO_LEVEL = 15
logging.addLevelName(DETAILED_INFO_LEVEL, "DETAILED_INFO")
logging.DETAILED_INFO = DETAILED_INFO_LEVEL

def detailed_info(msg, *args, **kwargs):
    logging.log(DETAILED_INFO_LEVEL, msg, *args, **kwargs)

logging.detailed_info = detailed_info