"""We add this  file so that we can attach island files, not attached to main

to be a dependency onto main. The island files are added as a dependency to this
file and then this file is added as a dependency for main.

This is necessary for config files to be able to call island files. This is
because
files are only visible to be dependencies for configs when they are called as
a dependency somewhere in the graph that extends from main.py
"""

from google3.experimental.largesensormodels.scenic.trainers.masking.masker_config import MaskStrategy_Config, Masker_Config
import google3.experimental.largesensormodels.scenic.utils.config_constants
import google3.experimental.largesensormodels.scenic.utils.predefined_configs
