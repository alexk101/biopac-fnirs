import numpy as np
import matplotlib.pyplot as plt
from itertools import compress
from pathlib import Path

import mne


fnirs_data_folder = Path('.').resolve()
fnirs_cw_amplitude_dir = fnirs_data_folder / 'fS_Exported_2023_01_15.m'
raw_intensity = mne.io.read_raw(fnirs_cw_amplitude_dir, verbose=True)
raw_intensity.load_data()