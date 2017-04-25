import mne
from mne.time_frequency import tfr_morlet
import numpy as np
import os.path as op
from etg_scalp_info import study_path, sujetos, sesiones, n_jobs
import matplotlib.pyplot as plt

exp_short_smaller = list()
exp_short_bigger = list()
exp_long_smaller = list()
exp_long_bigger = list()

