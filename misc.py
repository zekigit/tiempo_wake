import mne
from etg_scalp_info import data_path
import numpy as np
import matplotlib.pyplot as plt



suj = 5
ses = 1

archivo = data_path + 'etg_su{}_se{}.bdf' .format(str(suj), str(ses))
raw = mne.io.read_raw_edf(archivo, preload=True)


events = mne.find_events(raw)
mne.viz.plot_events(events)


p = np.arange(0.25, 2, 0.25)
d = 0.2

w = (2*np.abs(1-(p+d))) / (1+(p+d))

fig, axes = plt.subplots(1, len(d))

# subject, window, tau, condition


