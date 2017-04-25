import mne
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from etg_scalp_info import bad_channs
from eeg_etg_fxs import create_events, read_log_file, durations_from_log, check_events
from mne.time_frequency import tfr_morlet

# Subject
subj = '1'
ses = '1'
arch = 'etg_scalp_su{}_se{}' .format(subj, ses)
do_ica = False

# Names
path = '/Volumes/FAT/Time/ETG_scalp/'
file = arch + '_prepro'
log_filename = path + 'logs/' + arch + '_log.mat'

# Load
raw = mne.io.read_raw_eeglab(path + 'set/' + file + '.set', preload=True)
fs = raw.info['sfreq']

# Montage
montage = mne.channels.read_montage('biosemi128')
raw.set_montage(montage)
# mne.viz.plot_sensors(raw.info)

# Bad channels
bads = bad_channs[subj]
raw.info['bads'] = bads
picks_eeg = mne.pick_types(raw.info, eeg=True, meg=False, eog=False, exclude='bads')
# raw.plot(n_channels=128)

# Log
log = read_log_file(log_filename)

# Events
events = mne.find_events(raw)
events = check_events(events)
durations = durations_from_log(log)
new_events = create_events(events, durations, fs)
raw.add_events(new_events)
events_updated = mne.find_events(raw, shortest_event=1)
# mne.viz.plot_events(events_updated)

# Reject
reject = {'eeg': 120e-6}

if do_ica:
    n_components = 20
    method = 'fastica'
    decim = 3
    ica = mne.preprocessing.ICA(n_components=n_components, method=method)
    ica.fit(raw, picks=picks_eeg, decim=decim, reject=reject)
    print(ica)
    ica.plot_components()
    input('Press a key to continue')

# Interpolate
raw.interpolate_bads()

# ERP
t_min = -0.4
t_max = 1
baseline = (None, 0)
event_id = dict(s1=1, s2=2, exp=3)
epoch = mne.Epochs(raw, events_updated, event_id=event_id,
                   tmin=t_min, tmax=t_max,
                   baseline=baseline, reject=reject)
epoch.drop_bad()
epoch.plot_drop_log()
# epoch.average().plot()
evoked_s1 = epoch['s1'].average()
evoked_s2 = epoch['s2'].average()
evoked_exp = epoch['exp'].average()
evoked_s1.plot(spatial_colors=True)
evoked_s2.plot(spatial_colors=True)
evoked_exp.plot(spatial_colors=True)

evoked_exp.plot_topomap(times=[-0.2, -0.11, -0.015, 0, 0.015, 0.11, 0.24])

# # Time Frequency
freqs = np.arange(6, 30, 3)  # define frequencies of interest
n_cycles = freqs / 2.  # different number of cycle per frequency
power, itc = tfr_morlet(epoch['exp'], freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, decim=3, n_jobs=1)

power.plot_topo(baseline=(-0.5, 0), mode='zscore', title='Average power')
power.plot([41], baseline=(-0.5, 0), mode='zscore')
#
# # Raster Plot
# layout = mne.find_layout(epoch.info, 'eeg')
# mne.viz.plot_topo_image_epochs(epoch['exp'], layout, sigma=0.5, vmin=-10, vmax=10,
#                                colorbar=True)

