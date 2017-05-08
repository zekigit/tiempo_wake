import mne
from mne.time_frequency import tfr_morlet
import numpy as np
import os.path as op
import matplotlib.pyplot as plt
from etg_ieeg_info import dat_path, files, n_jobs, log_path, marks
from eeg_etg_fxs import read_log_file, check_events, durations_from_log_ieeg, create_events, add_event_condition
import pandas as pd
from mne.stats import permutation_cluster_1samp_test

pd.set_option('display.expand_frame_repr', False)

subj = 'P14'
cond = 1


file = op.join(dat_path, 'set', files[subj][cond] + '.set')
raw = mne.io.read_raw_eeglab(file, preload=True)


notchs = np.arange(50, 151, 50)

raw.filter(l_freq=1, h_freq=140, n_jobs=n_jobs)
raw.notch_filter(freqs=notchs, n_jobs=n_jobs)

# raw.plot_psd()
# raw.plot(n_channels=raw.info['nchan'], scalings={'eeg': 50e-6})

log_filename = log_path + 'ETG7a_{}.mat' .format(subj)
log = read_log_file(log_filename)

events = mne.find_events(raw)
events = check_events(events)
durations = durations_from_log_ieeg(log)
new_events = create_events(events, durations, raw.info['sfreq'])
raw.add_events(new_events)
events_updated = mne.find_events(raw, shortest_event=1)
events_updated = check_events(events_updated)
events_updated, log = add_event_condition(events_updated, log)

# Epoch
t_min = -0.7
t_max = 0.7
baseline = (None, None)
event_id = marks
epoch = mne.Epochs(raw, events_updated, event_id=event_id,
                   tmin=t_min, tmax=t_max, baseline=baseline,
                   preload=True)

exp_ready = epoch['exp_short_smaller', 'exp_long_smaller', 'exp_short_bigger', 'exp_long_bigger']
exp_evoked = exp_ready.average()

exp_lo_sma = epoch['exp_long_smaller']
exp_lo_big = epoch['exp_long_bigger']
exp_sho_sma = epoch['exp_short_smaller']
exp_sho_big = epoch['exp_short_bigger']

exp_lo_sma_evok = exp_lo_sma.average()
exp_lo_big_evok = exp_lo_big.average()
exp_sho_sma_evok = exp_sho_sma.average()
exp_sho_big_evok = exp_sho_big.average()

# mne.viz.plot_snr_estimate(exp_lo_sma_evok)

# fig_erp, ax = plt.subplots(2,2)
# exp_lo_sma_evok.plot(picks=[27], axes=ax[0, 0])
# exp_lo_big_evok.plot(picks=[27], axes=ax[0, 1])
# exp_sho_sma_evok.plot(picks=[27], axes=ax[1, 0])
# exp_sho_big_evok.plot(picks=[27], axes=ax[1, 1])

picks = [27, 95, 103]


# Time-Frequency
freqs = np.arange(10, 121, 1)  # define frequencies of interest
n_cycles = 5.  # different number of cycle per frequency
power = tfr_morlet(exp_lo_big, freqs=freqs, n_cycles=n_cycles, use_fft=True, average=False,
                   return_itc=False, decim=3, n_jobs=n_jobs, picks=picks)

for c in range(len(picks)):
    power.average().plot(picks=[c], baseline=(-0.7, -0.4), mode='logratio', fmin=1, fmax=120, tmin=-0.6, tmax=0.6, vmax=0.5, vmin=-0.5)
    plt.title('Channel {}' .format(exp_lo_big.info['ch_names'][picks[c]]))


power_bs = power.copy()
power_bs.apply_baseline(mode='logratio', baseline=(-0.7, -0))

# power_bs.apply_baseline(mode='logratio', baseline=(-0.7, -0.4)).average().plot([1], fmin=1, fmax=120, tmin=-0.6, tmax=0.6, vmax=0.5, vmin=-0.5)
# power.average().apply_baseline(mode='logratio', baseline=(-0.7, -0.4)).plot([1], fmin=1, fmax=120, tmin=-0.6, tmax=0.6, vmax=0.5, vmin=-0.5)


power_bs.average().plot([1])

power_bs.crop(-0.4, None)

epochs_power = power_bs.data[:, 1, :, :]
threshold = None
T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(epochs_power, n_permutations=100, threshold=threshold, tail=0)

times = 1e3 * power_bs.times
T_obs_plot = np.nan * np.ones_like(T_obs)
for c, p_val in zip(clusters, cluster_p_values):
    if p_val <= 0.05:
        T_obs_plot[c] = T_obs[c]

vmax = np.max(np.abs(T_obs))
vmin = -vmax

plt.subplot(1, 1, 1)
plt.imshow(T_obs, cmap=plt.cm.gray,
           extent=[times[0], times[-1], freqs[0], freqs[-1]],
           aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
plt.imshow(T_obs_plot, cmap=plt.cm.RdBu_r,
           extent=[times[0], times[-1], freqs[0], freqs[-1]],
           aspect='auto', origin='lower', vmin=vmin, vmax=vmax)

plt.colorbar()
plt.xlabel('Time (ms)')
plt.ylabel('Frequency (Hz)')
#plt.title('Induced power (%s)' % ch_name)

