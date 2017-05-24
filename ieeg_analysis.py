import mne
from mne.time_frequency import tfr_morlet
import numpy as np
import os.path as op
import matplotlib.pyplot as plt
from etg_ieeg_info import dat_path, files, n_jobs, log_path, marks, marks_exp, marks_p
from eeg_etg_fxs import read_log_file, check_events, durations_from_log_ieeg, create_events, add_event_condition, make_exp_baseline, \
    add_event_tr_id
import pandas as pd
from mne.stats import permutation_cluster_1samp_test, permutation_cluster_test

pd.set_option('display.expand_frame_repr', False)
plt.style.use('ggplot')

subj = 'P14'
conds = ['Longer', 'Shorter']

# Baseline Parameters
bs_min = -0.9  # Pre-s2 baseline
bs_max = -0.7

file = op.join(dat_path, 'set', files[subj][1] + '.set')
raw = mne.io.read_raw_eeglab(file, preload=True)

# Filters
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
events_tr_id = add_event_tr_id(events_updated)

# Epoch
t_min = -0.7
t_max = 0.7
reject = {'eeg': 150e6}

event_id = marks
epoch = mne.Epochs(raw, events_updated, event_id=marks_exp,
                   tmin=t_min, tmax=t_max, baseline=None,
                   preload=True)

pre_trial = mne.Epochs(raw, events_updated, event_id=marks_p, tmin=-0.25, tmax=0, baseline=None, reject=reject, preload=True)

exp_epochs, exp_ids = make_exp_baseline(epoch, pre_trial, events_tr_id, log, marks_exp)
exp_epochs.resample(256, n_jobs=n_jobs)

exp_sup_lon = exp_epochs['exp_sup_lon']
exp_sup_sho = exp_epochs['exp_sup_sho']
exp_sup_lon.apply_baseline(baseline=(bs_min, bs_max))
exp_sup_sho.apply_baseline(baseline=(bs_min, bs_max))

picks = [27, 95, 103]

# Time-Frequency
freqs = np.arange(4, 121, 0.1)  # define frequencies of interest
n_cycles = 4.  # different number of cycle per frequency
pow_list = list()
for ix_c, c in enumerate([exp_sup_lon, exp_sup_sho]):
    power = tfr_morlet(c, freqs=freqs, n_cycles=n_cycles, use_fft=True, average=False,
                       return_itc=False, n_jobs=n_jobs, picks=picks)
    pow_list.append(power.copy())
    for ix_ch, ch in enumerate(picks):
        if ch == picks[0]:
            vmax = 7
        else:
            vmax = 2

        power.average().plot(picks=[ix_ch], baseline=(-0.9, -0.7), mode='zscore', fmin=5, fmax=120, tmin=-0.4, tmax=0.4,
                             vmax=vmax, vmin=-vmax)
        plt.title('Channel {} - Cond: {}' .format(exp_sup_lon.info['ch_names'][ch], conds[ix_c]))

fmin = 8
fmax = 12
fq_mask = (freqs >= fmin) & (freqs <= fmax)

pows_corr = [p.copy().apply_baseline(mode='ratio', baseline=(-0.9, -0.7)) for p in pow_list]

# plt.plot(power.times, np.mean(pows_corr[0].data[:, 0, :, :], axis=2))
# power_bs.apply_baseline(mode='logratio', baseline=(-0.7, -0.4)).average().plot([1], fmin=1, fmax=120, tmin=-0.6, tmax=0.6, vmax=0.5, vmin=-0.5)
# power.average().apply_baseline(mode='logratio', baseline=(-0.7, -0.4)).plot([1], fmin=1, fmax=120, tmin=-0.6, tmax=0.6, vmax=0.5, vmin=-0.5)

power_lon = pows_corr[0].crop(-0.4, 0.4)
power_sho = pows_corr[1].crop(-0.4, 0.4)
# power_bs.average().plot([1])

power_c1 = power_lon.data[:, 1, :, :]
power_c2 = power_sho.data[:, 1, :, :]

# threshold = None
# T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(power_c1, n_permutations=100, threshold=threshold, tail=0)

threshold = None
T_obs, clusters, cluster_p_values, H0 = \
    permutation_cluster_test([power_c1, power_c2],
                             n_permutations=100, threshold=threshold, tail=0)


times = 1e3 * power_lon.times
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

