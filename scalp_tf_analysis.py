import mne
import os.path as op
import numpy as np
from mne.time_frequency import tfr_morlet
from mne.stats import permutation_cluster_1samp_test
from etg_scalp_info import study_path, subjects, n_jobs, reject
import matplotlib.pyplot as plt

bs_min = -0.7
bs_max = -0.4

# LOAD
exp_lon = list()
exp_sho = list()

for subj in subjects:
    epochs_file = op.join(study_path, 'epo', '{}_exp-epo.fif' .format(subj))
    epochs = mne.read_epochs(epochs_file, preload=True)
    if (len(epochs['exp_sup_lon']) > 30) and (len(epochs['exp_sup_sho']) > 10):
        epochs.apply_baseline(baseline=(bs_min, bs_max))
        exp_lon.append(epochs['exp_sup_lon'].average())
        exp_sho.append(epochs['exp_sup_sho'].average())


lon_epo = mne.combine_evoked(exp_lon, weights='nave')
sho_epo = mne.combine_evoked(exp_sho, weights='nave')
all_epo = mne.combine_evoked([lon_epo, sho_epo], weights='nave')

# Time Frequency
freqs = np.arange(4, 41, 0.25)  # define frequencies of interest
n_cycles = 4.  # different number of cycle per frequency

power = tfr_morlet(all_epo, freqs=freqs, n_cycles=n_cycles, use_fft=True, average=True,
                   return_itc=False, n_jobs=n_jobs)

power.plot_topo(baseline=(bs_min, bs_max), mode='percent', tmin=-0.4, tmax=0.4, vmin=-10, vmax=10,
                fmin=4, fmax=20)


# Stats - Vs. Baseline
chan = 'B2'
conds = ['Shorter', 'Longer']
ch = mne.pick_channels(power.info['ch_names'], [chan])
pows_sho_log = list()
pows_lon_log = list()
pows_sho = list()
pows_lon = list()
for ix, cond in enumerate([exp_sho, exp_lon]):
    pow_fig, axes = plt.subplots(nrows=5, ncols=4, sharey=True, sharex=True, figsize=(8, 10))
    for s, ax in zip(cond, axes.flat):
        power_subj = tfr_morlet(s, freqs=freqs, n_cycles=n_cycles, use_fft=True, average=True,
                                return_itc=False, n_jobs=n_jobs, picks=ch)
        power_subj.plot(baseline=(bs_min, bs_max), mode='percent', tmin=-0.4, tmax=0.4, vmin=-10, vmax=10,
                        fmin=4, fmax=40, picks=[0], axes=ax, colorbar=False)
        ax.vlines(0, ymin=0, ymax=40, linestyles='--')
        ax.xaxis.set_ticks(np.arange(-400, 400, 400))

        if ix == 0:
            pows_sho.append(power_subj.data)
        else:
            pows_lon.append(power_subj.data)

        power_subj.apply_baseline(mode='logratio', baseline=(bs_min, bs_max))
        power_subj.crop(tmin=-0.4, tmax=0.4)
        if ix == 0:
            pows_sho_log.append(power_subj.data)
        else:
            pows_lon_log.append(power_subj.data)
    pow_fig.suptitle('{} - Channel: {}' .format(conds[ix], chan))

group_dat_sho = np.vstack(pows_sho_log)
group_dat_lon = np.vstack(pows_lon_log)

threshold = None
T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(group_dat_lon, n_permutations=100,
                                                                       threshold=threshold, tail=0)

times = 1e3 * power_subj.times
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

# Stats - Between Conditions

