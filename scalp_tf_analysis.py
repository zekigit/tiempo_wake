import mne
import os.path as op
import numpy as np
from mne.time_frequency import tfr_morlet, tfr_multitaper, tfr_stockwell
from mne.stats import permutation_cluster_1samp_test
from etg_scalp_info import study_path, subjects, n_jobs, ROI_d, ROI_i
from eeg_etg_fxs import permutation_t_test
import matplotlib.pyplot as plt
import pandas as pd
from natsort import natsorted

plt.style.use('ggplot')

conds = ['Longer', 'Shorter']
bs_min = -0.95
bs_max = -0.7

# LOAD
exp_lon = list()
exp_sho = list()
trials = dict(subj=list(), exp_lon=list(), exp_sho=list())

for subj in subjects:
    epochs_file = op.join(study_path, 'epo', '{}_exp-epo.fif' .format(subj))
    epochs = mne.read_epochs(epochs_file, preload=True)
    if (len(epochs['exp_sup_lon']) > 30) and (len(epochs['exp_sup_sho']) > 20):
        epochs.apply_baseline(baseline=(bs_min, bs_max))
        exp_lon.append(epochs['exp_sup_lon'].average())
        exp_sho.append(epochs['exp_sup_sho'].average())
        trials['subj'].append(subj), trials['exp_lon'].append(len(epochs['exp_sup_lon'])), trials['exp_sho'].append(len(epochs['exp_sup_sho']))

tr_info = pd.DataFrame(trials)
tr_info.plot.bar()

lon_evo = mne.combine_evoked(exp_lon, weights='nave')
sho_evo = mne.combine_evoked(exp_sho, weights='nave')
all_evo = mne.combine_evoked([lon_evo, sho_evo], weights='equal')

# Time Frequency Parameters
freqs = np.arange(4, 41, 0.1)  # frequencies of interest
n_cycles = 4.

# TF Combined Conditions
power = tfr_morlet(all_evo, freqs=freqs, n_cycles=n_cycles, use_fft=True, average=True,
                   return_itc=False, n_jobs=n_jobs)

power.plot_topo(baseline=(bs_min, bs_max), mode='zscore', tmin=-0.4, tmax=0.4, vmin=-5, vmax=5,
                fmin=4, fmax=40, yscale='linear')

# Find electrode for comparison
fmin = 8
fmax = 12
tmin = -0.1
tmax = 0.1
ROI = natsorted(np.union1d(ROI_d, ROI_i))
fq_mask = (freqs >= fmin) & (freqs <= fmax)
t_mask = (power.times >= tmin) & (power.times <= tmax)
ch_mask = np.in1d(power.info['ch_names'], ROI)
ROI_chans = mne.pick_channels(power.info['ch_names'], ROI)
ROI_power = np.mean(power.apply_baseline(mode='zscore', baseline=(bs_min, bs_max)).data[ch_mask, :, :][:, fq_mask, :][:, :, t_mask], axis=(1, 2))
chan = ROI[np.argmax(ROI_power)]
print('Selected channel: ', chan)

# Calculate Power per Subject
ch = mne.pick_channels(power.info['ch_names'], [chan])
pows_sho_z = list()
pows_lon_z = list()
pows_sho = list()
pows_lon = list()

for ix, cond in enumerate([exp_sho, exp_lon]):
    pow_fig, axes = plt.subplots(nrows=4, ncols=4, sharey=True, sharex=True, figsize=(8, 10))
    for s, ax in zip(cond, axes.flat):
        power_subj = tfr_morlet(s, freqs=freqs, n_cycles=n_cycles, use_fft=True, average=True,
                                return_itc=False, n_jobs=n_jobs, picks=ch)
        power_subj.plot(baseline=(bs_min, bs_max), mode='zscore', tmin=-0.4, tmax=0.4, vmin=-20, vmax=20,
                        fmin=4, fmax=40, picks=[0], axes=ax, colorbar=False)
        ax.vlines(0, ymin=0, ymax=40, linestyles='--')
        ax.xaxis.set_ticks(np.arange(-400, 400, 400))

        if ix == 0:
            pows_sho.append(power_subj.data)
        else:
            pows_lon.append(power_subj.data)

        power_subj.apply_baseline(mode='zscore', baseline=(bs_min, bs_max))
        power_subj.crop(tmin=-0.4, tmax=0.4)
        if ix == 0:
            pows_sho_z.append(power_subj)
        else:
            pows_lon_z.append(power_subj)
    pow_fig.suptitle('{} - Channel: {}' .format(conds[ix], chan))


# Stats - Between Conditions
fmin = 8
fmax = 12
tmin = -0.1
tmax = 0.1
fq_mask = (freqs >= fmin) & (freqs <= fmax)
pow_lon_win = [np.mean(p.crop(tmin=tmin, tmax=tmax).data[:, fq_mask, :]) for p in pows_lon_z]
pow_sho_win = [np.mean(p.crop(tmin=tmin, tmax=tmax).data[:, fq_mask, :]) for p in pows_sho_z]

sd_lon = np.std(pow_lon_win)
sd_sho = np.std(pow_sho_win)
outl_lon = np.arange(len(pow_lon_win))[pow_lon_win > 3 * sd_lon]
outl_sho = np.arange(len(pow_sho_win))[pow_sho_win > 3 * sd_sho]
outliers = np.union1d(outl_sho, outl_lon)

pow_lon_win_ok = [np.mean(p.crop(tmin=-0, tmax=0.17).data[:, fq_mask, :]) for ix, p in enumerate(pows_lon_z) if ix not in outliers]
pow_sho_win_ok = [np.mean(p.crop(tmin=0, tmax=0.17).data[:, fq_mask, :]) for ix, p in enumerate(pows_sho_z) if ix not in outliers]

n_perm = 10e3
t_real, t_list, p_permuted = permutation_t_test(pow_lon_win_ok, pow_sho_win_ok, n_perm)
pow_dat = pd.DataFrame({'longer': pow_lon_win_ok, 'shorter': pow_sho_win_ok})
pow_dat.to_csv(op.join(study_path, 'tables', 'power_data_{}.csv' .format(chan)))

# Fig Stats
stat_fig, axes = plt.subplots(1, 2)
axes[0].violinplot([pow_lon_win_ok, pow_sho_win_ok], showmeans=True)
axes[0].set_ylabel('z-score \n (from pre-trial baseline)')
axes[0].set_title('{}-{} Hz Power \n {} to {}ms' .format(fmin, fmax, int(tmin*1000), int(tmax*1000)))
axes[1].hist(t_list, bins=50, facecolor='black')
axes[1].vlines(t_real, ymin=0, ymax=axes[1].get_ylim()[1], linestyles='--')
axes[1].set_title('p = {}' .format(p_permuted))
stat_fig.suptitle('Channel {}' .format(chan))

# Fig Power
pow_plot, axes = plt.subplots(1, 2, sharex=True, sharey=True)
for ix_c, c in enumerate([exp_lon, exp_sho]):
    c_ok = [s for ix, s in enumerate(c) if ix not in outliers]
    c_evo_ok = mne.combine_evoked(c_ok, weights='nave')
    c_power = tfr_morlet(c_evo_ok, freqs=freqs, n_cycles=n_cycles, use_fft=True, average=True,
                         return_itc=False, n_jobs=n_jobs)
    c_power.plot(ch, baseline=(bs_min, bs_max), mode='zscore', tmin=-0.4, tmax=0.4, vmin=-12, vmax=12,
                 fmin=4, fmax=40, axes=axes[ix_c], colorbar=False)
    axes[ix_c].set_title(conds[ix_c])
    axes[ix_c].vlines(0, ymin=0, ymax=axes[ix_c].get_ylim()[1], linestyles='--')
print('permuted p =', p_permuted)


# ----------------------------------
# - Vs. Baseline
# group_dat_sho = np.vstack([p.data for p in pows_sho_z])
# group_dat_lon = np.vstack([p.data for p in pows_lon_z])
#
# threshold = None
# T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(group_dat_lon, n_permutations=100,
#                                                                        threshold=threshold, tail=0)
#
# times = 1e3 * power_subj.times
# T_obs_plot = np.nan * np.ones_like(T_obs)
# for c, p_val in zip(clusters, cluster_p_values):
#     if p_val <= 0.05:
#         T_obs_plot[c] = T_obs[c]
#
# vmax = np.max(np.abs(T_obs))
# vmin = -vmax
#
# plt.subplot(1, 1, 1)
# plt.imshow(T_obs, cmap=plt.cm.gray,
#            extent=[times[0], times[-1], freqs[0], freqs[-1]],
#            aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
# plt.imshow(T_obs_plot, cmap=plt.cm.RdBu_r,
#            extent=[times[0], times[-1], freqs[0], freqs[-1]],
#            aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
#
# plt.colorbar()
# plt.xlabel('Time (ms)')
# plt.ylabel('Frequency (Hz)')
