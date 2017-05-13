import mne
import os.path as op
import numpy as np
from mne.time_frequency import tfr_morlet
from etg_scalp_info import study_path, subjects, n_jobs, ROI_d, ROI_i, rois
from eeg_etg_fxs import permutation_t_test
import matplotlib.pyplot as plt
import pandas as pd
from natsort import natsorted
from statsmodels.sandbox.stats.multicomp import multipletests


plt.style.use('ggplot')

# Baseline Parameters
bs_min = -0.95  # Pre-s2 baseline
bs_max = -0.7

# Time Frequency Parameters
freqs = np.arange(4, 41, 0.1)  # frequencies of interest
n_cycles = 4.

# Comparison Parameters
conds = ['Longer', 'Shorter']
fmin = 8
fmax = 12
tmin = -0.1
tmax = 0.1


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
# tr_info.plot.bar()

lon_evo = mne.combine_evoked(exp_lon, weights='nave')
sho_evo = mne.combine_evoked(exp_sho, weights='nave')
all_evo = mne.combine_evoked([lon_evo, sho_evo], weights='equal')

# TF Combined Conditions
power = tfr_morlet(all_evo, freqs=freqs, n_cycles=n_cycles, use_fft=True, average=True,
                   return_itc=False, n_jobs=n_jobs)

power.plot_topo(baseline=(bs_min, bs_max), mode='zscore', tmin=-0.4, tmax=0.4, vmin=-5, vmax=5,
                fmin=4, fmax=40, yscale='linear')


# Calculate Power per Subject and ROI
pows_lon = dict(a=list(), b=list(), c=list(), d=list(), e=list(), f=list())
pows_sho = dict(a=list(), b=list(), c=list(), d=list(), e=list(), f=list())

for ix, cond in enumerate([exp_lon, exp_sho]):
    # pow_fig, axes = plt.subplots(nrows=6, ncols=16, sharey=True, sharex=True, figsize=(20, 10))
    for ix_s, s in enumerate(cond):
        power_subj = tfr_morlet(s, freqs=freqs, n_cycles=n_cycles, use_fft=True, average=True,
                                return_itc=False, n_jobs=n_jobs)

        for ix_r, r in enumerate(sorted(rois)):
            roi = rois[r]
            pow_subj_c = power_subj.copy()
            roi_pow = pow_subj_c.pick_channels(roi)
            roi_pow.data = np.mean(roi_pow.data, 0, keepdims=True)
            # roi_pow.plot(baseline=(bs_min, bs_max), mode='zscore', tmin=-0.4, tmax=0.4, vmin=-10, vmax=10,
            #              fmin=4, fmax=40, picks=[0], axes=axes[ix_r, ix_s], colorbar=False)
            # print(r, roi)

            if ix == 0:
                pows_lon[r].append(roi_pow)
            else:
                pows_sho[r].append(roi_pow)

# Masks for Analysis
fq_mask = (freqs >= fmin) & (freqs <= fmax)
t_mask = (power.times >= tmin) & (power.times <= tmax)


# Stats - Between Conditions
outliers = dict(a=list(), b=list(), c=list(), d=list(), e=list(), f=list())
p_vals = list()
stat_fig, axes = plt.subplots(6, 2)

for ix_r, r in enumerate(sorted(rois)):
    pows_lon_z = pows_lon[r].copy()
    pows_sho_z = pows_sho[r].copy()

    pow_lon_win = [np.mean(p.copy().apply_baseline(mode='zscore', baseline=(bs_min, bs_max), verbose=False).crop(tmin=tmin, tmax=tmax).data[:, fq_mask, :])
                   for p in pows_lon_z]
    pow_sho_win = [np.mean(p.copy().apply_baseline(mode='zscore', baseline=(bs_min, bs_max), verbose=False).crop(tmin=tmin, tmax=tmax).data[:, fq_mask, :])
                   for p in pows_sho_z]

    # Detect Outliers
    outl_cond = list()
    for p in [pow_lon_win, pow_sho_win]:
        outl_cond.append([int(ix) for ix, s in enumerate(p) if (s > np.mean(p) + 3 * np.std(p)) or
                         (s < np.mean(p) - 3 * np.std(p))])

    outliers[r] = np.union1d(outl_cond[0], outl_cond[1])

    pow_lon_win_ok = [p for ix, p in enumerate(pow_lon_win) if ix not in outliers[r]]
    pow_sho_win_ok = [p for ix, p in enumerate(pow_sho_win) if ix not in outliers[r]]

    n_perm = 10e3
    t_real, t_list, p_permuted = permutation_t_test(pow_lon_win_ok, pow_sho_win_ok, n_perm)
    pow_dat = pd.DataFrame({'longer': pow_lon_win_ok, 'shorter': pow_sho_win_ok})
    # pow_dat.to_csv(op.join(study_path, 'tables', 'power_data_{}.csv' .format(r)))
    p_vals.append(p_permuted)

    # Fig Stats
    axes[ix_r, 0].violinplot([pow_lon_win_ok, pow_sho_win_ok], showmeans=True)
    axes[ix_r, 0].set_ylabel('z-score')
    axes[ix_r, 0].set_ylim(-5, 30)
    axes[ix_r, 1].hist(t_list, bins=50, facecolor='black')
    axes[ix_r, 1].vlines(t_real, ymin=0, ymax=axes[ix_r, 1].get_ylim()[1], linestyles='--')
    stat_fig.suptitle('{}-{} Hz Power \n {} to {}ms' .format(fmin, fmax, int(tmin*1000), int(tmax*1000)))

print('Outliers found)')
x = [print('ROI: {} - outliers: {} - subjects: {}' .format(r, len(outliers[r]), outliers[r])) for r in sorted(outliers)]
p_vals_corr = multipletests(p_vals, 0.05, 'holm')[1]
print(p_vals_corr)

# for ax in enumerate(axes[:, 1]):
#     ax.set_title('p = {}' .format(p_vals_corr[ix]))

# Fig Power
pow_plot, axes = plt.subplots(6, 2, sharex=True, sharey=True, figsize=(8, 10))
for ix_c, c in enumerate([exp_lon, exp_sho]):
    c_ok = [s for ix, s in enumerate(c) if ix not in outliers[r]]
    c_evo_ok = mne.combine_evoked(c_ok, weights='nave')
    c_power = tfr_morlet(c_evo_ok, freqs=freqs, n_cycles=n_cycles, use_fft=True, average=True,
                         return_itc=False, n_jobs=n_jobs)

    for ix_r, r in enumerate(sorted(rois)):
        roi = rois[r]
        roi_pow = c_power.copy()
        roi_pow.pick_channels(roi)
        roi_pow.data = np.mean(roi_pow.data, 0, keepdims=True)
        roi_pow.plot(baseline=(bs_min, bs_max), mode='zscore', tmin=-0.4, tmax=0.4, vmin=-15, vmax=15,
                     fmin=4, fmax=40, picks=[0], axes=axes[ix_r, ix_c], colorbar=False)
        axes[ix_r, ix_c].vlines(0, ymin=0, ymax=axes[ix_r, ix_c].get_ylim()[1], linestyles='--')
print('permuted p =', p_permuted)


# Subtraction
conds_powers = list()
for c in [lon_evo, sho_evo]:
    c_power = tfr_morlet(c, freqs=freqs, n_cycles=n_cycles, use_fft=True, average=True,
                         return_itc=False, n_jobs=n_jobs)
    conds_powers.append(c_power)

diff_pow = conds_powers[0] - conds_powers[1]
diff_pow.plot_topo(baseline=(bs_min, bs_max), mode='zscore', tmin=-0.4, tmax=0.4, vmin=-5, vmax=5,
                   fmin=4, fmax=40, yscale='linear')

diff_pow_dat = np.mean(power.apply_baseline(mode='zscore', baseline=(bs_min, bs_max)).data[:, fq_mask, :][:, :, t_mask], axis=(1, 2))

diff_rank_ch = [diff_pow.info['ch_names'][ch] for ch in np.argsort(diff_pow_dat)[::-1]]
diff_rank_val = [diff_pow_dat[ch] for ch in np.argsort(diff_pow_dat)[::-1]]

rank_fig, ax = plt.subplots(1)
ax.bar(np.arange(len(diff_rank_val)), diff_rank_val)
ax.set_xticks(np.arange(len(diff_rank_val)))
ax.set_xticklabels(diff_rank_ch, rotation='vertical')
ax.tick_params(labelsize=6)


# Find electrode for comparison
ROI = natsorted(np.union1d(ROI_d, ROI_i))
ch_mask = np.in1d(power.info['ch_names'], ROI)
ROI_chans = mne.pick_channels(power.info['ch_names'], ROI)
ROI_power = np.mean(power.apply_baseline(mode='zscore', baseline=(bs_min, bs_max)).data[ch_mask, :, :][:, fq_mask, :][:, :, t_mask], axis=(1, 2))
chan = ROI[np.argmax(ROI_power)]
ch = mne.pick_channels(power.info['ch_names'], [chan])
print('Selected channel: ', chan)

# ROIS


