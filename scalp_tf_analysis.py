import mne
import os.path as op
import numpy as np
from mne.time_frequency import tfr_morlet
from mne.channels import read_ch_connectivity
from etg_scalp_info import study_path, subjects, n_jobs, ROI_d, ROI_i, rois
from eeg_etg_fxs import permutation_t_test, set_dif_and_rt_exp
import matplotlib.pyplot as plt
import pandas as pd
from natsort import natsorted
from statsmodels.sandbox.stats.multicomp import multipletests
from mne.stats import permutation_cluster_test, permutation_cluster_1samp_test, permutation_t_test
from pandas.plotting import scatter_matrix


pd.set_option('display.expand_frame_repr', False)
plt.style.use('ggplot')

# Baseline Parameters
bs_min = -0.95  # Pre-s2 baseline
bs_max = -0.75

# Time Frequency Parameters
freqs = np.arange(4, 41, 0.1)  # frequencies of interest
n_cycles = 3.

# Comparison Parameters
conds = ['Longer', 'Shorter']
fmin = 8
fmax = 12
tmin = -0.15
tmax = 0.15


# LOAD
lon_epo = list()
sho_epo = list()
trials = dict(subj=list(), exp_lon=list(), exp_sho=list())

for subj in subjects:
    epochs_file = op.join(study_path, 'epo', '{}_exp-epo.fif' .format(subj))
    epochs = mne.read_epochs(epochs_file, preload=True)
    if (len(epochs['exp_sup_lon']) > 20) and (len(epochs['exp_sup_sho']) > 20):
        epochs.apply_baseline(baseline=(bs_min, bs_max))
        lon_epo.append(epochs['exp_sup_lon'])
        sho_epo.append(epochs['exp_sup_sho'])
        #exp_lon.append(epochs['exp_sup_lon'].average())
        #exp_sho.append(epochs['exp_sup_sho'].average())
        trials['subj'].append(subj), trials['exp_lon'].append(len(epochs['exp_sup_lon'])), trials['exp_sho'].append(len(epochs['exp_sup_sho']))

tr_info = pd.DataFrame(trials)
n_subj = len(tr_info)
# tr_info.plot.bar()

exp_lon = [ep.average() for ep in lon_epo]
exp_sho = [ep.average() for ep in sho_epo]

tr_info['mean'] = tr_info[['exp_lon', 'exp_sho']].mean(axis=1)
tr_info['std'] = tr_info[['exp_lon', 'exp_sho']].std(axis=1)
tr_info['sum'] = tr_info[['exp_lon', 'exp_sho']].sum(axis=1)
print('avg nr epochs: {} - std: {}' .format(tr_info['sum'].mean(), tr_info['sum'].std()))

lon_evo = mne.combine_evoked(exp_lon, weights='nave')
sho_evo = mne.combine_evoked(exp_sho, weights='nave')
all_evo = mne.combine_evoked([lon_evo, sho_evo], weights='equal')

# TF Combined Conditions
power = tfr_morlet(all_evo, freqs=freqs, n_cycles=n_cycles, use_fft=True, average=True,
                   return_itc=False, n_jobs=n_jobs)

# power.plot_topo(baseline=(bs_min, bs_max), mode='zscore', tmin=-0.4, tmax=0.4, vmin=-5, vmax=5,
#                 fmin=4, fmax=40, yscale='linear')


# Calculate Power per Subject and ROI
pows_lon = dict(a=list(), b=list(), c=list(), d=list(), e=list(), f=list())
pows_sho = dict(a=list(), b=list(), c=list(), d=list(), e=list(), f=list())

pow_x_subj = {'Longer': list(), 'Shorter': list()}
for ix, cond in enumerate([exp_lon, exp_sho]):
    # pow_fig, axes = plt.subplots(nrows=6, ncols=16, sharey=True, sharex=True, figsize=(20, 10))
    for ix_s, s in enumerate(cond):
        power_subj = tfr_morlet(s, freqs=freqs, n_cycles=n_cycles, use_fft=True, average=True,
                                return_itc=False, n_jobs=n_jobs)

        pow_x_subj[conds[ix]].append(power_subj)

        for ix_r, r in enumerate(sorted(rois)):
            roi = rois[r]
            pow_subj_c = power_subj.copy()
            roi_pow = pow_subj_c.pick_channels(roi)
            roi_pow.data = np.mean(roi_pow.data, 0, keepdims=True)
            roi_pow.info['nchan'] = 1
            roi_pow.info['chs'] = [roi_pow.info['chs'][0]]

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
stat_fig, axes = plt.subplots(6, 2, figsize=(8, 10))
dfs_list = list()
for ix_r, r in enumerate(sorted(rois)):
    pows_lon_z = pows_lon[r].copy()
    pows_sho_z = pows_sho[r].copy()

    pow_lon_win = [np.mean(p.copy().apply_baseline(mode='zscore', baseline=(bs_min, bs_max), verbose=False).crop(tmin=tmin, tmax=tmax).data[:, fq_mask, :])
                   for p in pows_lon_z]
    pow_sho_win = [np.mean(p.copy().apply_baseline(mode='zscore', baseline=(bs_min, bs_max), verbose=False).crop(tmin=tmin, tmax=tmax).data[:, fq_mask, :])
                   for p in pows_sho_z]

    # pow_lon_win = [np.mean(p.copy().apply_baseline(mode='mean', baseline=(bs_min, bs_max), verbose=False).crop(tmin=tmin, tmax=tmax).data[:, fq_mask, :])
    #                for p in pows_lon_z]
    # pow_sho_win = [np.mean(p.copy().apply_baseline(mode='mean', baseline=(bs_min, bs_max), verbose=False).crop(tmin=tmin, tmax=tmax).data[:, fq_mask, :])
    #                for p in pows_sho_z]

    # Detect Outliers
    outl_cond = list()
    for p in [pow_lon_win, pow_sho_win]:
        outl_cond.append([ix for ix, s in enumerate(p) if (s > np.mean(p) + 3.5 * np.std(p)) or
                         (s < np.mean(p) - 3.5 * np.std(p))])

    outliers[r] = np.union1d(outl_cond[0], outl_cond[1])

    pow_lon_win_ok = [p for ix, p in enumerate(pow_lon_win) if ix not in outliers[r]]
    pow_sho_win_ok = [p for ix, p in enumerate(pow_sho_win) if ix not in outliers[r]]

    n_perm = 10e3
    t_real, t_list, p_permuted = permutation_t_test(pow_lon_win_ok, pow_sho_win_ok, n_perm)

    # Save data
    subjs = [tr_info['subj'].loc[s] for s in np.arange(n_subj) if s not in outliers[r]]
    n_subj_roi = len(subjs)
    pow_dat = pd.DataFrame({'subject': np.tile(subjs, 2), 'roi': np.repeat(r, n_subj_roi*2),
                            'condition': ['longer']*n_subj_roi + ['shorter']*n_subj_roi,
                            'power': pow_lon_win_ok + pow_sho_win_ok})

    dfs_list.append(pow_dat)
    p_vals.append(p_permuted)

    # Fig Stats
    axes[ix_r, 0].violinplot([pow_lon_win_ok, pow_sho_win_ok], showmeans=True)
    axes[ix_r, 0].set_ylabel('z-score')
    # axes[ix_r, 0].set_ylabel('mean')
    axes[ix_r, 0].set_ylim(-7, 12)
    axes[ix_r, 1].hist(t_list, bins=50, facecolor='black')
    axes[ix_r, 1].vlines(t_real, ymin=0, ymax=900, linestyles='--')
    axes[ix_r, 1].set_ylim(-5, 900)
    axes[ix_r, 1].set_xlim(-5, 5)
    stat_fig.suptitle('{}-{} Hz Power \n {} to {}ms' .format(fmin, fmax, int(tmin*1000), int(tmax*1000)))

print('Outliers found)')
x = [print('ROI: {} - outliers: {} - subjects: {}' .format(r, len(outliers[r]), outliers[r])) for r in sorted(outliers)]
p_vals_corr = multipletests(p_vals, 0.05, 'holm')[1]
print(p_vals_corr)

stat_fig.savefig(op.join(study_path, 'figures', 'sclap_tf_stats.png'), format='png', dpi=300)

power_data = pd.concat(dfs_list)
power_data.to_csv(op.join(study_path, 'tables', 'power_data.csv'))

# for ax in enumerate(axes[:, 1]):
#     ax.set_title('p = {}' .format(p_vals_corr[ix]))

# Fig Power
pow_plot, axes = plt.subplots(6, 2, sharex=True, sharey=True, figsize=(8, 10))
for ix_c, c in enumerate([exp_lon, exp_sho]):
    for ix_r, r in enumerate(sorted(rois)):
        # c_ok = [s for ix, s in enumerate(c) if ix not in outliers[r]]
        c_evo_ok = mne.combine_evoked(c, weights='nave')
        # c_evo_ok = mne.combine_evoked(c, weights='nave')
        c_power = tfr_morlet(c_evo_ok, freqs=freqs, n_cycles=n_cycles, use_fft=True, average=True,
                             return_itc=False, n_jobs=n_jobs)

        roi = rois[r]
        roi_pow = c_power.copy()
        roi_pow.pick_channels(roi)
        roi_pow.data = np.mean(roi_pow.data, 0, keepdims=True)
        roi_pow.plot(baseline=(bs_min, bs_max), mode='zscore', tmin=-0.4, tmax=0.4, vmin=-20, vmax=20,
                     fmin=4, fmax=40, picks=[0], axes=axes[ix_r, ix_c], colorbar=False)
        # roi_pow.plot(baseline=(bs_min, bs_max), mode='mean', tmin=-0.4, tmax=0.4,
        #              fmin=4, fmax=40, picks=[0], axes=axes[ix_r, ix_c], colorbar=False)

        axes[ix_r, ix_c].vlines(0, ymin=0, ymax=axes[ix_r, ix_c].get_ylim()[1], linestyles='--')
        print('Number of trials -Cond {} -ROI {}: {}' .format(conds[ix_c], r, c_evo_ok.nave))
pow_plot.savefig(op.join(study_path, 'figures', 'sclap_tf_chart.png'), format='png', dpi=300)


# ---- TF Stats ------

r = 'e'
mode = 'zscore'

for r in sorted(rois.keys()):
    roi_pows_corr_lon = [p.copy().apply_baseline(mode=mode, baseline=(-0.95, -0.75)).crop(-0.5, 0.5).data for p in pows_lon[r]]
    roi_pows_corr_sho = [p.copy().apply_baseline(mode=mode, baseline=(-0.95, -0.75)).crop(-0.5, 0.5).data for p in pows_sho[r]]

    power_c1 = np.stack(roi_pows_corr_lon, axis=0).squeeze()
    power_c2 = np.stack(roi_pows_corr_sho, axis=0).squeeze()

    threshold = None
    T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(power_c1, n_permutations=1000, threshold=threshold, tail=1)

    # threshold = None
    # # threshold = {'start': 0, 'step': 0.2} # TFCE
    # T_obs, clusters, cluster_p_values, H0 = \
    #     permutation_cluster_test([power_c1, power_c2],
    #                              n_permutations=1000, threshold=threshold, tail=0, n_jobs=n_jobs)

    p_val = 0.01
    good_cluster_inds = np.where(cluster_p_values < p_val)[0]
    print(good_cluster_inds)
    print(len(good_cluster_inds))

    times = np.linspace(-0.5, 0.5, power_c1.shape[2])
    times = 1e3 * times

    T_obs_plot = np.nan * np.ones_like(T_obs)
    for c, p_val in zip(clusters, cluster_p_values):
        if p_val <= 0.05:
            T_obs_plot[c] = T_obs[c]

    # vmax = np.max(np.abs(T_obs))
    # vmin = -vmax

    vmax = 12
    vmin = -vmax

    plt.subplot(1, 1, 1)
    plt.imshow(T_obs, cmap=plt.cm.gray,
               extent=[times[0], times[-1], freqs[0], freqs[-1]],
               aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
    plt.imshow(T_obs_plot, cmap=plt.cm.Spectral,
               extent=[times[0], times[-1], freqs[0], freqs[-1]],
               aspect='auto', origin='lower', vmin=vmin, vmax=vmax)

    plt.colorbar()
    plt.xlabel('Time (ms)')
    plt.ylabel('Frequency (Hz)')
    # plt.title('Induced power (%s)' % ch_name)
    # plt.savefig(op.join(study_path, 'figures', 'Power_btw_conds_ROI_{}.svg' .format(r)), format='svg', dpi=300)
    plt.savefig(op.join(study_path, 'figures', 'Power_vs_base_ROI_{}_lon.svg' .format(r)), format='svg', dpi=300)
    plt.clf()


# Power desc
tf_fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
im1 = ax[0].imshow(np.mean(power_c1, axis=0), origin='lower', extent=[times[0], times[-1], freqs[0], freqs[-1]], aspect='auto', vmin=-4, vmax=4,
             cmap='coolwarm')
im2 = ax[1].imshow(np.mean(power_c2, axis=0), origin='lower', extent=[times[0], times[-1], freqs[0], freqs[-1]], aspect='auto', vmin=-4, vmax=4,
             cmap='coolwarm')
plt.colorbar(im1)


# Cluster power
sig_clust = [c for c, sig in zip(clusters, cluster_p_values) if sig < 0.05][0]
sig_mask = np.broadcast_to(sig_clust, power_c1.shape)
sig_pow = [np.where(sig_mask, p, np.nan) for p in [power_c1, power_c2]]
sig_pow_subj = [np.nanmean(p, axis=(1, 2)) for p in sig_pow]
t_real, t_list, p_permuted = permutation_t_test(sig_pow_subj[0], sig_pow_subj[1], 100)

plt.style.use('ggplot')
plt.violinplot(sig_pow_subj, showmeans=True)
plt.ylabel('Cluster Power \n (z-score from baseline)')
plt.xticks([1, 2], conds)

T_obs_lon, p_value_lon, H0_lon = permutation_t_test(np.array(sig_pow_subj[0], ndmin=2).T, n_permutations=10000, tail=0)
T_obs_sho, p_value_sho, H0_sho = permutation_t_test(np.array(sig_pow_subj[1], ndmin=2).T, n_permutations=10000, tail=0)
p_corr = multipletests([p_value_lon[0], p_value_sho[0]], method='holm')


# Get motor potential
all_log = pd.read_csv(op.join(study_path, 'tables', 's_trial_dat.csv'))
all_log = all_log.drop(['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1', 'a', 'b', 'c', 'd', 'e', 'f'], axis=1)
mp_epo_all = list()
for ix_s, s in enumerate(subjects):
    s_dat = all_log[(all_log.subject == int(s)) & (all_log.condition == 70)]
    s_dat.loc[:, 'tr_n'] = np.arange(len(s_dat))
    s_dat = set_dif_and_rt_exp(s_dat)
    s_epo = sho_epo[ix_s]
    mp_dat = s_dat[(s_dat['rt_exp'] > -0.55) & (s_dat['rt_exp'] < 0.55)]
    s_epo = s_epo[mp_dat.tr_n.tolist()]
    mp_epo = list()
    for ix_ep, ep in enumerate(s_epo):
        resp_mask = (s_epo.times < mp_dat.iloc[ix_ep].rt_exp + 0.125) & (s_epo.times > mp_dat.iloc[ix_ep].rt_exp - 0.125)
        mp_epo.append(ep[:, resp_mask])
    mp_epochs = np.dstack(mp_epo[:4])
    mp_avg = np.average(mp_epochs, axis=2)
    mp_epo_all.append(mp_avg)
mp_epochs_all = np.dstack(mp_epo_all)
mp_epochs_avg = np.average(mp_epochs_all, axis=2)
evk = mne.EvokedArray(mp_epochs_avg, s_epo.info)
evk.times = np.linspace(-0.125, 0.125, 64)
evk.apply_baseline((None, None))
evk.plot_topo()

evk.plot_topomap(np.linspace(-0.125, 0.125, 5))
sho_evo.plot_topomap(np.linspace(-0.125, 0.125, 5))
lon_evo.plot_topomap(np.linspace(-0.125, 0.125, 5))

df_pow = pd.DataFrame({'pow_lon': sig_pow_subj[0], 'pow_sho': sig_pow_subj[1]})
df_pow['subject'] = subjects
df_pow.to_csv(op.join(study_path, 'tables', 'pow_table.csv'))

# rt = all_log[all_log.condition == 90][['RT']].groupby(all_log['subject'], as_index=False).agg([np.nanmean])['RT']['nanmean'].tolist()
# df_pow['RT'] = rt
# df_pow['acc'] = all_log[['Accuracy']].groupby(all_log['subject'], as_index=False).agg(np.nanmean)


df_pow = pd.concat([df_pow, tr_info], axis=1, join='inner')
plt.style.use('classic')
scatter_matrix(df_pow)
from scipy.stats import spearmanr, pearsonr
df_pow['lon_log'] = np.log(df_pow['pow_lon'])
df_pow = df_pow.dropna()
spearmanr(df_pow['acc'], df_pow['RT'])
pearsonr(df_pow['lon_log'], df_pow['acc'])
plt.scatter(df_pow['acc'], df_pow['RT'])


# Fig Power 2
pow_plot, axes = plt.subplots(6, 2, sharex=True, sharey=True, figsize=(8, 10))
for ix_c, c in enumerate([pows_lon, pows_sho]):
    for ix_r, r in enumerate(sorted(rois)):

        c_pow = np.concatenate([p.data for p in c[r]])
        c_pow_mean = np.mean(c_pow, axis=0, keepdims=True)

        mean_TFR = mne.time_frequency.AverageTFR(c[r][0].info, c_pow_mean, c[r][0].times, c[r][0].freqs, len(c))

        mean_TFR.plot(baseline=(bs_min, bs_max), mode='zscore', tmin=-0.4, tmax=0.4, vmin=-20, vmax=20,
                      fmin=4, fmax=40, picks=[0], axes=axes[ix_r, ix_c], colorbar=False)
        # roi_pow.plot(baseline=(bs_min, bs_max), mode='mean', tmin=-0.4, tmax=0.4,
        #              fmin=4, fmax=40, picks=[0], axes=axes[ix_r, ix_c], colorbar=False)

        axes[ix_r, ix_c].vlines(0, ymin=0, ymax=axes[ix_r, ix_c].get_ylim()[1], linestyles='--')

pow_plot.savefig(op.join(study_path, 'figures', 'Power_each_cond_ROIs.png' .format(r)), format='png', dpi=300)




# # Subtraction
# conds_powers = list()
# for c in [lon_evo, sho_evo]:
#     c_power = tfr_morlet(c, freqs=freqs, n_cycles=n_cycles, use_fft=True, average=True,
#                          return_itc=False, n_jobs=n_jobs)
#     conds_powers.append(c_power)
#
# diff_pow = conds_powers[0] - conds_powers[1]
# diff_pow.plot_topo(baseline=(bs_min, bs_max), mode='zscore', tmin=-0.4, tmax=0.4, vmin=-5, vmax=5,
#                    fmin=4, fmax=40, yscale='linear')
#
# diff_pow_dat = np.mean(power.apply_baseline(mode='zscore', baseline=(bs_min, bs_max)).data[:, fq_mask, :][:, :, t_mask], axis=(1, 2))
#
# diff_rank_ch = [diff_pow.info['ch_names'][ch] for ch in np.argsort(diff_pow_dat)[::-1]]
# diff_rank_val = [diff_pow_dat[ch] for ch in np.argsort(diff_pow_dat)[::-1]]
#
# rank_fig, ax = plt.subplots(1)
# ax.bar(np.arange(len(diff_rank_val)), diff_rank_val)
# ax.set_xticks(np.arange(len(diff_rank_val)))
# ax.set_xticklabels(diff_rank_ch, rotation='vertical')
# ax.tick_params(labelsize=6)
#
#
# # Find electrode for comparison
# ROI = natsorted(np.union1d(ROI_d, ROI_i))
# ch_mask = np.in1d(power.info['ch_names'], ROI)
# ROI_chans = mne.pick_channels(power.info['ch_names'], ROI)
# ROI_power = np.mean(power.apply_baseline(mode='zscore', baseline=(bs_min, bs_max)).data[ch_mask, :, :][:, fq_mask, :][:, :, t_mask], axis=(1, 2))
# chan = ROI[np.argmax(ROI_power)]
# ch = mne.pick_channels(power.info['ch_names'], [chan])
# print('Selected channel: ', chan)
#
#

