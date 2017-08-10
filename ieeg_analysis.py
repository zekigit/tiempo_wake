import mne
from mne.time_frequency import tfr_morlet, AverageTFR
import numpy as np
import os.path as op
import matplotlib.pyplot as plt
from etg_ieeg_info import dat_path, files, n_jobs, log_path, marks, marks_exp, marks_p, study_path
from eeg_etg_fxs import read_log_file, check_events, durations_from_log_ieeg, create_events, add_event_condition, make_exp_baseline, \
    add_event_tr_id, permutation_t_test
from ieeg_fx import make_bip_chans, permutation_test
import pandas as pd
from mne.stats import permutation_cluster_1samp_test, permutation_cluster_test
from mpl_toolkits.axes_grid1 import ImageGrid
from scipy.stats import ttest_ind
from statsmodels.sandbox.stats.multicomp import multipletests
from subj_and_group_analysis import calc_wpli_over_time, decoding_analysis

pd.set_option('display.expand_frame_repr', False)
plt.style.use('ggplot')

subj = 'P14'
conds = ['lon', 'sho']
ref = 'avg'


# ----- LOAD AND PREPROCESS
ratios = 7
if ratios == 3:
    file_ix = 0
else:
    file_ix = 1

# Baseline Parameters
bs_min = -0.95  # Pre-s2 baseline
bs_max = -0.75

# Load Data
file = op.join(dat_path, 'set', files[subj][file_ix] + '.set')
raw = mne.io.read_raw_eeglab(file, preload=True)

# Channel Information
ch_info = pd.read_pickle(op.join(study_path, 'info', 's3_avg_info_coords.pkl'))
ch_dict = {}
for ix, ch in enumerate(raw.info['ch_names'][:-1]):
    ch_dict[ch] = ch_info['Electrode'].iloc[ix]
raw.rename_channels(ch_dict)

# Re-reference
if ref == 'bip':
    bip_chans, anodes, cathodes = make_bip_chans(ch_info)
    raw = mne.io.set_bipolar_reference(raw, anode=anodes, cathode=cathodes)
    bads = bip_chans['Electrode'][bip_chans['White Grey'] != 'Grey Matter'].tolist()
    raw.info['bads'] = bads
    picks = [72, 73,
             78, 79]

elif ref == 'avg':
    bads = ch_info['Electrode'][ch_info['White Grey'] != 'Grey Matter'].tolist()
    raw.info['bads'] = bads
    raw.drop_channels(bads)
    raw, re_ref = mne.io.set_eeg_reference(raw, ref_channels=None, copy=True)
    raw.apply_proj()
    picks = [60, 61, 62, 63,  # 63, 55
             52, 53, 54, 55]

# Filters
notchs = np.arange(50, 151, 50)
raw.filter(l_freq=1, h_freq=140, n_jobs=n_jobs)
raw.notch_filter(freqs=notchs, n_jobs=n_jobs)

# Log
log_filename = log_path + 'ETG{}a_{}.mat' .format(ratios, subj)
log = read_log_file(log_filename)

events = mne.find_events(raw)
events = check_events(events)
durations = durations_from_log_ieeg(log)
if ratios == 3:
    durations = [d for d in durations if ~np.isnan(d)]
    log = log[:-11]

new_events = create_events(events, durations, raw.info['sfreq'])
raw.add_events(new_events)
events_updated = mne.find_events(raw, shortest_event=1)
events_updated = check_events(events_updated)
events_updated, log = add_event_condition(events_updated, log)
events_tr_id = add_event_tr_id(events_updated)

# Epoch
t_min = -0.7
t_max = 0.7
reject = {'eeg': 150e5}

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


for ix, ep in enumerate([exp_sup_lon, exp_sup_sho]):
    ep.drop_channels(ep.info['bads'])
    ep.save(op.join(study_path, 'data', 'P14_{}-epo.fif' .format(conds[ix])))


# ---- ANALYSIS----

# Time-Frequency
freqs = np.arange(4, 121, 0.1)  # define frequencies of interest
# freqs = np.arange(8, 121, 0.1)  # define frequencies of interest
# n_cycles = freqs / 2.  # different number of cycle per frequency
n_cycles = 3.


pow_list = list()
tf_fig, axes = plt.subplots(len(picks), 2, figsize=(8, 10))
for ix_c, c in enumerate([exp_sup_lon, exp_sup_sho]):
    power = tfr_morlet(c, freqs=freqs, n_cycles=n_cycles, use_fft=True, average=False,
                       return_itc=False, n_jobs=n_jobs, picks=picks)

    pow_list.append(power.copy())
    for ix_ch, ch in enumerate(picks):
        vmax = 10

        power.average().plot(picks=[ix_ch], baseline=(bs_min, bs_max), mode='zscore', fmin=5, fmax=120, tmin=-0.4, tmax=0.4,
                             vmax=vmax, vmin=-vmax, axes=axes[ix_ch, ix_c], colorbar=None)

        # power.average().plot(picks=[ix_ch], baseline=(bs_min, bs_max), mode='mean', fmin=4, fmax=120, tmin=-0.4, tmax=0.4,
        #                      axes=axes[ix_ch, ix_c], colorbar=None)
        axes[ix_ch, ix_c].set_title(exp_sup_lon.info['ch_names'][ch])

tf_fig.tight_layout()
tf_fig.savefig(op.join(study_path, 'figures', 'tf_intra_mean.png'), format='png', dpi=300)

# Transform to z-score from baseline
pows_corr = [p.copy().apply_baseline(mode='zscore', baseline=(-0.95, -0.75)) for p in pow_list]
channels = pows_corr[0].info['ch_names']

# Single Trial Plot
tmax = 0.4
tmin = -tmax
for ix_c, c in enumerate(pows_corr):
    c.crop(tmin=tmin, tmax=tmax)
    for ix_ch, ch in enumerate(picks):
        sing_fig = plt.figure(figsize=(20, 10))
        grid = ImageGrid(sing_fig, 111,
                         nrows_ncols=(6, 7),
                         axes_pad=0.1,
                         share_all=True,
                         aspect=True,
                         cbar_mode=None,
                         cbar_pad='15%',
                         cbar_location='right')

        for ix_t, ax in enumerate(grid):
            ax.imshow(c.data[ix_t, ix_ch, :, :], origin='lower', aspect=0.125, vmin=-20, vmax=20)
            ax.vlines(np.where(c.times == 0), ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], linestyles='--')

        sing_fig.suptitle('Channel: {} - Condition: {}'.format(channels[ix_ch], conds[ix_c]))
        sing_fig.savefig(op.join(study_path, 'figures', 'ieeg_strial_power_{}_{}.pdf' .format(channels[ix_ch], conds[ix_c])),
                         format='pdf', dpi=300)


# Stats between conditions
tmin = -0.1
tmax = 0.1
t_mask = (pows_corr[0].times >= tmin) & (pows_corr[0].times <= tmax)

fmin = 8
fmax = 12
fq_mask = (freqs >= fmin) & (freqs <= fmax)

pow_win = [np.mean(p.copy().crop(tmin=tmin, tmax=tmax).data[:, :, fq_mask, :], axis=(2, 3)) for p in pows_corr]

n_perm = 10e3
p_vals = list()
dats = list()
stat_fig, axes = plt.subplots(len(picks), 2, figsize=(8, 10))
for ix_ch, ch in enumerate(channels):
    dat_lon = pow_win[0][:, ix_ch]
    dat_sho = pow_win[1][:, ix_ch]

    dat_ok = [[tr for tr in dat if (tr < np.mean(dat) + 3 * np.std(dat)) and
               (tr > np.mean(dat) - 3 * np.std(dat))] for dat in [dat_lon, dat_sho]]

    dats.append(dat_ok)

    print(len(dat_ok[0]), len(dat_ok[1]))

    t_real, p_real, t_list, p_permuted = permutation_test(dat_ok, n_perm)

    p_vals.append(p_permuted)

    # Fig Stats
    axes[ix_ch, 0].violinplot([dat_ok[0], dat_ok[1]], showmeans=True)
    # axes[ix_ch, 0].boxplot([dat_ok[0], dat_ok[1]])
    axes[ix_ch, 0].set_ylabel('z-score')
    axes[ix_ch, 0].set_title(ch)
    # axes[ix_ch, 0].set_ylim(-10, 15)
    axes[ix_ch, 1].hist(t_list, bins=50, facecolor='black')
    axes[ix_ch, 1].vlines(t_real, ymin=0, ymax=900, linestyles='--')
    axes[ix_ch, 1].set_ylim(-5, 900)
    axes[ix_ch, 1].set_xlim(-5, 5)
    stat_fig.suptitle('{}-{} Hz Power \n {} to {}ms'.format(fmin, fmax, int(tmin * 1000), int(tmax * 1000)))

p_vals_corr = multipletests(p_vals, 0.05, 'holm')[1]
print(p_vals_corr)

# plt.hist([dat_ok[0], dat_ok[1]])

# Behaviour
log_lon = log.loc[exp_sup_lon.selection]
log_sho = log.loc[exp_sup_sho.selection]

log_lon['SC4'] = pow_win[0][:, 0]
log_lon['HPi3'] = pow_win[0][:, 1]
log_lon['HP3'] = pow_win[0][:, 2]
log_lon['s2'] = 'longer'
log_sho['SC4'] = pow_win[1][:, 0]
log_sho['HPi3'] = pow_win[1][:, 1]
log_sho['HP3'] = pow_win[1][:, 2]
log_sho['s2'] = 'shorter'

all_dat = pd.concat([log_lon, log_sho])
all_dat.to_csv(op.join(study_path, 'tables', 'log_and_power_dat.csv'))

for ix_ch, ch in enumerate(channels):
    diff_l = [ix for ix, tr in enumerate(pow_win[0][:, ix_ch]) if tr not in dats[ix_ch][0]]
    diff_s = [ix for ix, tr in enumerate(pow_win[1][:, ix_ch]) if tr not in dats[ix_ch][1]]

    log_lon_ok = log_lon.drop(log_lon.index[diff_l])
    log_sho_ok = log_sho.drop(log_sho.index[diff_s])

    plt.scatter(dats[ix_ch][0], log_lon_ok['Response'].values)
    plt.scatter(dats[ix_ch][1], log_sho_ok['Response'].values)

    plt.scatter(log_lon_ok['Ratio'].values, dats[ix_ch][0])
    plt.scatter(log_sho_ok['Ratio'].values, dats[ix_ch][1])

    plt.scatter(log_lon_ok['RT'].values, dats[ix_ch][0])
    plt.scatter(log_sho_ok['RT'].values, dats[ix_ch][1])


# Decoding
td, gat = decoding_analysis(exp_sup_lon, exp_sup_sho)
td.plot()
gat.plot(vmin=0.4, vmax=0.7, tlim=(-0.7, 0.7, -0.7, 0.7))
gat.plot_diagonal()

# Connectivity
for ix_c, c in enumerate([exp_sup_lon, exp_sup_sho]):
    c.info['subject_info'] = 'P14'
    c.info['cond'] = conds[ix_c]
    calc_wpli_over_time(c)

dats = list()
for c in conds:
    dat = np.load(op.join(study_path, 'results', 'wpli', 'P14_{}_wpli.npz' .format(c)))
    dats.append(dat)

info = dat['info'].item()

for ix_c, c in enumerate(conds):
    tfr = AverageTFR(info, dats[ix_c]['con'][1, :, :, :], dat['times'], dat['freqs'], 1)
    tfr.plot(picks=[83], vmin=-1, vmax=1, cmap='viridis', title=c)


# TF ROI
if ref == 'avg':
    rois = {'HP_r': np.array(raw.ch_names)[picks[:4]], 'HP_l': np.array(raw.ch_names)[picks[4:]]}  # :4
elif ref == 'bip':
    rois = {'HP_l': np.array(raw.ch_names)[picks[:2]], 'HP_r': np.array(raw.ch_names)[picks[2:]]}

roi_fig, axes = plt.subplots(len(rois), len(conds), sharey=True, sharex=True)
roi_pows = {'HP_l': list(), 'HP_r': list()}

for ix_c, c in enumerate(conds):
    for ix_r, r in enumerate(rois):
        roi = rois[r]
        roi_pow = pow_list[ix_c].copy()
        roi_pow.pick_channels(roi)
        roi_pow.data = np.mean(roi_pow.data, 1, keepdims=True)
        roi_pow.info['chs'] = [roi_pow.info['chs'][0]]
        roi_pow.average().plot(baseline=(bs_min, bs_max), mode='zscore', tmin=-0.4, tmax=0.4, vmin=-5, vmax=5,
                               fmin=4, fmax=120, picks=[0], axes=axes[ix_r, ix_c], colorbar=False)
        # roi_pow.average().plot(baseline=(bs_min, bs_max), mode='mean', tmin=-0.4, tmax=0.4,
        #                        fmin=4, fmax=120, picks=[0], axes=axes[ix_r, ix_c], colorbar=False)

        axes[ix_r, ix_c].vlines(0, ymin=0, ymax=axes[ix_r, ix_c].get_ylim()[1], linestyles='--')
        axes[ix_r, ix_c].set_title(r)

        roi_pows[r].append(roi_pow)


# ROI Stats
roi_stat_fig, axes = plt.subplots(len(rois), 2)
p_vals = list()
for ix_r, r in enumerate(rois):
    if ix_r == 0:
        ch_nrs = np.arange(0, 4)
    else:
        ch_nrs = np.arange(4, 8)

    dat_lon = np.mean(pow_win[0][:, ch_nrs], axis=1)
    dat_sho = np.mean(pow_win[1][:, ch_nrs], axis=1)

    dat_ok = [[tr for tr in dat if (tr < np.mean(dat) + 3 * np.std(dat)) and
              (tr > np.mean(dat) - 3 * np.std(dat))] for dat in [dat_lon, dat_sho]]

    t_real, p_real, t_list, p_permuted = permutation_test(dat_ok, n_perm)
    p_vals.append(p_permuted)

    axes[ix_r, 0].violinplot([dat_ok[0], dat_ok[1]], showmeans=True)
    # axes[ix_ch, 0].boxplot([dat_ok[0], dat_ok[1]])
    axes[ix_r, 0].set_ylabel('z-score')
    axes[ix_r, 0].set_title(r)
    # axes[ix_ch, 0].set_ylim(-10, 15)
    axes[ix_r, 1].hist(t_list, bins=50, facecolor='black')
    axes[ix_r, 1].vlines(t_real, ymin=0, ymax=900, linestyles='--')
    axes[ix_r, 1].set_ylim(-5, 900)
    axes[ix_r, 1].set_xlim(-5, 5)
    roi_stat_fig.suptitle('ROI {}-{} Hz Power \n {} to {}ms'.format(fmin, fmax, int(tmin * 1000), int(tmax * 1000)))

p_vals_corr = multipletests(p_vals, 0.05, 'holm')[1]
print(p_vals_corr)


# ----------- END-----------

for r in ['HP_r', 'HP_l']:

    roi_pows_corr = [p.copy().apply_baseline(mode='zscore', baseline=(-0.95, -0.75)) for p in roi_pows[r]]

    power_lon = roi_pows_corr[0].crop(-0.4, 0.4)
    power_sho = roi_pows_corr[1].crop(-0.4, 0.4)

    power_c1 = power_lon.data[:, 0, :, :]
    power_c2 = power_sho.data[:, 0, :, :]
    #
    # threshold = None
    # T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(power_c1, n_permutations=1000, threshold=threshold, tail=1)
    #
    threshold = None
    T_obs, clusters, cluster_p_values, H0 = \
        permutation_cluster_test([power_c1, power_c2],
                                 n_permutations=500, threshold=threshold, tail=0)

    times = 1e3 * power_lon.times
    T_obs_plot = np.nan * np.ones_like(T_obs)
    for c, p_val in zip(clusters, cluster_p_values):
        if p_val <= 0.05:
            T_obs_plot[c] = T_obs[c]

    # vmax = np.max(np.abs(T_obs))
    # vmin = -vmax

    vmax = 14
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
    # plt.title('Induced power (%s)' % ch_name)
    plt.savefig(op.join(study_path, 'figures', 'iEEG_Power_btw_conds_ROI_{}.svg'.format(r)), format='svg', dpi=300)
    plt.clf()


# Plot electrode location
