import mne
import os.path as op
import numpy as np
from mne.time_frequency import tfr_morlet, AverageTFR
from mne.connectivity import spectral_connectivity, seed_target_indices
from mne.stats import permutation_cluster_test
from mne.decoding import TimeDecoding, GeneralizationAcrossTime, get_coef
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import StratifiedKFold
from etg_scalp_info import study_path, subjects, n_jobs, rois
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import sem
pd.set_option('display.expand_frame_repr', False)


def load_subj(subj):
    # Baseline Parameters
    bs_min = -0.95  # Pre-s2 baseline
    bs_max = -0.75

    epochs_file = op.join(study_path, 'epo', '{}_exp-epo.fif' .format(subj))
    epochs = mne.read_epochs(epochs_file, preload=True)

    if (len(epochs['exp_sup_lon']) > 30) and (len(epochs['exp_sup_sho']) > 20):
        epochs.info['subject_info'] = subj
        epochs.drop_channels(['STI 014'])
        epochs.apply_baseline(baseline=(bs_min, bs_max))

        exp_lon = epochs['exp_sup_lon']
        exp_sho = epochs['exp_sup_sho']
        exp_lon.info['cond'] = 'lon'
        exp_sho.info['cond'] = 'sho'


        # log
        log_file = op.join(study_path, 'logs', 'exp_ok', '{}_log_exp.csv'.format(subj))
        log_ok = pd.read_csv(log_file)
        log_ok['a'] = np.nan; log_ok['b'] = np.nan; log_ok['c'] = np.nan; log_ok['d'] = np.nan; log_ok['e'] = np.nan; log_ok['f'] = np.nan
        return exp_lon, exp_sho, log_ok


def tf_single_trial(epochs, log):
    # Window of interest
    fmin = 8
    fmax = 12
    tmin = -0.1
    tmax = 0.1

    # TF parameters
    freqs = np.arange(fmin, fmax, 0.1)  # frequencies of interest
    n_cycles = 4.

    power = tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, use_fft=True, average=False,
                       return_itc=False, n_jobs=n_jobs)

    power.apply_baseline(baseline=(-0.95, -0.75), mode='mean')

    for ix_r, r in enumerate(sorted(rois)):
        roi = rois[r]
        pow_subj_c = power.copy()
        roi_pow = pow_subj_c.pick_channels(roi)
        tr_pow = np.mean(roi_pow.crop(tmin, tmax).data, axis=(1, 2, 3))
        log.loc[epochs.selection, r] = tr_pow
    return log


def calc_connect(epochs):
    cwt_freqs = np.arange(8, 30, 1)
    cwt_cycles = cwt_freqs / 4.
    epochs.crop(-0.7, 0.7)
    con, freqs, times, _, _ = spectral_connectivity(epochs, method='wpli2_debiased', mode='cwt_morlet', sfreq=epochs.info['sfreq'],
                                                    cwt_frequencies=cwt_freqs, cwt_n_cycles=cwt_cycles, n_jobs=n_jobs)

    np.savez(op.join(study_path, 'results', 'wpli', '{}_{}_wpli' .format(epochs.info['subject_info'], epochs.info['cond'])),
             con=con, times=epochs.times, freqs=cwt_freqs, nave=len(epochs), info=epochs.info)

    # tfr = AverageTFR(epochs.info, con[33, :, :, :], epochs.times, freqs, len(epochs))
    # tfr.plot_topo(fig_facecolor='w', font_color='k', border='k', vmin=0, vmax=1, cmap='viridis')


def con_analysis(subjects, log):
    conds = ['lon', 'sho']
    roi_cons = {}
    roi = rois['f']

    for ix_s, s in enumerate(subjects):
        print('subject {} of {}' .format(ix_s+1, len(subjects)))
        for ix_c, c in enumerate(conds):
            filename = op.join(study_path, 'results', 'wpli', '{}_{}_wpli.npz' .format(s, c))
            dat = np.load(filename)
            info = dat['info'].item()

            # Create results matrices
            if ix_s == 0:
                roi_cons[c] = np.empty((len(subjects), dat['con'].shape[0], dat['con'].shape[-2], dat['con'].shape[-1]))

            # Get ROI connectivity
            # roi_ixs = [ix for ix, ch in enumerate(info['ch_names']) if ch in roi]
            # roi_con = np.mean(dat['con'][roi_ixs, :, :, :], axis=0)
            roi_ixs = 33
            roi_con = dat['con'][roi_ixs, :, :, :]
            roi_con[roi_ixs, :, :] = 1.0
            roi_cons[c][ix_s, :, :, :] = roi_con.copy()

    avg_con = [np.mean(roi_cons[c], axis=0) for c in conds]

    for ix_c, c in enumerate(conds):
        tfr = AverageTFR(info, avg_con[ix_c], dat['times'], dat['freqs'], len(subjects))
        tfr.plot_topo(fig_facecolor='w', font_color='k', border='k', vmin=0, vmax=0.5, cmap='viridis', title=c)

    # s = 1
    # for c in conds:
    #     tfr = AverageTFR(info, roi_cons[c][s, :,:,:], dat['times'], dat['freqs'], len(subjects))
    #     tfr.plot_topo(fig_facecolor='w', font_color='k', border='k', vmin=0, vmax=1, cmap='viridis', title=c)

    # Stats
    test_con = [roi_cons[c][:, 19, :, :] for c in conds]

    threshold = None
    T_obs, clusters, cluster_p_values, H0 = \
        permutation_cluster_test([test_con[0], test_con[1]],
                                 n_permutations=1000, threshold=threshold, tail=0)

    times = dat['times']
    times *= 1e3
    freqs = dat['freqs']

    fig, ax = plt.subplots(1)
    T_obs_plot = np.nan * np.ones_like(T_obs)
    for c, p_val in zip(clusters, cluster_p_values):
        if p_val <= 0.05:
            T_obs_plot[c] = T_obs[c]

    ax.imshow(T_obs,
               extent=[times[0], times[-1], freqs[0], freqs[-1]],
               aspect='auto', origin='lower', cmap='gray')
    ax.imshow(T_obs_plot,
               extent=[times[0], times[-1], freqs[0], freqs[-1]],
               aspect='auto', origin='lower', cmap='RdBu_r')

    plt.xlabel('Time (ms)')
    plt.ylabel('Frequency (Hz)')
    plt.title('ROI Connectivity')


def decoding_analysis(c1, c2):
    td = TimeDecoding(predict_mode='cross-validation', n_jobs=1)
    epochs = mne.concatenate_epochs([c1, c2])
    td.fit(epochs)
    td.score(epochs)
    # td.plot('Subject: ', c1.info['subject_info'])
    scores = td.scores_


    # # GAT
    # y = np.zeros(len(epochs.events), dtype=int)
    # y[epochs.events[:, 2] == 90] = 1
    # cv = StratifiedKFold(y=y)  # do a stratified cross-validation
    #
    # gat = GeneralizationAcrossTime(predict_mode='cross-validation', n_jobs=1,
    #                                cv=cv, scorer=roc_auc_score)
    #
    # # fit and score
    # gat.fit(epochs, y=y)
    # gat.score(epochs)
    #
    # # plot
    # gat.plot(vmin=0, vmax=1)
    # gat.plot_diagonal()
    return scores


def plot_time_decoding(subj_scores):
    scores_arr = np.array(subj_scores)
    mean_sco = np.mean(scores_arr, axis=0)
    sem_sco = sem(scores_arr, axis=0)

    t_masks = [[lon.times < -0.7], [lon.times > -0.7]]
    dec_fig, axes = plt.subplots(1, 2, sharey=True, gridspec_kw={'width_ratios': [1,3]})
    for ix_m, m in enumerate(t_masks):
        axes[ix_m].plot(lon.times[m], mean_sco[m])
        axes[ix_m].fill_between(lon.times[m], mean_sco[m]+sem_sco[m], mean_sco[m]-sem_sco[m], alpha=0.3)
        axes[ix_m].set_ylim(0.45, 0.7)
    axes[1].vlines(0, ymin=axes[1].get_ylim()[0], ymax=axes[1].get_ylim()[1], linestyles='--')
    axes[1].set_xlim(-0.7, 0.7)
    dec_fig.tight_layout(w_pad=0.1)

# subjects = subjects[:8]
if __name__ == '__main__':
    all_s_dat = list()
    all_scores = list()
    for subj in subjects:
        print('Subject: ', subj)
        lon, sho, log = load_subj(subj)
        scores = decoding_analysis(lon, sho)
        all_scores.append(scores)

        # Single trial time-frequency & connectivity
        for cond in [lon, sho]:
            # log = tf_single_trial(cond, log)
            log = log.dropna()
            # calc_connect(cond)
        all_s_dat.append(log)

    # Log
    all_dat = pd.concat(all_s_dat, ignore_index=True)
    # all_dat.to_csv(op.join(study_path, 'tables', 's_trial_dat.csv'))

    plot_time_decoding(all_scores)

    # Connectivity analysis
    # all_dat = pd.read_csv(op.join(study_path, 'tables', 's_trial_dat.csv'))
    # con_analysis(subj, all_dat, lon.info)