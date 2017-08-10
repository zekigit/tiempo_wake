import mne
import os.path as op
import numpy as np
from mne.time_frequency import tfr_morlet, AverageTFR
from mne.connectivity import spectral_connectivity, seed_target_indices
from mne.stats import permutation_cluster_test
from mne.decoding import TimeDecoding, GeneralizationAcrossTime, get_coef
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import StratifiedKFold
from etg_scalp_info import study_path, subjects, n_jobs, rois, conditions
from eeg_etg_fxs import create_con_mat
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import pandas as pd
from scipy.stats import sem
from scipy.io import savemat
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


def calc_wpli_over_time(epochs):
    cwt_freqs = np.arange(8, 30, 1)
    cwt_cycles = cwt_freqs / 4.
    epochs.crop(-0.7, 0.7)
    con, freqs, times, _, _ = spectral_connectivity(epochs, method='wpli2_debiased', mode='cwt_morlet', sfreq=epochs.info['sfreq'],
                                                    cwt_frequencies=cwt_freqs, cwt_n_cycles=cwt_cycles, n_jobs=n_jobs)

    np.savez(op.join(study_path, 'results', 'wpli', '{}_{}_wpli' .format(epochs.info['subject_info'], epochs.info['cond'])),
             con=con, times=epochs.times, freqs=cwt_freqs, nave=len(epochs), info=epochs.info, chans=epochs.info['ch_names'])

    # tfr = AverageTFR(epochs.info, con[33, :, :, :], epochs.times, freqs, len(epochs))
    # tfr.plot_topo(fig_facecolor='w', font_color='k', border='k', vmin=0, vmax=1, cmap='viridis')


def calc_wpli_over_epochs(epochs):
    fmin = (1, 4, 8, 13, 30)
    fmax = (4, 7, 12, 30, 40)

    epochs.crop(-0.7, 0.7)
    con, freqs, times, _, _ = spectral_connectivity(epochs, method='wpli_debiased', mode='multitaper', fmin=fmin,  fmax=fmax, faverage=True,
                                                    mt_adaptive=False, n_jobs=n_jobs, verbose=False)

    np.savez(op.join(study_path, 'results', 'wpli', '{}_{}_dwpli_epochs' .format(epochs.info['subject_info'], epochs.info['cond'])),
             con=con, times=epochs.times, freqs=(fmin, fmax), nave=len(epochs), info=epochs.info, chans=epochs.info['ch_names'])


def wpli_analysis_time(subjects, log):
    conds = ['lon', 'sho']
    roi_cons = {}
    roi = rois['f']
    spatial_con = mne.channels.read_ch_connectivity('biosemi64')

    for ix_s, s in enumerate(subjects):
        print('subject {} of {}' .format(ix_s+1, len(subjects)))
        for ix_c, c in enumerate(conds):
            filename = op.join(study_path, 'results', 'wpli', 'over_time', '{}_{}_wpli.npz' .format(s, c))
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

    s = 12
    for c in conds:
        tfr = AverageTFR(info, roi_cons[c][s, :, :, :], dat['times'], dat['freqs'], len(subjects))
        tfr.plot_topo(fig_facecolor='w', font_color='k', border='k', vmin=0, vmax=1, cmap='viridis', title=c)

    # Stats
    test_con = [roi_cons[c][:, 19, :, :] for c in conds]

    #threshold = None
    threshold = dict(start=0, step=0.2)
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


def wpli_analysis_epochs(subjects):
    # load data
    mats, freqs, chans = load_wpli_over_epochs(subjects)

    # load spatial structure
    connectivity, ch_names = mne.channels.read_ch_connectivity(
        '/Users/lpen/Documents/MATLAB/Toolbox/fieldtrip-20170628/template/neighbours/biosemi128_neighb.mat')

    # square matrix
    for c in conditions:
        for s in range(len(subjects)):
            mats[c][s, :, :, :] = create_con_mat(mats[c][s, :, :, :])

    avg_mat = {key: np.nanmean(x, axis=0) for (key, x) in mats.items()}

    # avg plot
    subj_con_fig = plt.figure(figsize=(15, 5))
    grid = ImageGrid(subj_con_fig, 111,
                     nrows_ncols=(len(conditions), 5),
                     axes_pad=0.3,
                     cbar_mode='single',
                     cbar_pad='10%',
                     cbar_location='right')

    for idx, ax in enumerate(grid):
        if idx <= 4:
            im = ax.imshow(avg_mat[0][:, :, idx], vmin=0, vmax=0.4)
        else:
            im = ax.imshow(avg_mat[1][:, :, idx-5], vmin=0, vmax=0.4)

    cb = subj_con_fig.colorbar(im, cax=grid.cbar_axes[0])
    cb.ax.set_title('wPLI', loc='right')


    # subjs plot
    n_s = len(subjects)
    n_f = freqs.shape[1]
    n_c = len(conditions)

    plt_s = np.tile(np.arange(n_s), n_f*n_c)
    plt_c = np.tile(np.concatenate((np.repeat(0, n_s), np.repeat(1, n_s))), n_f)
    plt_f = np.repeat(np.arange(0, 5), n_s*n_c)

    plt.style.use('ggplot')
    subj_con_fig = plt.figure(figsize=(15, 10))
    grid = ImageGrid(subj_con_fig, 111,
                     nrows_ncols=(len(conditions)*freqs.shape[1], len(subjects)),
                     axes_pad=0.05,
                     share_all=True,
                     aspect=True,
                     cbar_mode='single',
                     cbar_pad='10%',
                     cbar_location='right')

    for (idx, ax), s, c, f in zip(enumerate(grid), plt_s, plt_c, plt_f):
        print(idx, s, c, f)
        im = ax.imshow(mats[conditions[c]][s, :, :, f], vmin=0, vmax=0.7)
        ax.set_xticks([], [])
        ax.set_yticks([], [])
        ax.grid(False)

    cb = subj_con_fig.colorbar(im, cax=grid.cbar_axes[0], ticks=np.arange(0, 1.1, 0.1))
    cb.ax.set_title('wPLI', loc='right')
    subj_con_fig.savefig(op.join(study_path, 'figures', 'subj_wpli.eps'), format='eps', dpi=300)

    # channel mean
    avg_ch_mat = [np.nanmean(mats[x], axis=2) for x in mats]
    avg_ch_mat = [np.transpose(x, (0, 2, 1)) for x in avg_ch_mat]

    # s = 16
    # plt.imshow(avg_ch_mat[0][s, :, :], aspect='auto', vmax=1, vmin=0)
    # plt.colorbar()

    threshold = dict(start=0, step=0.1)
    n_perm = 100

    fq_dat = [np.nan_to_num(mats[x][:, :, :, 2]) for x in mats]
    fq_dat = np.nan_to_num(fq_dat)

    T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test(fq_dat, n_permutations=n_perm, connectivity=connectivity,
                                                                     threshold=threshold, tail=0, n_jobs=2)

    plt.hist(cluster_p_values)


def load_wpli_over_epochs(subjects, save=False):
    mats = dict(lon=list(), sho=list())
    for s in subjects:
        for c in conditions:
            fname = op.join(study_path, 'results', 'wpli', 'over_epochs', '{}_{}_dwpli_epochs.npz' .format(s, c))
            raw = np.load(fname)
            mats[c].append(raw['con'])
    for c in conditions:
        mats[c] = np.stack(mats[c])

    if save:
        savemat(op.join(study_path, 'tables', 'wpli_epochs_all_dat.mat'), {'lon': mats['lon'], 'sho': mats['sho'],
                                                                            'freqs': raw['freqs'], 'chans': raw['chans']})

    freqs = raw['freqs']
    chans = raw['chans']
    return mats, freqs, chans


def decoding_analysis(c1, c2):
    td = TimeDecoding(predict_mode='cross-validation', n_jobs=1)
    epochs = mne.concatenate_epochs([c1, c2])
    td.fit(epochs)
    td.score(epochs)
    # td.plot('Subject: ', c1.info['subject_info'])

    # GAT
    y = np.zeros(len(epochs.events), dtype=int)
    y[epochs.events[:, 2] == 90] = 1
    cv = StratifiedKFold(y=y)  # do a stratified cross-validation

    gat = GeneralizationAcrossTime(predict_mode='cross-validation', n_jobs=1,
                                   cv=cv, scorer=roc_auc_score)

    # fit and score
    gat.fit(epochs, y=y)
    gat.score(epochs)

    # # plot
    # gat.plot(vmin=0, vmax=1)
    # gat.plot_diagonal()
    return td, gat


def plot_decoding(subj_scores):
    # Time Decoding
    scores_arr = np.array(subj_scores['td'])
    mean_sco = np.mean(scores_arr, axis=0)
    sem_sco = sem(scores_arr, axis=0)

    t_masks = [[lon.times < -0.7], [lon.times > -0.7]]
    dec_fig, axes = plt.subplots(1, 2, sharey=True, gridspec_kw={'width_ratios': [1, 3]})
    for ix_m, m in enumerate(t_masks):
        axes[ix_m].plot(lon.times[m], mean_sco[m])
        axes[ix_m].fill_between(lon.times[m], mean_sco[m]+sem_sco[m], mean_sco[m]-sem_sco[m], alpha=0.3)
        axes[ix_m].set_ylim(0.45, 0.7)
    axes[0].set_xlim(-0.95, -0.7)
    axes[0].hlines(0.5, xmin=axes[0].get_xlim()[0], xmax=axes[0].get_xlim()[1], linestyles=':')
    axes[1].hlines(0.5, xmin=axes[1].get_xlim()[0], xmax=axes[1].get_xlim()[1], linestyles=':')
    axes[1].vlines(0, ymin=axes[1].get_ylim()[0], ymax=axes[1].get_ylim()[1], linestyles='--')
    axes[1].set_xlim(-0.7, 0.7)
    axes[0].set_ylabel('Classification Performance (AUC)')
    dec_fig.tight_layout(w_pad=0.1)
    dec_fig.savefig(op.join(study_path, 'figures', 'group_decoding.eps'), format='eps', dpi=300)

    # GAT
    plt.style.use('ggplot')
    scores_gat = np.array(subj_scores['gat'])
    np.savez(op.join(study_path, 'results', 'decoding', 'group_gat'), gat=scores_gat)
    mean_gat = np.mean(scores_gat, axis=0)

    s2_exp = np.where(lon.times == 0)[0][0]
    s2_on = (np.abs(lon.times - -0.7)).argmin()
    ticks = np.array([0, 64, s2_exp-2*s2_on, s2_exp-s2_on, s2_exp, s2_exp+s2_on, s2_exp+2*s2_on], subok=True)
    ticks_labels = [-0.25, 0, -0.5, -0.25, 0, 0.25, 0.5]

    gat_fig, ax = plt.subplots(1, 1)
    cax = ax.imshow(mean_gat, origin='lower', vmin=0.5, vmax=0.65, cmap='plasma')
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(ticks_labels)
    ax.set_yticklabels(ticks_labels)
    ax.vlines(s2_on, ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], linestyles='-')
    ax.hlines(s2_on, xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1], linestyles='-')
    ax.vlines(s2_exp, ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], linestyles=':')
    ax.hlines(s2_exp, xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1], linestyles=':')
    ax.set_ylabel('Training Time (s)')
    ax.set_xlabel('Testing Time (s)')
    cbar = gat_fig.colorbar(cax, ticks=np.arange(0.4, 0.7, 0.05))
    gat_fig.savefig(op.join(study_path, 'figures', 'group_gat.eps'), format='eps', dpi=300)

    ax.set_xticklabels([-0.25, 0, 0.25, 0.5])


if __name__ == '__main__':
    all_s_dat = list()
    all_scores = dict(td=list(), gat=list())
    for subj in subjects:
        print('Subject: ', subj)

        # Load
        lon, sho, log = load_subj(subj)

        # Decoding
        td, gat = decoding_analysis(lon, sho)
        all_scores['td'].append(td.scores_)
        all_scores['gat'].append(gat.scores_)

        # Single trial time-frequency & connectivity
        for cond in [lon, sho]:
            # log = tf_single_trial(cond, log)
            log = log.dropna()
            # calc_wpli_over_epochs(cond)
        all_s_dat.append(log)

    # Log
    all_dat = pd.concat(all_s_dat, ignore_index=True)
    # all_dat.to_csv(op.join(study_path, 'tables', 's_trial_dat.csv'))

    # Plot group decoding
    plot_decoding(all_scores)

    # Connectivity analysis
    #  all_dat = pd.read_csv(op.join(study_path, 'tables', 's_trial_dat.csv'))
    #  wpli_analysis_time(subj, all_dat, lon.info)
