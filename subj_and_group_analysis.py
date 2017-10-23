import mne
import os.path as op
import numpy as np
from mne.time_frequency import tfr_morlet, AverageTFR
from mne.connectivity import spectral_connectivity, seed_target_indices
from mne.stats import permutation_cluster_test, spatio_temporal_cluster_1samp_test
from mne.decoding import TimeDecoding, GeneralizationAcrossTime
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import StratifiedKFold
from etg_scalp_info import study_path, subjects, n_jobs, rois, conditions
from eeg_etg_fxs import create_con_mat, set_dif_and_rt_exp, permutation_pearson
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import pandas as pd
from scipy.stats import sem
from scipy.io import savemat
import pickle
import seaborn
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
        log_ok = log_ok[log_ok.Condition == 2.0]
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
    c1.events[:, 2] = 0
    c2.events[:, 2] = 1
    c1.event_id['exp_sup_lon'] = 0
    c2.event_id['exp_sup_sho'] = 1
    epochs = mne.concatenate_epochs([c1, c2])

    # td = TimeDecoding(predict_mode='cross-validation', n_jobs=1, scorer=roc_auc_score)
    # td.fit(epochs)
    # td.score(epochs)
    # td.plot('Subject: ', c1.info['subject_info'], chance=True)

    # GAT
    y = epochs.events[:, 2]
    # y = np.zeros(len(epochs.events), dtype=int)
    # y[epochs.events[:, 2] == 90] = 1
    cv = StratifiedKFold(y=y)  # do a stratified cross-validation

    gat = GeneralizationAcrossTime(predict_mode='cross-validation', n_jobs=1,
                                   cv=cv, scorer=roc_auc_score)

    # fit and score
    gat.fit(epochs, y=y)
    gat.score(epochs)

    # # plot
    # gat.plot(vmin=0, vmax=1)
    # gat.plot_diagonal()
    return gat


def decoding_x_acc(epochs, log):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import SelectKBest, f_regression
    from sklearn.svm import SVC, SVR
    from sklearn.linear_model import LassoLars, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from imblearn.over_sampling import SMOTE
    from sklearn.model_selection import cross_val_score, train_test_split

    #clf = RandomForestClassifier(n_estimators=20)

    # clf = make_pipeline(StandardScaler(),  # z-score normalization
    #                     RandomForestClassifier())

    clf = make_pipeline(StandardScaler(),   # z-score normalization
                        LassoLars())

    cvs = 3

    sub_log = log[log.condition == 90]

    # all_x = epochs.get_data()
    # y = sub_log.Response.values
    # y[np.isnan(y)] = 0

    y = sub_log.RT.values
    all_x = epochs[~np.isnan(y)].get_data()
    y = y[~np.isnan(y)]

    times = epochs.times

    scores = list()
    for s in range(len(epochs.times)):
        x = all_x[:, :, s]

        cv_scores = list()
        for cv in range(cvs):
            x_tr, x_val, y_tr, y_val = train_test_split(x, y)
            # x_resam, y_resam = SMOTE(k_neighbors=3).fit_sample(x_tr, y_tr)
            # clf.fit(x_resam, y_resam)

            anova_filt = SelectKBest(f_regression, k=20)
            anova_reg = make_pipeline(anova_filt, clf)

            anova_reg.fit(x_tr, y_tr)
            s_score = anova_reg.score(x_val, y_val)
            cv_scores.append(s_score)
        scores.append(np.mean(cv_scores))
    plt.plot(times, scores), plt.ylim(-1, 1)



    # # GAT
    # y = np.array(log.loc[log.condition == 90, 'Response'])
    # y[np.isnan(y)] = 0.
    # epochs.events[:, 2] = y
    #
    # td = TimeDecoding(predict_mode='cross-validation', n_jobs=1, scorer=roc_auc, clf=clf)
    # td.fit(epochs, y=y)
    # td.score(epochs)
    # td.plot('Subject: ', epochs.info['subject_info'], chance=True)

    # cv = StratifiedKFold(y=y)  # do a stratified cross-validation
    #
    # gat = GeneralizationAcrossTime(predict_mode='cross-validation', n_jobs=1,
    #                                cv=cv, scorer=roc_auc_score)
    #
    # # fit and score
    # gat.fit(epochs, y=y)
    # gat.score(epochs)
    #
    # # # plot
    # gat.plot(vmin=0, vmax=1)
    # gat.plot_diagonal()
    return scores


def stats_decoding(scores):
    chance = 0.5
    x = scores - chance

    X = x[:, :, None] if x.ndim == 2 else x
    T_obs_, clusters, p_values, _ = spatio_temporal_cluster_1samp_test(
        X, out_type='mask', n_permutations=2 ** 12,
        n_jobs=n_jobs, connectivity=None)

    p_values_ = np.ones_like(X[0]).T
    for cluster, pval in zip(clusters, p_values):
        p_values_[cluster.T] = pval
    p_vals = np.squeeze(p_values_).T

    return p_vals


def load_decoding(subjects):
    all_scores = []

    for s in subjects:
        dec_file = op.join(study_path, 'results', 'decoding', '%s_gat.pickle' % s)
        if op.isfile(dec_file):
            with open(dec_file, 'rb') as handle:
                all_scores.append(pickle.load(handle))
    return all_scores


def plot_decoding(subj_scores):
    scores = [sc.scores_ for sc in subj_scores]
    times = subj_scores[0].train_times_['times']

    # Time Decoding
    tds = [np.diag(sc) for sc in scores]

    scores_arr = np.array(tds)
    mean_sco = np.mean(scores_arr, axis=0)
    sem_sco = sem(scores_arr, axis=0)
    p_vals_td = stats_decoding(scores_arr)

    t_masks = [[times < -0.7], [times > -0.7]]
    dec_fig, axes = plt.subplots(1, 2, sharey=True, gridspec_kw={'width_ratios': [1, 3]})
    for ix_m, m in enumerate(t_masks):
        axes[ix_m].plot(times[m], mean_sco[m])
        axes[ix_m].fill_between(times[m], mean_sco[m]+sem_sco[m], mean_sco[m]-sem_sco[m], alpha=0.3)
        axes[ix_m].set_ylim(0.45, 0.7)
        axes[ix_m].fill_between(times[m], 0.45, 0.7, where=p_vals_td[m] < 0.01, alpha=0.1, color='k')

    axes[0].set_xlim(-0.95, -0.7)
    axes[0].hlines(0.5, xmin=axes[0].get_xlim()[0], xmax=axes[0].get_xlim()[1], linestyles=':')
    axes[1].hlines(0.5, xmin=axes[1].get_xlim()[0], xmax=axes[1].get_xlim()[1], linestyles=':')
    axes[1].vlines(0, ymin=axes[1].get_ylim()[0], ymax=axes[1].get_ylim()[1], linestyles='--')
    axes[1].set_xlim(-0.7, 0.7)
    axes[0].set_ylabel('Classification Performance (AUC)')
    dec_fig.tight_layout(w_pad=0.1)
    dec_fig.savefig(op.join(study_path, 'figures', 'group_decoding_stats.eps'), format='eps', dpi=300)

    # GAT
    plt.style.use('ggplot')

    scores_gat = np.array(scores)
    mean_gat = np.mean(scores_gat, axis=0)
    p_vals_gat = stats_decoding(scores_gat)
    p_plot = mean_gat.copy()
    p_plot[p_vals_gat > 0.01] = np.nan

    s2_exp = np.where(times == 0)[0][0]
    s2_on = (np.abs(times + 0.7)).argmin()
    ticks = np.array([0, 64, s2_exp-2*s2_on, s2_exp-s2_on, s2_exp, s2_exp+s2_on, s2_exp+2*s2_on], subok=True)
    ticks_labels = [-0.25, 0, -0.5, -0.25, 0, 0.25, 0.5]

    gat_fig, ax = plt.subplots(1, 1)
    cax = ax.imshow(mean_gat, origin='lower', vmin=0.5, vmax=0.65, cmap='Greys')
    cax = ax.imshow(p_plot, origin='lower', vmin=0.5, vmax=0.65, cmap='plasma')
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
    gat_fig.savefig(op.join(study_path, 'figures', 'group_gat_stats.eps'), format='eps', dpi=300)

    plt.imshow(scores_arr, aspect='auto', interpolation='nearest', cmap='magma', vmin=0.5, vmax=0.8)
    plt.colorbar()
    plt.xticks(ticks, ticks_labels)


def plot_decoding_acc(subjects):
    all_scores = []

    for s in subjects:
        dec_file = op.join(study_path, 'results', 'decoding', '%s_gat_res.pickle' % s)
        if op.isfile(dec_file):
            with open(dec_file, 'rb') as handle:
                all_scores.append(pickle.load(handle))
    scores = [sc.scores_ for sc in all_scores]
    times = all_scores[0].times_['times']

    scores_arr = np.array(scores)
    mean_sco = np.mean(scores_arr, axis=0)
    sem_sco = sem(scores_arr, axis=0)
    p_vals_td = stats_decoding(scores_arr)

    t_masks = [[times < -0.7], [times > -0.7]]
    dec_fig, axes = plt.subplots(1, 2, sharey=True, gridspec_kw={'width_ratios': [1, 3]})
    for ix_m, m in enumerate(t_masks):
        axes[ix_m].plot(times[m], mean_sco[m])
        axes[ix_m].fill_between(times[m], mean_sco[m]+sem_sco[m], mean_sco[m]-sem_sco[m], alpha=0.3)
        axes[ix_m].set_ylim(0.45, 0.7)
        axes[ix_m].fill_between(times[m], 0.45, 0.7, where=p_vals_td[m] < 0.05, alpha=0.1, color='k')

    axes[0].set_xlim(-0.95, -0.7)
    axes[0].hlines(0.5, xmin=axes[0].get_xlim()[0], xmax=axes[0].get_xlim()[1], linestyles=':')
    axes[1].hlines(0.5, xmin=axes[1].get_xlim()[0], xmax=axes[1].get_xlim()[1], linestyles=':')
    axes[1].vlines(0, ymin=axes[1].get_ylim()[0], ymax=axes[1].get_ylim()[1], linestyles='--')
    axes[1].set_xlim(-0.7, 0.7)
    axes[0].set_ylabel('Classification Performance (AUC)')
    dec_fig.tight_layout(w_pad=0.1)


def behavior_graphs(log_df):
    def set_dif(row):
        if row['Order'] == 2.0:
            val = row['Standard'] - row['Comparison']
        else:
            val = row['Comparison'] - row['Standard']
        return val/1000

    log_df = log_df.drop(['Unnamed: 0', 'Unnamed: 0.1',  'Unnamed: 0.1.1' ,'a', 'b', 'c', 'd', 'e', 'f'], axis=1)
    log_df = set_dif_and_rt_exp(log_df)

    log_df = log_df.dropna()

    ratios_dat = np.array(log_df['Ratio'][log_df['Ratio'] != 1.0])
    plt.hist(ratios_dat, bins=np.linspace(0.25, 2.0, 8), align='left', rwidth=0.5)
    plt.xticks(np.linspace(0.25, 1.75, 7))
    plt.xlabel('Ratio')
    plt.ylabel('Nr of Trials')

    fig, axes = plt.subplots(2, 1, figsize=(15, 5), sharey=True)
    rts = log_df[['RT', 'condition']][log_df['Ratio'] != 1.0].dropna()
    for ix_c, (cond_name, cond_mark) in enumerate(zip(['S2 Shorter', 'S2 Longer'], [70, 90])):
        axes[ix_c].hist(rts['RT'][rts['condition'] == cond_mark], bins=np.linspace(0, 5, 41), alpha=0.5, align='left')
        axes[ix_c].legend([cond_name])
        axes[ix_c].set_xticks(np.linspace(0, 5, 21))

    plt.hist(log_df['dif'][log_df['dif'] != 0], bins=80, align='left')
    plt.xticks(np.linspace(-1.6, 1.6, 17))

    ratios = np.linspace(0.25, 1.75, 7)
    df_agg = log_df[['Ratio', 'Response']].groupby(log_df['Ratio'], as_index=False).agg(['mean', 'sem'])
    df_agg = df_agg['Response']
    plt.plot(ratios, df_agg['mean'].tolist())
    plt.errorbar(ratios, df_agg['mean'], yerr=df_agg['sem'].tolist())
    plt.xticks(ratios)
    plt.ylim((0, 1))

    for rt_type in ('RT', 'rt_exp'):
        fig_rt, axes = plt.subplots(2, 1, figsize=(18, 5), sharey=True, sharex=True)
        for ix_c, (cond_name, cond_mark) in enumerate(zip(['S2 Longer', 'S2 Shorter'], [90, 70])):
            c_data = log_df[log_df['condition'] == cond_mark]
            axes[ix_c].hist(c_data[rt_type], bins=np.linspace(-1, 5, 31), align='left', alpha=0.7, color='green')
            axes[ix_c].set_ylim([0, 250])
            axes[ix_c].set_xlim([-1.2, 5])
            axes[ix_c].legend([cond_name])
            axes[ix_c].vlines(0, ymin=0, ymax=250, linestyles='--')
        axes[1].set_xticks(np.linspace(-1, 5, 31))
        axes[1].set_xlabel(rt_type)

    plt.violinplot(c_data['rt_exp'].tolist(), vert=False)
    plt.xticks(np.linspace(-1, 4, 21))


def test_measures_corr(subjects):
    from pandas.plotting import scatter_matrix
    from statsmodels.sandbox.stats import multicomp
    from mne.stats import permutation_t_test
    from scipy.stats import wilcoxon
    plt.style.use('ggplot')

    log_df = pd.read_csv(op.join(study_path, 'tables', 's_trial_dat.csv'))
    log_df = log_df.drop(['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1', 'a', 'b', 'c', 'd', 'e', 'f'], axis=1)
    log_df = set_dif_and_rt_exp(log_df)

    clf_scores = load_decoding(subjects)
    scores = [sc.scores_ for sc in clf_scores]
    times = clf_scores[0].train_times_['times']
    tds = np.array([np.diag(sc) for sc in scores])
    mean_td = np.mean(tds, axis=0)
    mean_scores = np.mean(tds[:, (times > -0.2) & (times < 0.7)], axis=1)
    clf_peak = np.max(tds, axis=1)
    clf_lat = times[np.argmax(tds, axis=1)]

    fig, ax = plt.subplots(3, 1, figsize=(18, 5), sharex=True, sharey=False)
    for ix, (c_mark, color) in enumerate(zip([90, 70], ['blue', 'red'])):
        ax[0].hist(log_df.loc[log_df.condition == c_mark, 'rt_exp'], bins=np.linspace(-1, 5, 31), align='left', alpha=0.5, color=color)
        ax[1].hist(log_df.loc[log_df.condition == c_mark, 'dif'], bins=np.linspace(-1, 5, 61), align='left', alpha=0.5, color=color)
    ax[1].set_xticks(np.linspace(-1, 5, 31))
    ax[1].set_ylim([0, 200])
    ax[1].set_xlim([-1.2, 5])
    ax[1].legend(['S2 Longer', 'S2 Shorter'])
    ax[2].plot(times[times > -0.7], mean_td[times > -0.7])
    ax[2].set_ylim([0.5, 0.65])
    ax[0].vlines(0, ymin=0, ymax=200, linestyles='--')
    ax[1].vlines(0, ymin=0, ymax=200, linestyles='--')
    ax[2].vlines(0, ymin=0.5, ymax=0.65, linestyles='--')

    corr_dat = pd.read_csv(op.join(study_path, 'tables', 'pow_table.csv'))
    corr_dat = corr_dat.drop('Unnamed: 0', axis=1)
    conn_dat = pd.read_csv(op.join(study_path, 'tables', 'conn_x_subj_lon_sho.csv'), header=None)
    corr_dat['conn_lon'] = conn_dat[0]
    corr_dat['conn_sho'] = conn_dat[1]
    corr_dat['conn_dif'] = corr_dat['conn_lon'] - corr_dat['conn_sho']
    rt_lon = log_df[log_df.condition == 90][['RT']].groupby(log_df['subject'], as_index=False).agg([np.nanmedian])['RT']['nanmedian'].tolist()
    rt_sho = log_df[log_df.condition == 70][['RT']].groupby(log_df['subject'], as_index=False).agg([np.nanmedian])['RT']['nanmedian'].tolist()
    rt = log_df[['RT']].groupby(log_df['subject'], as_index=False).agg([np.nanmedian])['RT']['nanmedian'].tolist()
    corr_dat['rt_sho'] = rt_sho
    corr_dat['rt_lon'] = rt_lon
    corr_dat['rt'] = rt
    corr_dat['clf'] = mean_scores
    corr_dat['clf_pk'] = clf_peak
    corr_dat['clf_lat'] = clf_lat
    corr_dat['acc_lon'] = log_df[log_df.condition == 90][['Accuracy']].groupby(log_df['subject'], as_index=False).agg(np.nanmean)
    corr_dat['acc_sho'] = log_df[log_df.condition == 70][['Accuracy']].groupby(log_df['subject'], as_index=False).agg(np.nanmean)
    corr_dat['acc'] = log_df[['Accuracy']].groupby(log_df['subject'], as_index=False).agg(np.nanmean)
    corr_dat['rt_exp'] = log_df[log_df.condition == 70][['rt_exp']].groupby(log_df['subject'], as_index=False).agg(np.nanmedian)
    corr_dat['abs'] = np.abs(corr_dat['clf_lat'] - corr_dat['rt_exp'])
    corr_dat['pow_dif'] = np.abs(corr_dat['pow_lon'] - corr_dat['pow_sho'])
    corr_dat.corr()
    corr_dat.to_csv(op.join(study_path, 'tables', 'corr_table.csv'))
    # plt.style.use('classic')
    # scatter_matrix(corr_dat[['clf_pk', 'rt_lon', 'rt_sho']])
    # corr_dat.corr()

    n_perm = 10000
    # r_sho, p_sho = permutation_pearson(corr_dat['rt_sho'], corr_dat['clf_pk'], n_perm)
    # r_lon, p_lon = permutation_pearson(corr_dat['rt_lon'], corr_dat['clf_pk'], n_perm)

    r_dec_rt, p_dec_rt = permutation_pearson(corr_dat['rt'], corr_dat['clf_pk'], n_perm)
    r_conn_rt, p_conn_rt = permutation_pearson(corr_dat['rt'], corr_dat['conn_dif'], n_perm)
    r_tf_rt, p_tf_rt = permutation_pearson(corr_dat['rt'], corr_dat['pow_dif'], n_perm)

    r_dec_lon, p_dec_lon = permutation_pearson(corr_dat['acc_lon'], corr_dat['clf_pk'], n_perm)
    r_dec_sho, p_dec_sho = permutation_pearson(corr_dat['acc_sho'], corr_dat['clf_pk'], n_perm)

    r_conn_lon, p_conn_lon = permutation_pearson(corr_dat['acc_lon'], corr_dat['conn_lon'], n_perm)
    r_conn_sho, p_conn_sho = permutation_pearson(corr_dat['acc_sho'], corr_dat['conn_sho'], n_perm)

    r_tf_lon, p_tf_lon = permutation_pearson(corr_dat['acc_lon'], corr_dat['pow_lon'], n_perm)
    r_tf_sho, p_tf_sho = permutation_pearson(corr_dat['acc_sho'], corr_dat['pow_sho'], n_perm)

    all_p = [p_dec_rt, p_conn_rt, p_tf_rt, p_dec_lon, p_dec_sho, p_conn_lon, p_conn_sho, p_tf_lon, p_tf_sho]

    p_corr = multicomp.multipletests([p_dec_rt, p_conn_rt, p_tf_rt],
                                     method='fdr_bh')

    seaborn.regplot(corr_dat['rt'], corr_dat['clf_pk'], ci=None)
    plt.title('r = %0.3f  p = %0.3f' % (r_dec_rt, p_dec_rt))
    plt.savefig(op.join(study_path, 'figures', 'clf_pk_vs_RT_all.eps'), dpi=300)

    fig, ax = plt.subplots(1, 2, sharey=True, sharex=True)
    seaborn.regplot(corr_dat['acc_lon'], corr_dat['conn_dif'], ax=ax[0], ci=None)
    seaborn.regplot(corr_dat['acc_sho'], corr_dat['conn_dif'], ax=ax[1], ci=None)
    [ax[ix].set_title('r = {} p = {}' .format(round(r, 3), round(p, 3))) for ix, (r, p) in enumerate(zip([r_conn_lon, r_conn_sho],
                                                                                                         [p_conn_lon, p_conn_sho]))]
    [ax[ix].set_ylabel('Accuracy') for ix in range(len(ax))]
    [ax[ix].set_xlabel('Connectivity Difference (wSMI) \n %s' % lab) for ix, lab in enumerate(['S2 Longer', 'S2 Shorter'])]
    # ax[0].set_xlim(0, 0.05)
    fig.savefig(op.join(study_path, 'figures', 'conn_vs_acc.eps'), dpi=300)

    seaborn.regplot(corr_dat['pow_lon'], corr_dat['acc_lon'], ci=None)
    plt.title('r = %0.3f p = %0.3f' % (r_tf_lon, p_tf_lon))
    plt.savefig(op.join(study_path, 'figures', 'pow_lon_vs_acc_lon.eps'), dpi=300)

    seaborn.regplot(corr_dat['conn_lon'], corr_dat['acc_lon'], ci=None)


    plt.violinplot([corr_dat['conn_lon'], corr_dat['conn_sho']])

    T_obs, p_values, H0 = permutation_t_test(np.array(corr_dat['abs'], ndmin=2).T, n_permutations=10000, tail=1)
    corr_dat['abs'].plot(kind='box')
    plt.title('p = %.8f' % p_values[0])


do_decoding = True
do_tf = False

if __name__ == '__main__':
    all_s_dat = list()
    all_scores = dict(td=list(), gat=list())
    for subj in subjects:
        print('Subject: ', subj)

        # Load
        lon, sho, log = load_subj(subj)

        # Decoding
        if do_decoding:
            # gat = decoding_analysis(lon, sho)
            # dec_file = op.join(study_path, 'results', 'decoding', '%s_gat.pickle' % subj)
            # with open(dec_file, 'wb') as handle:
            #     pickle.dump(gat, handle, protocol=-1)

            if log.loc[log.condition == 90][['Response']].agg(np.nanmean).values > 0.1:
                gat_acc = decoding_x_acc(lon, log)
                dec_file = op.join(study_path, 'results', 'decoding', '%s_gat_res.pickle' % subj)
                with open(dec_file, 'wb') as handle:
                    pickle.dump(gat_acc, handle, protocol=-1)

        # Single trial time-frequency & connectivity
        if do_tf:
            for cond in [lon, sho]:
                log = tf_single_trial(cond, log)
                log = log.dropna()

        all_s_dat.append(log)

    # Log
    # all_log = pd.concat(all_s_dat, ignore_index=True)
    # all_log.to_csv(op.join(study_path, 'tables', 's_trial_dat.csv'))
    all_log = pd.read_csv(op.join(study_path, 'tables', 's_trial_dat.csv'))

    # Load decoding results
    #all_scores = load_decoding(subjects)


    # Plot group decoding
    # plot_decoding(all_scores)


    # Connectivity analysis
    #  all_dat = pd.read_csv(op.join(study_path, 'tables', 's_trial_dat.csv'))
    #  wpli_analysis_time(subj, all_dat, lon.info)
