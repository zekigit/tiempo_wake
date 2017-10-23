import numpy as np
import pandas as pd
from etg_scalp_info import study_path, subjects
from sklearn import decomposition, linear_model
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from scipy.io import loadmat
import os.path as op
import matplotlib.pyplot as plt
from scipy.stats import sem
from mpl_toolkits.axes_grid1 import ImageGrid

pd.set_option('display.expand_frame_repr', False)

conds = ['S2 Longer', 'S2 Shorter']


def load_subject(subject):
    # Load data
    mat_file = op.join(study_path, 'results', 'wsmi', 'mov_win', subject + '_mi_CSD_mov_win.mat')
    log_file = op.join(study_path, 'logs', 'exp_ok', subject + '_log_exp.csv')
    if op.isfile(mat_file):
        s_mat = loadmat(mat_file)['wSMI_Results']
        s_log = pd.read_csv(log_file)
        s_log = s_log[(s_log.Condition == 2.0)]

    if len(s_log) != s_mat.shape[2]:
        print('Unequal nr of trials and log')
        # continue

    # Select data
    diff_trials = s_log.Ratio != 1.0
    X = s_mat[:, :, diff_trials.values]
    cond = s_log.loc[s_log.Ratio != 1.0].condition.values
    cond[cond == 90] = 0
    cond[cond == 70] = 1

    acc = s_log.loc[(s_log.Ratio != 1.0) & (s_log.condition == 90)].Accuracy.values
    rt = s_log.loc[(s_log.Ratio != 1.0) & (s_log.condition == 90)].RT.values
    return X, cond, s_log


def run_analysis(subjects):
    model_clf = linear_model.LogisticRegression()
    model_reg = linear_model.LinearRegression()
    
    # model = linear_model.ElasticNet()
    # model = RandomForestClassifier()
    # model = SVC()

    n_pca = 44
    cvs = 3

    times = np.arange(-0.55, 0.55, 0.025)

    s_cond_scores = np.ndarray((len(subjects), len(times)))
    s_acc_scores = np.ndarray((len(subjects), len(times)))
    s_rt_scores = np.ndarray((len(subjects), len(times)))

    s_pca = np.ndarray((len(subjects), len(times), n_pca, len(conds)))
    
    for ix_s, s in enumerate(subjects):
        print('Processing Subject: ' + s)

        X, cond, s_log = load_subject(s)

        acc = s_log.loc[(s_log.Ratio != 1.0) & (s_log.condition == 90)].Accuracy.values
        rt = s_log.loc[(s_log.Ratio != 1.0) & (s_log.condition == 90)].RT.values
        
        # PCA
        pca_epo = np.ndarray((X.shape[1], n_pca, X.shape[2]))
        for ep in range(X.shape[2]):
            epo = X[:, :, ep].T
            pca = decomposition.PCA(n_components=n_pca)
            pca_epo[:, :, ep] = pca.fit_transform(epo)

        for c in range(len(conds)):
            s_pca[ix_s, :, :, c] = pca_epo[:, :, cond == c].mean(axis=2)

        pca_lon = pca_epo[:, :, cond == 0]
        
        # Decoding
        clf = make_pipeline(StandardScaler(), model_clf)
        reg = make_pipeline(StandardScaler(), model_reg)
        
        cond_scores = np.ndarray((len(times),))
        rt_scores = np.ndarray((len(times),))
        acc_scores = np.ndarray((len(times),))

        for t in range(len(times)):
            dat_all = pca_epo[t, :, :].T
            dat_lon = pca_lon[t, :, :].T

            cond_cv_scores = np.ndarray((cvs,))
            rt_cv_scores = np.ndarray((cvs,))
            acc_cv_scores = np.ndarray((cvs,))
            
            for cv in range(cvs):
                # Condition
                x_train, x_test, y_train, y_test = train_test_split(dat_all, cond)
                x_resam, y_resam = SMOTE().fit_sample(x_train, y_train)
                clf.fit(x_resam, y_resam)
                # clf.fit(x_tr, y_tr)
                cond_sco = clf.score(x_test, y_test)
                cond_cv_scores[cv] = cond_sco

                # RT
                dat_x_rt = dat_lon[~np.isnan(rt), :]
                rt_ok = rt[~np.isnan(rt)]
                x_train, x_test, y_train, y_test = train_test_split(dat_x_rt, rt_ok)
                reg.fit(x_train, y_train)
                rt_sco = reg.score(x_test, y_test)
                rt_cv_scores[cv] = rt_sco

                # Accuracy
                if sum(acc == 0) > 10:
                    y_train = np.arange(2)
                    while sum(y_train == 0) < 4:
                        x_train, x_test, y_train, y_test = train_test_split(dat_lon, acc, train_size=0.5, test_size=0.5, shuffle=True)
                    x_resam, y_resam = SMOTE(k_neighbors=2).fit_sample(x_train, y_train)
                    clf.fit(x_resam, y_resam)
                    # clf.fit(x_tr, y_tr)
                    acc_sco = clf.score(x_test, y_test)
                    acc_cv_scores[cv] = acc_sco
                else:
                    acc_cv_scores[cv] = 0.0
                
            cond_scores[t] = cond_cv_scores.mean()
            rt_scores[t] = rt_cv_scores.mean()
            acc_scores[t] = acc_cv_scores.mean()
        # plot_subj_results([cond_scores, rt_scores, acc_scores], pca_epo, times, cond)

        s_cond_scores[ix_s, :] = cond_scores
        s_rt_scores[ix_s, :] = rt_scores
        s_acc_scores[ix_s, :] = acc_scores

    all_scores = np.stack([s_cond_scores, s_rt_scores, s_acc_scores])        
    np.save(op.join(study_path, 'results', 'decoding', 'conn_decoding_Cond_Rt_Acc'), all_scores)
    np.save(op.join(study_path, 'results', 'wsmi', 'mov_win', 'conn_pca'), s_pca)


def plot_subj_results():
    scores = np.load(op.join(study_path, 'results', 'decoding', 'conn_decoding_Cond_Rt_Acc.npy'))
    pca = np.load(op.join(study_path, 'results', 'wsmi', 'mov_win', 'conn_pca.npy'))
    n_pca = pca.shape[2]
    times = np.arange(-0.55, 0.55, 0.025)
    n_subj = pca.shape[0]

    s_fig, s_axes = plt.subplots(n_subj, 4, figsize=(20, 10), sharex=True)
    for s in range(n_subj):
        s_axes[s, 0].plot(times, scores[0, s, :])
        s_axes[s, 0].set_ylim((0.4, 0.7))
        s_axes[s, 0].hlines(0.5, xmin=times[0], xmax=times[-1])
        s_axes[s, 0].set_yticks([0.5, 0.7])
        s_axes[s, 0].set_ylabel('{}' .format(s+1))

        s_axes[s, 1].plot(times, scores[1, s, :])
        s_axes[s, 1].set_ylim((-0.2, 0.4))
        s_axes[s, 1].hlines(0, xmin=times[0], xmax=times[-1])
        s_axes[s, 1].set_yticks([0, 0.2])

        s_axes[s, 2].plot(times, scores[2, s, :])
        s_axes[s, 2].set_ylim((0.4, 0.7))
        s_axes[s, 2].hlines(0.5, xmin=times[0], xmax=times[-1])
        s_axes[s, 2].set_yticks([0.5, 0.7])

        for comp in range(n_pca):
            s_axes[s, 3].plot(times, pca[s, :, comp, 0], color='orange', linestyle='-')

        s_axes[s, 3].set_ylim((-2, 3))
        s_axes[s, 3].hlines(0, xmin=times[0], xmax=times[-1])
        s_axes[s, 3].set_yticks([0, 2])

    for ix_t, tit in enumerate(['Condition', 'RT', 'Accuracy', 'PCA']):
        s_axes[0, ix_t].set_title(tit)
    plt.show()

    acc_subj = np.array([ix for ix in range(n_subj) if (scores[2, ix, 1] != 0.0)])

    avg_fig, axes = plt.subplots(1, 3)
    for ix, ax in enumerate(axes):
        if ix == 2:
            ax.plot(times, scores[ix, acc_subj, :].mean(axis=0))
            ax.fill_between(times,
                            scores[ix, acc_subj, :].mean(axis=0) + sem(scores[ix, acc_subj, :], axis=0),
                            scores[ix, acc_subj, :].mean(axis=0) - sem(scores[ix, acc_subj, :], axis=0), alpha=0.3)

        ax.plot(times, scores[ix, :, :].mean(axis=0))
        ax.fill_between(times,
                        scores[ix, :, :].mean(axis=0)+sem(scores[ix, :, :], axis=0),
                        scores[ix, :, :].mean(axis=0)-sem(scores[ix, :, :], axis=0), alpha=0.3)
    axes[0].set_ylim((0.4, 0.7))
    axes[0].hlines(0.5, xmin=times[0], xmax=times[-1])
    axes[1].set_ylim((-1, 0.4))
    axes[1].hlines(0, xmin=times[0], xmax=times[-1])
    axes[2].set_ylim((0.4, 0.7))
    axes[2].hlines(0.5, xmin=times[0], xmax=times[-1])
    for ix_t, tit in enumerate(['Condition', 'RT', 'Accuracy']):
        axes[ix_t].set_title(tit)
    plt.show()


def pca_results(subjects):
    pca = np.load(op.join(study_path, 'results', 'wsmi', 'mov_win', 'conn_pca.npy'))  # subj x times x comps x cond (lon, sho)
    pca_lon = pca[:, :, :, 0]
    pca_sho = pca[:, :, :, 1]
    times = np.arange(-0.55, 0.55, 0.025)

    #c_zero = [0 if pca_lon[s, times==0.0, 0] > pca_lon[s, times==0], 0, 0]

    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
    for ix_cond, cond in enumerate(conds):
        for comp in (0, 1):
            for s in range(len(subjects)):
                axes[comp, ix_cond].plot(times, pca[s, :, comp, ix_cond])


def time_conn_analysis(subjects):
    times = np.arange(-0.55, 0.55, 0.025)
    n_pca = 2
    s_pca = np.ndarray((len(subjects), len(times), n_pca))
    all_log = list()
    s_vars = np.ndarray((len(subjects), n_pca))
    s_wsmi = list()
    for ix_s, s in enumerate(subjects):
        X, cond, sub_log = load_subject(s)
        x = X[:, :, cond == 0]

        # pca_lon = np.ndarray((X.shape[1], n_pca, X.shape[2]))
        # pca = decomposition.PCA(n_components=n_pca)
        #
        # for ep in range(x.shape[2]):
        #     epo = x[:, :, ep].T
        #     pca = decomposition.PCA(n_components=n_pca)
        #     pca_lon[:, :, ep] = pca.fit_transform(epo)

        con_avg = x.mean(axis=2).T
        s_wsmi.append(con_avg)
        pca = decomposition.PCA(n_components=n_pca)
        pca_avg = pca.fit_transform(con_avg)
        s_pca[ix_s, :, :] = pca_avg
        s_vars[ix_s, :] = pca.explained_variance_

        all_log.append(sub_log)

    zero_time = (times > -0.01) & (times < 0.01)
    subj_zero = [0 if (float(s_pca[ix_s, zero_time, 0]) > float(s_pca[ix_s, zero_time, 1])) else 1 for ix_s, s in enumerate(subjects)]

    comp_dat = np.ndarray((len(subjects), len(times)))
    var_dat = np.empty(len(subjects))
    for ix_s, s in enumerate(subj_zero):
        # plt.plot(times, s_pca[ix_s, :, s])
        comp_dat[ix_s, :] = s_pca[ix_s, :, s]
        var_dat[ix_s] = s_vars[ix_s, s]

    comp_pk = np.abs(times[np.argmax(comp_dat, axis=1)])

    from mne.stats import permutation_cluster_1samp_test
    T_obs_, clusters, p_values, _ = permutation_cluster_1samp_test(comp_dat, threshold=None, n_permutations=1000, tail=0)
    good_clust = clusters[0]
    good_clust_inds = np.arange(len(times))[good_clust]

    from scipy.stats import sem, pearsonr, zscore

    plt.style.use('ggplot')
    fig, ax = plt.subplots(1, 1)
    ax.plot(times, comp_dat[:, :].mean(axis=0))
    ax.fill_between(times, comp_dat[:, :].mean(axis=0) - sem(comp_dat[:, :]),
                    comp_dat[:, :].mean(axis=0) + sem(comp_dat[:, :]), alpha=0.2)
    ax.set_ylim(-0.6, 0.8)
    ax.fill_between(times[good_clust], -0.6, 0.8, alpha=0.1, color='k')
    ax.vlines(0, ymin=-0.6, ymax=0.8, linestyles='--')
    ax.set_title('Mean variance explained: %0.2f' % var_dat.mean())
    ax.set_ylabel('PC (wSMI)')
    fig.savefig(op.join(study_path, 'figures', 'PCA_conn.eps'), format='eps', dpi=300)

    s_wsmi = np.array(s_wsmi)
    wSMI_gr_avg = s_wsmi.mean(axis=0)
    pc_gr_avg = comp_dat.mean(axis=0)

    corr_pc = np.empty(wSMI_gr_avg.shape[1])
    for pair in range(len(corr_pc)):
        corr_pc[pair] = pearsonr(wSMI_gr_avg[:, pair], pc_gr_avg)[0]

    from scipy.io import savemat
    savemat(op.join(study_path, 'results', 'wsmi', 'mov_win', 'wsmi_pc_corr.mat'), {'wsmi_pc_corr': corr_pc})

    plt.plot(times, zscore(wSMI_gr_avg[:, corr_pc > 0.8]))

    # behav
    conn_z = zscore(s_wsmi, axis=1)

    top_pairs = s_wsmi[:, :, corr_pc > 0.4][:, good_clust_inds, :]
    s_top = np.mean(top_pairs, axis=(1, 2))

    good_pca = comp_dat[:, good_clust_inds].mean(axis=1)

    from eeg_etg_fxs import set_dif_and_rt_exp
    log = pd.concat(all_log)
    log = log[log.condition != 80]
    log = set_dif_and_rt_exp(log)
    log['dif'].plot(kind='hist', bins=200)
    plt.xticks(np.linspace(-1.5, 1.5, 31))

    s_log = log[log.condition == 90].groupby('subject')[['Accuracy', 'RT']].agg(np.nanmean)
    s_log['con'] = s_top
    s_log['pca'] = good_pca
    s_log['pca_pk'] = np.abs(comp_pk)
    s_log.corr()

    from seaborn import regplot
    regplot(s_log['Accuracy'], s_log['pca_pk'])


if __name__ == '__main__':
    run_analysis(subjects)
    plot_subj_results()
    


