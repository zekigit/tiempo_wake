import mne
import numpy as np
import pandas as pd
from etg_scalp_info import study_path, subjects, marks_off
import os.path as op
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
pd.set_option('display.expand_frame_repr', False)


def load_subj(subj):
    eeg_file = op.join(study_path, 'epo', '%s_s2_off-epo.fif' % subj)
    epochs = mne.read_epochs(eeg_file, preload=True)

    log_file = op.join(study_path, 'logs', 'off_ok', '%s_log_off.csv' % subj)
    log = pd.read_csv(log_file)
    return epochs, log


def erp_analysis(subjects):
    from mne.stats import spatio_temporal_cluster_test
    from mne.channels import read_ch_connectivity
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from scipy import stats as stats

    all_evo = {'off_sup_lon': list(), 'off_sup_sho': list()}
    all_log = list()
    condition_names = ['S2 Longer', 'S2 Shorter']
    n_subjects = len(subjects)

    # Load
    for subj in subjects:
        epochs, log = load_subj(subj)
        epochs = epochs[['off_sup_lon', 'off_sup_sho']]
        log = log.loc[(log.Condition == 2.0) & (log.Ratio != 1.0)]

        corr_log = list()
        for k in all_evo.keys():
            c = marks_off[k]
            sub_log = log[log.condition == c]
            sub_log['epo_ix'] = np.arange(len(sub_log))
            # corr_ix = sub_log['epo_ix'].loc[(sub_log.condition == c) & (sub_log.Accuracy == 1.0)].values
            # sub_log = sub_log.loc[(sub_log.condition == c) & (sub_log.Accuracy == 1.0)]
            corr_ix = sub_log['epo_ix']
            all_evo[k].append(epochs[k][corr_ix].average())
            corr_log.append(sub_log)
            print(k, c, len(corr_ix))

        all_log.append(pd.concat(corr_log))

    all_log = pd.concat(all_log)
    all_log.groupby('condition')[['condition']].agg(np.count_nonzero).plot(kind='bar')

    # Plot
    evoked = {k: mne.combine_evoked(all_evo[k], weights='nave') for k in all_evo.keys()}
    mne.viz.plot_evoked_topo([evoked[ev] for ev in evoked.keys()])


    # Stats
    connectivity, ch_names = read_ch_connectivity('/Users/lpen/Documents/MATLAB/Toolbox/fieldtrip-20170628/template/neighbours/biosemi128_neighb.mat')

    #threshold = {'start': 5, 'step': 0.5}
    threshold = None
    p_threshold = 0.001
    t_threshold = -stats.distributions.t.ppf(p_threshold / 2., n_subjects - 1)

    x = {k: np.array([all_evo[k][ix_s].data for ix_s in range(len(subjects))]) for k in sorted(all_evo.keys())}
    x = [np.transpose(x[k], (0, 2, 1)) for k in sorted(x.keys())]

    t_obs, clusters, p_values, _ = spatio_temporal_cluster_test(x, n_permutations=1000,
                                                                threshold=t_threshold, tail=0,
                                                                n_jobs=2,
                                                                connectivity=connectivity,
                                                                )

    p_val = 0.01
    good_cluster_inds = np.where(p_values < p_val)[0]
    print(good_cluster_inds)
    print(len(good_cluster_inds))

    # configure variables for visualization
    times = evoked['off_sup_lon'].times * 1e3
    colors = 'r', 'b',
    linestyles = '-', '-',

    # grand average as numpy arrray
    grand_ave = np.array(x).mean(axis=1)

    # get sensor positions via layout
    pos = mne.find_layout(evoked['off_sup_lon'].info).pos

    # loop over significant clusters
    for i_clu, clu_idx in enumerate(good_cluster_inds):
        # unpack cluster infomation, get unique indices
        time_inds, space_inds = np.squeeze(clusters[clu_idx])
        ch_inds = np.unique(space_inds)
        time_inds = np.unique(time_inds)

        # get topography for F stat
        f_map = t_obs[time_inds, ...].mean(axis=0)

        # get signals at significant sensors
        signals = grand_ave[..., ch_inds].mean(axis=-1)
        sig_times = times[time_inds]

        # create spatial mask
        mask = np.zeros((f_map.shape[0], 1), dtype=bool)
        mask[ch_inds, :] = True

        # initialize figure
        fig, ax_topo = plt.subplots(1, 1, figsize=(10, 3))
        title = 'Cluster #{0}'.format(i_clu + 1)
        fig.suptitle(title, fontsize=14)

        # plot average test statistic and mark significant sensors
        image, _ = mne.viz.plot_topomap(f_map, pos, mask=mask, axes=ax_topo,
                                        cmap='magma', vmin=np.min, vmax=np.max)

        # advanced matplotlib for showing image with figure and colorbar
        # in one plot
        divider = make_axes_locatable(ax_topo)

        # add axes for colorbar
        ax_colorbar = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(image, cax=ax_colorbar)
        ax_topo.set_xlabel('Averaged F-map ({:0.1f} - {:0.1f} ms)'.format(
            *sig_times[[0, -1]]
        ))

        # add new axis for time courses and plot time courses
        ax_signals = divider.append_axes('right', size='300%', pad=1.5)
        for signal, name, col, ls in zip(signals, condition_names, colors,
                                         linestyles):
            ax_signals.plot(times, signal, color=col, linestyle=ls, label=name)

        # add information
        ax_signals.axvline(0, color='k', linestyle=':', label='stimulus onset')
        ax_signals.set_xlim([times[0], times[-1]])
        ax_signals.set_ylim([-10e-7, 20e-7])
        ax_signals.set_xlabel('time [ms]')
        ax_signals.set_ylabel('Amplitude')
        ax_signals.hlines(0, xmin=times[0], xmax=times[-1], linestyles='--')

        # plot significant time range
        ymin, ymax = ax_signals.get_ylim()
        ax_signals.fill_betweenx((ymin, ymax), sig_times[0], sig_times[-1],
                                 color='orange', alpha=0.3)
        ax_signals.legend(loc='lower right')
        ax_signals.set_ylim(ymin, ymax)

        # clean up viz
        mne.viz.tight_layout(fig=fig)
        fig.subplots_adjust(bottom=.05)
        plt.show()
        fig.savefig(op.join(study_path, 'figures', 'ERP_off_ckust_{}.eps' .format(i_clu)), format='eps', dpi=300)

    # Cluster Amplitude
    # t_mask = np.arange(len(times))[(times > 300) & (times < 400)]
    # sig_amp = {k: np.array([x[ix_c][ix_s, t_mask, :][:, ch_inds].mean() for ix_s, s in enumerate(subjects)]) for ix_c, k in enumerate(['lon', 'sho'])}
    sig_amp = {k: np.array([x[ix_c][ix_s, time_inds, :][:, ch_inds].mean() for ix_s, s in enumerate(subjects)]) for ix_c, k in enumerate(['lon', 'sho'])}

    subj_cond = all_log.groupby('subject')[['RT', 'Accuracy']].agg(np.mean)
    subj_cond['acc_lon'] = all_log[all_log.condition == 90].groupby('subject')[['Accuracy']].agg(np.mean)
    subj_cond['acc_sho'] = all_log[all_log.condition == 70].groupby('subject')[['Accuracy']].agg(np.mean)
    subj_cond['amp_lon'] = sig_amp['lon']
    subj_cond['amp_sho'] = sig_amp['sho']
    subj_cond['amp_dif'] = subj_cond['amp_sho'] - subj_cond['amp_lon']

    subj_cond.corr(method='pearson')

    from seaborn import regplot
    from eeg_etg_fxs import permutation_pearson

    r_sho, p_sho = permutation_pearson(subj_cond['amp_dif'].values, subj_cond['acc_sho'].values, 10000)
    r_lon, p_lon = permutation_pearson(subj_cond['amp_dif'].values, subj_cond['acc_lon'].values, 10000)

    plt.style.use('ggplot')
    fig, axes = plt.subplots(1, 2, sharey=True, sharex=True)
    for ix, (r, p, c) in enumerate(zip([r_lon, r_sho], [p_lon, p_sho], ['acc_lon', 'acc_sho'])):
        regplot(subj_cond['amp_dif'], subj_cond[c], ci=None, ax=axes[ix])
        axes[ix].set_title('r = %0.3f   p = %0.3f' % (r, p))
    fig.savefig(op.join(study_path, 'figures', 'ERP_diff_acc.eps'), format='eps', dpi=300)

    mne.viz.plot_compare_evokeds([evoked[ev] for ev in evoked.keys()], picks=2)
    plt.savefig(op.join(study_path, 'figures', 'ERP_diff_A4.eps'), format='eps', dpi=300)
