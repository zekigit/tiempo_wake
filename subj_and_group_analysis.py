import mne
import os.path as op
import numpy as np
from mne.time_frequency import tfr_morlet
from etg_scalp_info import study_path, subjects, n_jobs, rois
from eeg_etg_fxs import permutation_t_test
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.expand_frame_repr', False)


def load_subj(subj):
    # Baseline Parameters
    bs_min = -0.95  # Pre-s2 baseline
    bs_max = -0.75

    epochs_file = op.join(study_path, 'epo', '{}_exp-epo.fif' .format(subj))
    epochs = mne.read_epochs(epochs_file, preload=True)
    if (len(epochs['exp_sup_lon']) > 30) and (len(epochs['exp_sup_sho']) > 20):
        epochs.apply_baseline(baseline=(bs_min, bs_max))
        exp_lon = epochs['exp_sup_lon']
        exp_sho = epochs['exp_sup_sho']

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


if __name__ == '__main__':
    all_s_dat = list()
    for subj in subjects:
        print('Subject: ', subj)
        lon, sho, log = load_subj(subj)

        # Single trial time-frequency
        for cond in [lon, sho]:
            log = tf_single_trial(cond, log)
            log_ok = log.dropna()
        all_s_dat.append(log_ok)
    all_dat = pd.concat(all_s_dat, ignore_index=True)
    all_dat.to_csv(op.join(study_path, 'tables', 's_trial_dat.csv'))
