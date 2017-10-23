import mne
import pandas as pd
import os.path as op
from etg_scalp_info import study_path, subjects, n_jobs, reject
from eeg_etg_fxs import check_events, add_event_condition, add_event_tr_id, make_exp_baseline, durations_from_log, add_s2_off_events
pd.set_option('display.expand_frame_repr', False)

marks_s = {'s1_sub': 1, 's2_sub': 2, 's1_sup': 10, 's2_sup': 20}
marks_exp = {'exp_sub_sho': 7, 'exp_sub_eq': 8, 'exp_sub_lon': 9,
             'exp_sup_sho': 70, 'exp_sup_eq': 80, 'exp_sup_lon': 90}
marks_p = {'pre_trial_sub': 2, 'pre_trial_sup': 20}
marks_off = {'off_sub_sho': 1001, 'off_sub_lon': 1002, 'off_sup_sho': 1011, 'off_sup_lon': 1012}

for subj in subjects:
    eeg_file = op.join(study_path, 'fif', '{}_prep-raw.fif' .format(subj))
    raw = mne.io.read_raw_fif(eeg_file, preload=True)

    log_file = op.join(study_path, 'logs', '{}_prep-raw.csv' .format(subj))
    log = pd.read_csv(log_file)

    events = mne.find_events(raw)
    events = check_events(events)

    durations = durations_from_log(log)
    new_events = add_s2_off_events(events, durations, raw.info['sfreq'])
    raw.add_events(new_events)

    events = mne.find_events(raw)
    events = check_events(events)
    events, log = add_event_condition(events, log)
    # events = check_events(events)

    # mne.viz.plot_events(events)

    events_tr_id = add_event_tr_id(events)
    #
    # s1_s2 = mne.Epochs(raw, events, event_id=marks_s, tmin=-0.25, tmax=1, baseline=(None, 0), reject=reject, preload=True)
    # s1_s2.resample(256, n_jobs=n_jobs)
    # s1_s2.save(op.join(study_path, 'epo', '{}_s1_s2-epo.fif' .format(subj)))
    #
    # exp = mne.Epochs(raw, events, event_id=marks_exp, tmin=-0.7, tmax=0.7, baseline=None, reject=reject, preload=True)
    #
    # pre_trial = mne.Epochs(raw, events, event_id=marks_p, tmin=-0.25, tmax=0, baseline=None, reject=reject, preload=True)
    #
    # exp_epochs, exp_ids = make_exp_baseline(exp, pre_trial, events_tr_id, log, marks_exp)
    # exp_epochs.resample(256, n_jobs=n_jobs)
    # exp_epochs.save(op.join(study_path, 'epo', '{}_exp-epo.fif' .format(subj)))
    # log_ok = log.iloc[exp_ids]
    # log_ok.to_csv(op.join(study_path, 'logs', 'exp_ok', '{}_log_exp.csv' .format(subj)))

    s2_off = mne.Epochs(raw, events, event_id=marks_off, tmin=-0.25, tmax=0.75, baseline=(None, 0), reject=reject, preload=True)
    s2_off.resample(256, n_jobs=n_jobs)
    s2_off.save(op.join(study_path, 'epo', '{}_s2_off-epo.fif' .format(subj)))

    off_events = s2_off.selection
    off_trials = events_tr_id[off_events, 3]
    log_off = log.loc[off_trials]
    log_off.to_csv(op.join(study_path, 'logs', 'off_ok', '{}_log_off.csv' .format(subj)))

