import mne
import pandas as pd
import os.path as op
from etg_scalp_info import study_path, data_path, log_path, subjects, sessions, bad_channs, n_jobs
from eeg_etg_fxs import read_log_file, check_events, durations_from_log, create_events, check_events_and_log

subj = '16'


def run_prepro(subj):
    ses_x_subj = sessions[subj]

    montage = mne.channels.read_montage('biosemi128')
    head_channs = montage.ch_names[:128]
    head_channs.append('STI 014')

    all_ses = list()
    all_log = list()
    for ses in ses_x_subj:
        # EEG
        file = 'etg_su{}_se{}' .format(subj, ses)
        full_path = op.join(data_path, file + '.bdf')
        if not op.isfile(full_path):
            print('The session {} of subject {} is missing' .format(ses, subj))
            continue
        print('Processing Subject {} - Session {}' .format(subj, ses))
        raw = mne.io.read_raw_edf(full_path, preload=True)

        # LOG
        log_filename = op.join(log_path, file[:4] + 'scalp_' + file[4:] + '.mat')
        log = read_log_file(log_filename)
        log['session'] = ses
        log['subject'] = subj
        all_log.append(log)

        # Events
        events = mne.find_events(raw)
        events = check_events(events)
        durations = durations_from_log(log)

        new_events = create_events(events, durations, raw.info['sfreq'])
        raw.add_events(new_events)

        raw.pick_channels(ch_names=head_channs)

        all_ses.append(raw)

    if len(all_ses) > 1:
        raw_cat = mne.concatenate_raws(all_ses, preload=True)
        logs = pd.concat(all_log, ignore_index=True)
    else:
        raw_cat = raw
        logs = log

    del all_ses
    events = mne.find_events(raw_cat)
    events = check_events(events)
    # mne.viz.plot_events(events)

    try:
        check_events_and_log(events, logs)
    except ValueError as err:
        print(err, ' - subj:', subj)
        return

    raw_cat.set_montage(montage)

    bads = bad_channs[subj]
    raw_cat.info['bads'] = bads

    picks_eeg = mne.pick_types(raw_cat.info, eeg=True, meg=False, eog=False, exclude='bads')

    raw, ref = mne.io.set_eeg_reference(raw_cat, ref_channels=None)
    raw.apply_proj()

    raw.filter(l_freq=0.1, h_freq=40, filter_length='auto', l_trans_bandwidth='auto', h_trans_bandwidth='auto', n_jobs=n_jobs)
    raw.plot(events, n_channels=128, duration=20, scalings={'eeg': 40e-6})

    # ICA
    reject = {'eeg': 250e-6}
    n_components = 20
    method = 'extended-infomax'
    decim = 4
    ica = mne.preprocessing.ICA(n_components=n_components, method=method)
    ica.fit(raw_cat, picks=picks_eeg, decim=decim, reject=reject)
    print(ica)
    ica.plot_components()
    comp_to_delete = [int(x) for x in input('Components to delete (ej: 1 2 3): ').split()]
    if comp_to_delete:
        ica.plot_properties(raw_cat, picks=comp_to_delete)
        comp_to_delete = [int(x) for x in input('Confirm components to delete (ej: 1 2 3): ').split()]
        if comp_to_delete:
            ica.exclude = comp_to_delete
            raw_cat = ica.apply(raw_cat)

    raw_cat.interpolate_bads()

    raw_cat.save(op.join(study_path, 'fif',  '{}_prep-raw.fif' .format(subj)), overwrite=True)
    logs.to_csv(op.join(study_path, 'logs',  '{}_prep-raw.csv' .format(subj)))
    del raw_cat
    print('\n' * 4)

if __name__ == '__main__':
    for subject in subjects:
        run_prepro(subject)
