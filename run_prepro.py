import mne
import numpy as np
import os.path as op
from etg_scalp_info import study_path, data_path, log_path, sujetos, sesiones, bad_channs, ch_to_remove, marks, n_jobs
from eeg_etg_fxs import read_log_file, check_events, durations_from_log, create_events, add_event_condition


def run_prepro(subj):
    ses_x_subj = sesiones[subj]
    for ses in ses_x_subj:
        archivo = 'etg_su{}_se{}' .format(subj, ses)
        full_path = op.join(data_path, archivo + '.bdf')
        if not op.isfile(full_path):
            print('The session {} of subject {} is missing' .format(ses, subj))
            continue
        print('Processing Subject {} - Session {}' .format(subj, ses))

        raw = mne.io.read_raw_edf(full_path, preload=True)
        # raw.plot(n_channels=32, duration=2)
        montage = mne.channels.read_montage('biosemi128')
        head_channs = montage.ch_names[:128]
        head_channs.append('STI 014')

        raw.pick_channels(ch_names=head_channs)
        raw.set_montage(montage)

        bads = bad_channs[subj]
        raw.info['bads'] = bads

        picks_eeg = mne.pick_types(raw.info, eeg=True, meg=False, eog=False, exclude='bads')

        raw, ref = mne.io.set_eeg_reference(raw, ref_channels=None)

        raw.apply_proj()

        raw.filter(l_freq=0.1, h_freq=40)
        raw.plot(n_channels=32, duration=5)

        # ICA
        reject = {'eeg': 120e-6}
        n_components = 20
        method = 'infomax'
        decim = 4
        ica = mne.preprocessing.ICA(n_components=n_components, method=method)
        ica.fit(raw, picks=picks_eeg, decim=decim, reject=reject)
        print(ica)
        ica.plot_components()
        comp_to_delete = [int(x) for x in input('Components to delete (ej: 1 2 3): ').split()]
        if comp_to_delete:
            ica.exclude = comp_to_delete
            raw = ica.apply(raw)

        raw.interpolate_bads()

        # Log
        log_filename = op.join(log_path, archivo[:4] + 'scalp_' + archivo[4:] + '.mat')

        log = read_log_file(log_filename)

        # Events
        events = mne.find_events(raw)
        events = check_events(events)
        durations = durations_from_log(log)

        new_events = create_events(events, durations, raw.info['sfreq'])
        raw.add_events(new_events)
        events_updated = mne.find_events(raw, shortest_event=1)
        events_updated = check_events(events_updated)
        events_updated = add_event_condition(events_updated, log)
        mne.viz.plot_events(events_updated)

        # Epoch
        t_min = -0.25
        t_max = 1
        baseline = (None, 0)
        # event_id = dict(s1=1, s2=2, exp=3)
        event_id = marks
        epoch = mne.Epochs(raw, events_updated, event_id=event_id,
                           tmin=t_min, tmax=t_max, baseline=baseline,
                           reject=reject, preload=True)
        # epoch.plot(n_epochs=10)

        # Downsample
        epoch.resample(256, n_jobs=n_jobs)
        epoch.save(op.join(study_path, 'fif/', archivo + 'avg_subcond-epo.fif'))
        print('\n' * 4)


# sujetos = '9'

for subject in sujetos:
    run_prepro(subject)


