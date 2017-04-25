import numpy as np
import scipy.io as spio
import pandas as pd


def check_events(events):
    ev = 0
    while events[ev][2] != 1:
        events = np.delete(events, ev, 0)
        print('Deleted one event')
    if len(events) != 336:
        print('Events: {}' .format(len(events)))
    events[events == 258] = 3
    bad_marks = []
    for ix, ev in enumerate(events):
        if ev[2] not in (1, 2, 3, 255):
            bad_marks.append(ix)
    events = np.delete(events, [bad_marks], 0)
    return events


def durations_from_log(log):
    durations = []
    for idx, trial in log.iterrows():
        if trial['Order'] == 1:
            durations.append(trial['Standard'])
            durations.append(trial['Comparison'])
        else:
            durations.append(trial['Comparison'])
            durations.append(trial['Standard'])
        if not np.isnan(log['Response'].iloc[idx]):
            durations.append('Resp')
    return durations


def durations_from_log_ieeg(log):
    durations = []
    for idx, trial in log.iterrows():
        if trial['Order'] == 1:
            durations.append(trial['Standard'])
            durations.append(trial['Comparison'])
        else:
            durations.append(trial['Comparison'])
            durations.append(trial['Standard'])
    return durations


def create_events(events, durations, fs):
    if len(events) == len(durations):
        new_events = [ev[0] + (durations[idx-1]*fs)/1000 for idx, ev in enumerate(events) if ev[2] == 2]
        new_events = np.array(new_events, dtype=int)
        zeros = np.zeros(len(new_events))
        marks = np.repeat(3, len(new_events))
        new_events = np.vstack((new_events, zeros, marks))
        new_events = np.transpose(new_events)
        return new_events
    else:
        print('Unequal size of events and log')


def add_event_condition(events_updated, log):
    exp_types = []
    conds = []
    for idx, tr in log.iterrows():
        conds.extend([tr['Condition']])
        exp_types.extend([8 if tr['Standard'] == tr['Comparison']
                          else 7 if tr['Order'] == 1 and tr['Standard'] > tr['Comparison']
                          else 7 if tr['Order'] == 2 and tr['Standard'] < tr['Comparison']
                          else 9])
    ev_conditions = [exp_t if cond == 1 else exp_t + 10 for exp_t, cond in zip(exp_types, conds)]
    print('Number of trials : {}' .format(len(ev_conditions)))

    indices_1 = [i for i, x in enumerate(events_updated[:, 2]) if x == 1]
    indices_2 = [i for i, x in enumerate(events_updated[:, 2]) if x == 2]
    indices_3 = [i for i, x in enumerate(events_updated[:, 2]) if x == 3]
    print('Nr of events type 1: {} ' .format(len(indices_1)))
    print('Nr of events type 2: {} '.format(len(indices_2)))
    print('Nr of events type 3: {} '.format(len(indices_3)))

    if len(indices_1) == len(indices_2) == len(indices_3) == len(conds) == len(exp_types):
        print('Nr of trials OK')
    else:
        print('Nr of trials UNEQUAL')

    for idx, cond in enumerate(conds):
        events_updated[indices_3[idx], 2] = exp_types[idx]
        if cond == 2:
            events_updated[indices_1[idx], 2] *= 10
            events_updated[indices_2[idx], 2] *= 10
            events_updated[indices_3[idx], 2] *= 10
    return events_updated


def read_log_file(log_filename):
    log = spio.loadmat(log_filename)
    log = log['Results']
    log = pd.DataFrame(data=log, columns=['Trial Nr', 'Block Nr', 'S1 Onset', 'S1 Onset bk', 'S2 Onset',
                                          'S2 Onset bk', 'Condition', 'Ratio', 'Standard', 'Comparison',
                                          'Gap', 'Order', 'Resp Onset', 'RT', 'Key', 'Response', 'Accuracy'])
    log = log[['Trial Nr', 'Block Nr', 'Condition', 'Ratio', 'Standard', 'Comparison',
               'Gap', 'Order', 'RT', 'Response', 'Accuracy']]
    if len(log) != 112:
        print('LOG does not have 112 trials')
    return log


