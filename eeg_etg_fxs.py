import numpy as np
import scipy.io as spio
import pandas as pd
import mne
from scipy.stats import ttest_rel
from mpl_toolkits.axes_grid1 import ImageGrid
import os.path as op


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
    ev_conditions = [exp_t if cond == 1 else exp_t * 10 for exp_t, cond in zip(exp_types, conds)]
    log['condition'] = ev_conditions
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
    return events_updated, log


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


def check_events_and_log(events, log):
    print('Events s1:', len(events[events[:, 2] == 1]))
    print('Events s2:', len(events[events[:, 2] == 2]))
    print('Events exp:', len(events[events[:, 2] == 3]))
    print('Trials log:', len(log))
    if len(log) != len(events[events[:, 2] == 3]):
        raise ValueError('Unequal nr of events in eeg and log')


def add_event_tr_id(events):
    tr_id = -1
    tr_list = list()
    for ev in events:
        if ev[2] in (1, 10):
            tr_id += 1
        tr_list.append(tr_id)
    events_tr_id = np.column_stack((events, np.array(tr_list)))
    return events_tr_id


def make_exp_baseline(exp, pre_trial, events_tr_id, log, marks):
    exp_ids = events_tr_id[exp.selection, 3]
    pre_ids = events_tr_id[pre_trial.selection, 3]

    complete_tr_id = exp_ids[np.in1d(exp_ids, pre_ids)]
    exp_ids_complete = np.in1d(exp_ids, pre_ids)
    pre_ids_complete = np.in1d(pre_ids, exp_ids)

    exp_dat = exp._data[exp_ids_complete, :, :]
    pre_dat = pre_trial._data[pre_ids_complete, :, :]
    new_dat = np.concatenate((pre_dat, exp_dat), axis=2)

    conds = log['condition'].iloc[complete_tr_id]

    events = np.vstack((np.arange(0, len(conds)), np.zeros(len(conds)), conds)).transpose().astype(int)
    exp_ok = mne.EpochsArray(new_dat, info=exp.info, events=events, event_id=marks, tmin=-0.95)

    return exp_ok, complete_tr_id


def permutation_t_test(a, b, n_perm):
    t_real, p_real = ttest_rel(a, b)

    t_list = list()
    for per in range(int(n_perm)):
        joint = np.concatenate((a, b))
        np.random.shuffle(joint)
        split = np.split(joint, 2)
        t_perm, p_perm = ttest_rel(split[0], split[1])
        t_list.append(t_perm)
    p_permuted = len(np.where(t_list > t_real)[0]) / n_perm
    return t_real, t_list, p_permuted


def create_con_mat(con):
    con_mat = np.copy(con)

    upper = np.triu_indices_from(con_mat[:, :, 0])
    for fq in range(con.shape[2]):
        swap = np.swapaxes(con_mat[:, :, fq], 0, 1)
        for val in zip(upper[0], upper[1]):
            con_mat[:, :, fq][val] = swap[val]
    return con_mat


def get_subj_info(subjects):
    log_path = '/Users/lpen/Documents/Experimentos/Drowsy Time/TimeGeneralization/analisis/scalp_behaviour/logs'
    subj_table = {'lastname': [], 'age': [], 'tot_duration': []}
    for s in subjects:
        durations = []
        for ses in [1, 2, 3]:
            fname = op.join(log_path, 'ETG_scalp_su{}_se{}.mat' .format(s, ses))
            if op.isfile(fname):
                subj_dat = spio.loadmat(fname)
                ses_duration = subj_dat['Results'][-1, 12] - subj_dat['Results'][0, 3]
                durations.append(ses_duration)

                if ses == 1:
                    subj_table['lastname'].append(subj_dat['Info'][0][1].item())
                    if subj_dat['Info'][0][2].size > 0:
                        subj_table['age'].append(subj_dat['Info'][0][2].item())
                    else:
                        subj_table['age'].append('None')
        subj_table['tot_duration'].append(np.sum(np.array(durations) / 2))
    df_subj_info = pd.DataFrame(subj_table)

    df_subj_info['age'].iloc[0] = 32
    df_subj_info['age'].iloc[1] = 25
    df_subj_info['age'].iloc[2] = 22
    print('Mean Duration: ', df_subj_info['tot_duration'].mean() / 60)
    print('Std Duration: ', df_subj_info['tot_duration'].std())


def set_dif_and_rt_exp(log):
    def set_dif(row):
        if row['Order'] == 2.0:
            val = row['Standard'] - row['Comparison']
        else:
            val = row['Comparison'] - row['Standard']
        return val/1000

    def set_rt_exp(row):
        if row.condition == 90:
            val = row['RT'] + row['dif']
        elif row.condition == 70:
            val = row['RT'] - np.abs(row['dif'])
        return val

    log['dif'] = log.apply(set_dif, axis=1)
    log['rt_exp'] = log.apply(set_rt_exp, axis=1)
    return log


def permutation_pearson(x, y, n_perm):
    from scipy.stats import pearsonr
    r, p = pearsonr(x, y)

    r_perm_dist = list()
    for n in range(n_perm):
        y_rand = y.copy()
        np.random.shuffle(y_rand)
        r_perm, p_perm = pearsonr(x, y_rand)
        r_perm_dist.append(r_perm)
    if r > 0:
        p_permuted = sum(r_perm_dist>r) / n_perm
    else:
        p_permuted = sum(r_perm_dist<r) / n_perm
    return r, p_permuted

