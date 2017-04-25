import mne
from mne.time_frequency import tfr_morlet
import numpy as np
import os.path as op
from etg_scalp_info import study_path, sujetos, sesiones, n_jobs
import matplotlib.pyplot as plt

s1_short = list()
s2_short = list()
s1_long = list()
s2_long = list()

exp_short_eq = list()
exp_short_smaller = list()
exp_short_bigger = list()

exp_long_eq = list()
exp_long_smaller = list()
exp_long_bigger = list()


for subj in sujetos:
    sesiones_x_subj = sesiones[subj]
    for ses in sesiones_x_subj:
        archivo = 'etg_su{}_se{}'.format(subj, ses)
        full_path = op.join(study_path, 'fif', archivo + 'avg_subcond-epo.fif')
        epochs = mne.read_epochs(full_path)
        epochs.drop_bad()
        # epochs.plot_drop_log()

        s1_short.append(epochs['s1_short'].average())
        s2_short.append(epochs['s2_short'].average())
        s1_long.append(epochs['s1_long'].average())
        s2_long.append(epochs['s2_long'].average())

        exp_short_eq.append(epochs['exp_short_equal'].average())
        exp_short_smaller.append(epochs['exp_short_smaller'].average())
        exp_short_bigger.append(epochs['exp_short_bigger'].average())

        exp_long_eq.append(epochs['exp_long_equal'].average())
        exp_long_smaller.append(epochs['exp_long_smaller'].average())
        exp_long_bigger.append(epochs['exp_long_bigger'].average())

s1_short_ready = mne.combine_evoked(s1_short)
s2_short_ready = mne.combine_evoked(s2_short)
s1_long_ready = mne.combine_evoked(s1_long)
s2_long_ready = mne.combine_evoked(s2_long)

exp_short_eq_ready = mne.combine_evoked(exp_short_eq)
exp_short_smaller_ready = mne.combine_evoked(exp_short_smaller)
exp_short_bigger_ready = mne.combine_evoked(exp_short_bigger)

exp_long_eq_ready = mne.combine_evoked(exp_long_eq)
exp_long_smaller_ready = mne.combine_evoked(exp_long_smaller)
exp_long_bigger_ready = mne.combine_evoked(exp_long_bigger)

s2_exp_all = mne.combine_evoked([exp_short_smaller_ready, exp_short_bigger_ready, exp_long_smaller_ready, exp_long_bigger_ready])
s2_exp_short = mne.combine_evoked([exp_short_smaller_ready, exp_short_bigger_ready])
s2_exp_long = mne.combine_evoked([exp_long_smaller_ready, exp_long_bigger_ready])

# Butterfly
s1_short_ready.plot(spatial_colors=True)
s1_long_ready.plot(spatial_colors=True)

s2_short_ready.plot(spatial_colors=True)
s2_long_ready.plot(spatial_colors=True)

s2_exp_all.plot(spatial_colors=True)
s2_exp_short.plot(spatial_colors=True)
s2_exp_long.plot(spatial_colors=True)

# exp_long_bigger_ready.plot(spatial_colors=True)

# Comp ERP
mne.viz.plot_evoked_topo([s2_short_ready, s2_long_ready], color=['w', 'b'], title='S2 Short vs S2 Long')
mne.viz.plot_evoked_topo(s2_exp_all, color='lime', title='S2 Expected Ending')

# Topography
s2_exp_all.plot_topomap(times=[-0.25, -0.10, -0.015, 0, 0.015, 0.10, 0.25], title='S2 exp all')
s2_exp_short.plot_topomap(times=[-0.25, -0.10, -0.015, 0, 0.015, 0.10, 0.25], title='S2 exp short')
s2_exp_long.plot_topomap(times=[-0.25, -0.10, -0.015, 0, 0.015, 0.10, 0.25], title='S2 exp long')

# Time-Frequency
freqs = np.arange(6, 30, 3)  # define frequencies of interest
n_cycles = freqs / 2.  # different number of cycle per frequency
power = tfr_morlet(exp_long_bigger_ready, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                   return_itc=False, decim=3, n_jobs=n_jobs, average=True)


power.plot_topo(baseline=(-0.25, -0.15), mode='mean', title='Mean power vs baseline')
power.plot_topomap(tmin=-0.1, tmax=0.1, fmin=10, fmax=12, mode=None)

pow_erp = power.data[:, 2, :]
times = np.tile(power.times, (128, 1))

all_pow_fig = plt.subplot(111)
for ix, ch in enumerate(pow_erp):
    all_pow_fig.plot(times[ix,:], pow_erp[ix,:])
all_pow_fig.set(title='Raw 12hz power x channel ')

ch = 27
chan_pow = plt.subplot(111)
chan_pow.plot(power.times, pow_erp[ch, :])
chan_pow.plot([0,0], [0, chan_pow.get_ybound()[1]], 'k--')
chan_pow.set(title='Raw 12 hz power - ch:' + str(ch))
