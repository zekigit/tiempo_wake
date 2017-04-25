import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')


def calc_decision(p, t):
    w = (2 * (np.abs(1 - p))) / (1 + p)
    decision = w < t
    return w, decision


def calc_decision_ls(p, t, d):
    w_SC = (2 * (np.abs(1 - (p-d)))) / (1 + (p-d))
    w_CS = (2 * (np.abs(1 - (p+d)))) / (1 + (p+d))
    decision_sc = w_SC < t
    decision_cs = w_CS < t
    return w_SC, decision_sc, w_CS, decision_cs

p = np.arange(0.25, 2, 0.25)
thresholds = np.arange(0, 0.3, 0.05)

calcs = list()
calcs_sc = list()
calcs_cs = list()
decisions = list()
decisions_sc = list()
decisions_cs = list()

d = 0.2

for ix, t in enumerate(thresholds):
    w, decision = calc_decision(p, t)
    w_SC, decision_sc, w_CS, decision_cs = calc_decision_ls(p, t, d)
    calcs.append(w)
    calcs_sc.append(w_SC)
    calcs_cs.append(w_CS)
    decisions.append(decision)
    decisions_sc.append(decision_sc)
    decisions_cs.append(decision_cs)

fig, axes = plt.subplots(1, len(thresholds), sharey=True, sharex=True)
for ax, calc, dec, t in zip(axes, calcs, decisions, thresholds):
    ax.plot(p, calc, 'r')
    ax.plot(p, dec, 'g')
    ax.plot([0, 2], [t, t], 'b-', lw=1)
    ax.set_xticks([0.5, 1, 1.5])
    ax.axis([0, 2, -0.1, 1.4])
    ax.set_title('t= ' + str(t))
    ax.legend(('w', 'decision', 't'))

fig_order, axes = plt.subplots(2, len(thresholds), sharex=True, sharey=True)

