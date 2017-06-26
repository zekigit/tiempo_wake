from pyhht import EMD
from pyhht.visualization import plot_imfs
from scipy.signal import hilbert
from pyhht.utils import inst_freq
import matplotlib.pyplot as plt
import numpy as np

lon = []
lon = lon.crop(-0.5, 0.5)
signal = lon._data[5, 57, :]

decomposer = EMD(signal)
imfs = decomposer.decompose()
plot_imfs(signal, imfs, lon.times)
ht = hilbert(imfs)
plt.imshow(np.abs(ht), aspect='auto', origin='lower')


