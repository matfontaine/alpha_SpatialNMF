import pylab as plt

from scipy.stats import vonmises, levy_stable
from scipy.signal import unit_impulse

def compute_sampling(self):
    if gpu < 0:
        import numpy as xp
    else:
        import cupy as xp
        print("Use GPU " + str(gpu))
        xp.cuda.Device(gpu).use()
    SM_NP = xp.zeros((N, P)).astype(np.float32)  # (N, P)
    S_NTP = xp.zeros((N, T, P)).astype(xp.float32)  # sources (N, T, P)
    Y_NTM = xp.zeros((N, T, M)).astype(xp.float32)  # img_sources (N, T, M)

    for n in range(N):
