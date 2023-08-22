import numpy as np
import scipy
from tqdm.auto import trange
import random as rn
from brian2 import *
from brian2hears import *


def low_pass_filter(dU):
    fs = 1
    nyquist = fs / 2
    cutoff = .1
    b, a = scipy.signal.butter(2, cutoff, btype='lowpass')
    dUfilt = scipy.signal.filtfilt(b, a, dU[:, :])
    dUfilt = np.array(dUfilt)
    return dUfilt


def nonlinear_compressor(x, gamma=.1):
    return 1 / (1 + np.exp(-gamma * x)) - .5


def diff_t(x):
    return (x[1:, :] - x[:-1, :]) / 1e-4


# TODO delta x is wrong here
def diff_x(x):
    return (x[:, 1:] - x[:, :-1]) / 3e-3


def mu_to_convolve(nt):
    tau = 8 * 1e-3
    t = np.arange(nt)
    return np.exp(-t / tau)


def audioprocessing(sig, plot=True):
    cf = erbspace(65 * Hz, 3.5 * kHz, 129)
    fb = Gammatone(Sound(sig), cf)
    y_coh = fb.process()

    if plot:
        plt.figure(figsize=(10, 10))
        plt.imshow(y_coh[2500:3000].T)
        plt.title('Cochlear filters')
        plt.show()

    y_AN = low_pass_filter(nonlinear_compressor(diff_t(y_coh)))
    if plot:
        plt.figure(figsize=(10, 10))
        plt.imshow(y_AN[2500:3000].T)
        plt.title('Auditory nerve')
        plt.show()

    y_LIN = np.maximum(diff_x(y_AN), 0)
    if plot:
        plt.figure(figsize=(10, 10))
        plt.imshow(y_LIN[2500:3000].T)
        plt.title('Lateral inhibiton')
        plt.show()

    # mu = mu_to_convolve(y_LIN.shape[0])
    # y_final = np.array([np.convolve(y_LIN[:, i], mu, mode='same') for i in trange(y_LIN.shape[1])])
    # if plot:
    #     plt.figure(figsize=(10, 10))
    #     plt.imshow(y_final[:, 2500:3000])
    #     plt.title('Loss of phase locking in the midbrain')
    #     plt.show()

    return y_LIN.T


import timit_utils as tu
import timit_utils.audio_utils as au
import timit_utils.drawing_utils as du


