#! /usr/bin/env python3
# coding: utf-8

import numpy as np
import sys, os
import librosa
import soundfile as sf
import time
import pickle as pic
import scipy as sc
import h5py
import pyroomacoustics as pyroom
from progressbar import progressbar
try:
    FLAG_GPU_Available = True
except:
    print("---Warning--- You cannot use GPU acceleration because chainer or cupy is not installed")


class alpha_IVA():
    def __init__(self, n_source=2, DIR_PATH=None, alpha=1.8, beta=0, seed=1, xp=np):
        """ initialize alpha_IVA

        Parameters:
        -----------
            n_source: int
                the number of sources
            n_basis: int
                the number of NMF bases of each source
            init_SM: str
                how to initialize spatial measure{all one: "ones", circulant matrix: "circ"}
        """
        super(alpha_IVA, self).__init__()
        self.eps = 1e-7
        self.DIR_PATH = DIR_PATH
        self.n_source = n_source
        self.alpha = alpha # characteristic exponent
        self.l1 = 0.0  # l1 regularization
        self.beta = beta # beta divergence
        self.method_name = "alpha_IVA"
        self.DIR_PATH = "/media/mafontai/SSD 2/data/speech_separation/wsj0/data/mix/"
        self.xp = xp
        self.seed = seed
        # self.rand_s = self.xp.random.RandomState(self.seed)


    def convert_to_NumpyArray(self, data):
        if self.xp == np:
            return data
        else:
            return self.xp.asnumpy(data)

    def load_spectrogram(self, X_FTM):
        """ load complex spectrogram

        Parameters:
        -----------
            X_FTM: self.xp.array [ F x T x M ]
                power spectrogram of observed signals
        """
        self.maxF = int(3000 * 2048 / 16000)
        self.minF = int(1000 * 2048 / 16000)
        _, self.n_time, self.n_mic = X_FTM.shape
        self.X_FTM = self.xp.asarray(X_FTM, dtype=self.xp.complex)
        self.n_freq = self.maxF - self.minF

    def save_A(self):
        f_model = h5py.File('/media/mafontai/SSD 2/data/speech_separation/RIRmodel.mat')
        A_model = f_model[u'a_model'].value[self.minF:self.maxF]

        P = A_model.shape[1] * A_model.shape[2]
        A_model = np.reshape(A_model, (A_model.shape[0], P, A_model.shape[3]))
        np.savez(self.DIR_PATH + "Theta-range=(1000-3000).npz", Theta_FPM=A_model)
        print('Theta are saved in {}'.format(self.DIR_PATH))

    def load_Theta(self):
        self.Theta_FPM = np.load(self.DIR_PATH + "Theta-range=(1000-3000).npz")['Theta_FPM'][..., :self.n_mic]
        self.Theta_FPM = self.xp.asarray(self.Theta_FPM)

    def compute_psi(self):
        file_name = os.path.join(self.DIR_PATH, "Psi-nfft=2048-alpha={}-n_mic={}-range=(1000-3000).npz".format(self.alpha, self.n_mic))
        self.Psi_FxPP = np.load(file_name)['Psi_FPP']
        self.P = self.Psi_FxPP.shape[1]
        self.Psi_FxPP = np.reshape(self.Psi_FxPP, (self.n_freq * self.P, self.P))
        self.Psi_FxPP = self.xp.asarray(self.Psi_FxPP)


    def init_SMs(self):
        self.SM_P = self.xp.ones((self.P)).astype(self.xp.float)


    def update_SM(self):
        # N x F x T x P
        Xhat_FxP = self.Psi_FxPP @ self.SM_P
        self.SM_P *= (self.eps + self.Psi_FxPP.T @ (self.X_FxP * Xhat_FxP ** (self.beta -2.))) /\
                     (self.eps + self.Psi_FxPP.T @ (Xhat_FxP ** (self.beta -1.)) + self.l1)

    def source_detection(self):
        self.SM_P /= self.SM_P.max()
        self.peak_indices = sc.signal.find_peaks(self.convert_to_NumpyArray(self.SM_P), height=0.1, distance=10)
        self.n_source_estimated = len(self.peak_indices)
        print('real / estimated # of sources : {}/ {}'.format(self.n_source, self.n_source_estimated))

    def separation(self):
        f_model = h5py.File('/media/mafontai/SSD 2/data/speech_separation/RIRmodel.mat')
        A_model = f_model[u'a_model'].value

        P = A_model.shape[1] * A_model.shape[2]
        A_model = self.xp.reshape(self.xp.array(A_model), (A_model.shape[0], P, A_model.shape[3]))

        self.A_FMN = self.xp.zeros((A_model.shape[0], self.n_mic, self.n_source_estimated)).astype(self.xp.complex64)
        for n in range(self.n_source_estimated):
            self.A_FMN[..., n] = A_model[:, self.peak_indices[n], :]

        self.S_NFT = (np.linalg.pinv(self.A_FMN)[:, None] * self.X_FTM[:, :, None]).sum(axis=-1).transpose(2, 0, 1)
        
    def compute_Lexp_est(self):

        """
        Compute empirical Levy_exponent of X := -ln (chf.(X)) with constant depending on alpha
        """

        c = 2 ** (1. / self.alpha)
        eps = 1e-10

        ThX = self.xp.real((self.Theta_FPM[:, None].conj() * self.X_FTM[self.minF:self.maxF, :, None]).sum(axis=-1))  # F x T x P
        Chf = self.xp.exp((1j / c) * ThX).mean(axis=1) + eps  # F x P

        self.X_FxP = np.reshape(-2 * self.xp.log(self.xp.abs(Chf)), (self.n_freq * self.P))


    def M_Step(self):
        self.update_SM()

    def save_parameter(self, fileName):
        # Base_path, id = os.path.split(os.path.split(self.DIR_PATH)[0])[0], os.path.split(self.DIR_PATH)[1]
        # File_sp = [os.path.join(Base_path, "s" + str(i_sp), id) for i_sp in range(1, self.n_source + 1)]
        if self.xp != np:
            SM_P = self.convert_to_NumpyArray(self.SM_P)
        np.savez(fileName, SM_P=SM_P, alpha=self.alpha, beta=self.beta, iteration=self.n_iteration)

    def solve(self, n_iteration=100, save_likelihood=False, save_parameter=False, save_wav=False, save_path="./", interval_save_parameter=30):
        """
        Parameters:
            save_likelihood: boolean
                save likelihood and lower bound or not
            save_parameter: boolean
                save parameter or not
            save_wav: boolean
                save intermediate separated signal or not
            save_path: str
                directory for saving data
            interval_save_parameter: int
                interval of saving parameter
        """

        # Initialization (Spatial Mask, Spatial Measure, NMF)
        self.n_iteration = n_iteration
        self.compute_psi()
        self.init_SMs()
        # self.save_A()
        self.load_Theta()
        self.compute_Lexp_est()
        self.make_filename_suffix()

        beta_div_array = []
        sdr_array = []

        for it in progressbar(range(self.n_iteration)):
            self.M_Step()
            if save_likelihood and ((it+1) % interval_save_parameter == 0) and ((it+1) != self.n_iteration):
                beta_div_array.append(self.calculate_beta_div())
                # self.Y_NFTM = self.convert_to_NumpyArray(self.Y_NFTM)
                # hop_length = int((self.n_freq - 1) / 2)
                # for n in range(self.n_source):
                #     for m in range(self.n_mic):
                #         tmp = librosa.core.istft(self.Y_NFTM[n, :, :, m], hop_length=hop_length)
                #         if n == 0 and m == 0:
                #             separated_signal = np.zeros([self.n_source, len(tmp), self.n_mic])
                #             separated_signal[n] = tmp
                # separated_signal /= np.abs(separated_signal).max() * 1.2
                #
                # bss_results = self.calculate_sdr(est_s)
                # sdr_array.append(bss_results[0])
                # sir_array.append(bss_results[1])
                # sar_array.append(bss_results[2])
        self.source_detection()
        self.separation()
        if save_likelihood and (it+1 == self.n_iteration):
            beta_div_array.append(self.calculate_beta_div())
            pic.dump(beta_div_array, open(save_path + "{}-likelihood-interval={}-{}.pic".format(self.method_name, interval_save_parameter, self.filename_suffix), "wb"))
            # pic.dump(sdr_array, open(save_path + "{}-sdr-interval={}-{}.pic".format(self.method_name, interval_save_parameter, self.filename_suffix), "wb"))
            # pic.dump(sir_array, open(save_path + "{}-sir-interval={}-{}.pic".format(self.method_name, interval_save_parameter, self.filename_suffix), "wb"))
            # pic.dump(sar_array, open(save_path + "{}-sar-interval={}-{}.pic".format(self.method_name, interval_save_parameter, self.filename_suffix), "wb"))
        if save_wav and ((it+1) % interval_save_parameter == 0) and ((it+1) != self.n_iteration):
            self.save_separated_signal(save_path+"{}-{}-{}".format(self.method_name, self.filename_suffix, it + 1))

        if save_wav and ((it+1) == self.n_iteration):
            self.E_Step()
            self.save_separated_signal(save_path+"{}-{}".format(self.method_name, self.filename_suffix))
        if save_parameter:
            self.save_parameter(save_path + "{}-parameters-{}.npz".format(self.method_name, self.filename_suffix))

    def calculate_beta_div(self):
        Xhat_FxP = self.Psi_FxPP @ self.SM_P
        if self.beta == 0.:      # IS Divergence
            value = ((self.X_FxP / (Xhat_FxP + self.eps)) -
                     (self.xp.log(self.X_FxP) - self.xp.log(Xhat_FxP)) -
                     1.).sum()

        elif self.beta == 1.:    # KL Divergence
            value = (self.X_FxP * (self.xp.log(self.X_FxP) - self.xp.log(Xhat_FxP)) +
                     (Xhat_FxP - self.X_FxP) -
                     1.).sum()
        else:
            value = ((self.beta) * (self.beta -1.)) ** (-1) * (
                     (self.X_FxP ** (self.beta)) +
                     (self.beta - 1) * (Xhat_FxP) ** (self.beta) -
                     (self.beta) * self.X_FxP *
                     (Xhat_FxP) ** (self.beta - 1.)
                     ).sum()
        return value

    def save_separated_signal(self, save_fileName="sample.wav"):
        self.Y_NFTM = self.convert_to_NumpyArray(self.Y_NFTM)
        hop_length = int((self.n_freq - 1) / 2)
        for n in range(self.n_source):
            for m in range(self.n_mic):
                tmp = librosa.core.istft(self.Y_NFTM[n, :, :, m], hop_length=hop_length)
                if n == 0 and m == 0:
                    separated_signal = np.zeros([self.n_source, len(tmp), self.n_mic])
                separated_signal[n, :, m] = tmp
        separated_signal /= np.abs(separated_signal).max() * 1.2

        for n in range(self.n_source):
            sf.write(save_fileName + "-N={}.wav".format(n), separated_signal[n], 16000)

    def make_filename_suffix(self):
        self.filename_suffix = "M={}-S={}-it={}-rand={}".format(self.n_mic, self.n_source, self.n_iteration, self.seed)

        if hasattr(self, "file_id"):
            self.filename_suffix += "-ID={}".format(self.file_id)
        return self.filename_suffix

if __name__ == "__main__":
    import argparse
    import pickle as pic
    import sys

    import glob as glob
    import os as os

    parser = argparse.ArgumentParser()
    parser.add_argument(         '--gpu', type= int, default=     0, help='GPU ID')
    parser.add_argument(       '--n_fft', type= int, default=  1024, help='number of frequencies')
    parser.add_argument(    '--n_speaker', type= int, default=    2, help='number of sources')
    parser.add_argument(     '--n_basis', type= int, default=     8, help='number of basis')
    parser.add_argument(    '--init_SCM', type= str, default="unit", help='unit, obs, two_step, ILRMA')
    parser.add_argument( '--n_iteration', type= int, default=   100, help='number of iteration')
    parser.add_argument( '--n_inter', type= int, default=  200, help='number of intervals')
    parser.add_argument( '--determined',   dest='determined', action='store_true', help='put the determined case (M=J)')
    parser.add_argument( '--alpha',   dest='alpha', type=float, default=2,  help='Gaussian case (alpha=2)')
    parser.add_argument( '--seed',   dest='seed', type=int, default=0,  help='random seed for experiments')
    parser.add_argument('--data', type=str, default='dev', help='available: dev or test')
    parser.add_argument( '--beta',   dest='p', type=float, default=2,  help='p-spectrogram (2 = power, 1=magnitude)')
    args = parser.parse_args()
