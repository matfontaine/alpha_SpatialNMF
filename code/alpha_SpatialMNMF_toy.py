#! /usr/bin/env python3
# coding: utf-8

import numpy as np
import sys, os
import librosa
import soundfile as sf
import time
import pickle as pic

from scipy.io import loadmat
from progressbar import progressbar
try:
    FLAG_GPU_Available = True
except:
    print("---Warning--- You cannot use GPU acceleration because chainer or cupy is not installed")


class alpha_SpatialMNMF():
    def __init__(self, n_source=2, n_mic=2, n_sample=2000, DIR_PATH=None, alpha=1.8, beta=0, seed=1, n_Th=180, xp=np, init_SM="unit"):
        """ initialize alpha_SpatialMNMF

        Parameters:
        -----------
            n_source: int
                the number of sources
            n_Th :      int
                number of sphere sampling
            init_SM: str
                how to initialize spatial measure{all one: "ones", circulant matrix: "circ"}
        """
        super(alpha_SpatialMNMF, self).__init__()
        self.eps = 1e-7
        self.DIR_PATH = DIR_PATH
        self.n_Th = n_Th # number of sphere sampling
        self.n_source = n_source
        self.alpha = alpha # characteristic exponent
        self.n_mic = n_mic
        self.n_sample = n_sample
        self.l1 = 0.0  # l1 regularization
        self.beta = beta # beta divergence
        if self.beta < 1.:
            self.e = (2 - self.beta) ** (-1)
        elif self.beta >= 1 and self.beta <= 2:
            self.e = 1.
        else:
            self.e = (self.beta - 1.) ** (-1)

        self.method_name = "alpha_SpatialMNMF"
        self.xp = xp
        self.seed = seed
        self.rand_s = self.xp.random.RandomState(self.seed)
        self.init_SM = init_SM

        # precompute sphere sampling and variables


    def convert_to_NumpyArray(self, data):
        if self.xp == np:
            return data
        else:
            return self.xp.asnumpy(data)

    def alpha_sampling(self, alpha=1, beta=1, mu=0, sigma=1, shape=None, seed=None):
        """
           Input
           -----
           alpha: 0 < alpha <=2
               exponential characteristic coefficient
           beta: -1 <= beta <= 1
               skewness parameter
           mu: real
               the mean
           sigma: positive real
                  scale parameter
           shape: as you want :) (give a tuple)
                  size and number of sampling

           Returns
           -------
           S: shape
               give a sampling of an S(alpha, beta, mu, sigma) variable

           """

        if seed is None:
            W = self.xp.random.exponential(1, shape)
            U = self.xp.random.uniform(-self.xp.pi / 2., self.xp.pi / 2., shape)

            c = -beta * self.xp.tan(self.xp.pi * alpha / 2.)
            if alpha != 1:
                ksi = 1 / alpha * self.xp.arctan(-c)
                res = ((1. + c ** 2) ** (1. / 2. * alpha)) * self.xp.sin(alpha * (U + ksi)) / ((self.xp.cos(U)) ** (1. / alpha)) \
                      * ((self.xp.cos(U - alpha * (U + ksi))) / W) ** ((1. - alpha) / alpha)

            else:
                ksi = self.xp.pi / 2.
                res = (1. / ksi) * ((self.xp.pi / 2. + beta * U) * self.xp.tan(U) -
                                    beta * self.xp.log((self.xp.pi / 2. * W * self.xp.cos(U)) / (self.xp.pi / 2. + beta * U)))

        else:
            _random = self.xp.random.RandomState(seed)
            W = _random.exponential(1, shape)
            U = _random.uniform(-self.xp.pi / 2., self.xp.pi / 2., shape)

            c = -beta * self.xp.tan(self.xp.pi * alpha / 2.)
            if alpha != 1:
                ksi = 1 / alpha * self.xp.arctan(-c)
                res = ((1. + c ** 2) ** (1. / 2. * alpha)) * self.xp.sin(alpha * (U + ksi)) / ((self.xp.cos(U)) ** (1. / alpha)) \
                      * ((self.xp.cos(U - alpha * (U + ksi))) / W) ** ((1. - alpha) / alpha)

            else:
                ksi = self.xp.pi / 2.
                res = (1. / ksi) * ((self.xp.pi / 2. + beta * U) * self.xp.tan(U) -
                                    beta * self.xp.log((self.xp.pi / 2. * W * self.xp.cos(U)) / (self.xp.pi / 2. + beta * U)))
        return res * sigma + mu

    def compute_sampling(self):
        self.SM_true_NP = self.xp.zeros((self.n_source, self.n_Th)).astype(self.xp.float32)  # (N, P)
        self.lambda_true_NT = self.rand_s.rand(self.n_source, self.n_sample)
        S_NTP = self.xp.zeros((self.n_source, self.n_sample, self.n_Th)).astype(self.xp.float32)  # sources (N, T, P)
        self.Y_true_NTM = self.xp.zeros((self.n_source, self.n_sample, self.n_mic)).astype(self.xp.float32)  # img_sources (N, T, M)
        Cste = float(2. * self.xp.pi ** (self.n_mic) / np.math.factorial(self.n_mic))
        Cste /= self.n_Th

        for n in range(self.n_source):  # Dirac assumption
            self.SM_true_NP[n, int(self.n_Th / (self.n_source + 2))] = 1.
            for dP in range(self.n_Th):
                S_NTP[n, :, dP] = self.alpha_sampling(alpha=self.alpha, beta=0,
                                                      mu=0, sigma=self.SM_true_NP[n, dP],
                                                      shape=(self.n_sample),
                                                      seed=self.seed)
                S_NTP[n, :, dP] *= (self.lambda_true_NT[n] ** (1. / self.alpha))
            self.Y_true_NTM[n, ...] = Cste * np.sum(self.Theta_PM[None, ...] *
                                                    S_NTP[n, :, :, None], axis=-2)
    def compute_Theta_Oracle(self):  # Nearfield region assumption and Spatial measure = Diracs
        data = loadmat("rir_info.mat")
        index = 5000
        mic_pos = data['INFO'][0][index]['mic_pos'][:self.n_mic]
        spk_pos = data['INFO'][0][index]['spk_pos'][:self.n_source]
        self.Theta_PM = self.xp.zeros((self.n_Th, self.n_mic)).astype(complex)
        r_PM = self.xp.zeros((self.n_Th, self.n_mic)).astype(float)
        for m in range(self.n_mic):
            for n in range(self.n_source):
                r_PM[n, m] = self.xp.linalg.norm(spk_pos[n] - mic_pos[m])

        for p in range(self.n_source, self.n_Th):
            r_PM[p, :] = self.xp.abs(self.rand_s.rand(self.n_mic)) + 0.5

        c =340. # speed of the sound in the air
        freq_F = self.xp.fft.rfftfreq(2048, 1. / 16000.).astype(np.float32)

        # nearfield assumption with the real distance
        self.Theta_PM = 1./self.xp.maximum(3., r_PM) *\
                     self.xp.exp(-2j * self.xp.pi *
                                 freq_F[len(freq_F) // 2, None, None] *
                                 r_PM / c)
        self.Theta_PM /= self.xp.linalg.norm(self.Theta_PM, axis=-1, keepdims=True)

    def init_variable(self):
        # hypersphere sampling
        # self.Theta_FPM = self.rand_s.normal(0, 1, size=(self.n_Th, self.n_mic)) +\
        #                 1j * self.rand_s.normal(0, 1, size=(self.n_Th, self.n_mic))
        # self.Theta_FPM /= self.xp.linalg.norm(self.Theta_PM, axis=-1, keepdims=True)

        # auxiliary spatial variables
        # P x P' x M
        self.compute_Theta_Oracle()
        self.compute_sampling()
        self.X_TM = self.Y_true_NTM.sum(axis=0)
        self.ThTh_PP = (self.Theta_PM.conj()[:, None] * self.Theta_PM[None]).sum(axis=-1)  # F x P x P'
        self.Psi_PP = self.xp.abs(self.ThTh_PP) ** (self.alpha)

        phi = self.xp.sum(self.Psi_PP * self.Psi_PP.conj()).real / self.n_mic
        self.Psi_PP = self.Psi_PP / self.xp.sqrt(phi)
        # Sketching and spatial parameters
        # self.compute_Lexp_est()
        self.compute_Lexp_est2()
        self.reset_variable()


        # oracle spatial measure
        # self.SM_NFP[0, :, 0] = 1.
        # self.SM_NFP[1, :, 1] = 1.
        # self.SM_NFP[0, :, 1] = 1e-7
        # self.SM_NFP[1, :, 0] = 1e-7


    def init_SMs(self):
        if "ones" == self.init_SM:
            self.SM_NP = self.xp.ones((self.n_source, self.n_Th)).astype(self.xp.float)
            # self.SM_NFP = self.rand_s.rand(self.n_source, self.n_freq, self.n_Th)
        elif "circ" == self.init_SM:
            self.SM_NP =  self.xp.maximum(1e-2, self.xp.zeros([self.n_source, self.n_Th], dtype=self.xp.float))
            for p in range(self.n_Th):
                self.SM_NP[p % self.n_source, p] = 1
        else:
            print("Specify how to initialize the Spatial Measure {ones, circ}")
            raise ValueError

    def init_PSD(self):
        self.lambda_NT = self.rand_s.rand(self.n_source, self.n_sample)

    def update_PSD(self):
        # N x T x Pp
        num_l = (self.Y_TP[None] ** (self.beta - 2.) *\
                      self.X_TP[None] *\
                      self.G_NP[:, None]).sum(axis=-1)
        den_l = (self.Y_TP[None] ** (self.beta - 1.) *\
                 self.G_NP[:, None]).sum(axis=-1) + self.eps + self.l1

        self.lambda_NT *= (num_l/den_l) ** self.e
        self.reset_variable()

    def update_SM(self):
        # N x T x P
        num_SM = (self.lambda_NT[..., None] *\
                  (self.Psi_PP[None] * self.X_TP[:, None]).sum(axis=-1)[None] *\
                  self.Y_TP[None] ** (self.beta - 2.)).sum(axis=1)
        den_SM = (self.lambda_NT[..., None] *\
                  (self.Psi_PP[None] * (self.Y_TP[:, None] ** (self.beta -1.))).sum(axis=-1)[None]
                  ).sum(axis=1) + self.eps + self.l1
        self.SM_NP *= (num_SM / den_SM) ** self.e
#        self.SM_NFP[0, :, 0] = 1.
#        self.SM_NFP[1, :, 1] = 1.
#        self.SM_NFP[0, :, 1] = 1e-3
#        self.SM_NFP[1, :, 0] = 1e-3
        self.reset_variable()



    def reset_variable(self):
        self.G_NP = (self.Psi_PP[None, ...] * self.SM_NP[:, None, :]).sum(axis=-1)
        self.Y_TP = (self.lambda_NT[..., None] * self.G_NP[:, None]).sum(axis=0)

    def compute_Lexp_est(self):

        """
        Compute empirical Levy_exponent of X := -ln (chf.(X)) with constant depending on alpha
        """

        c = 2 ** (1. / self.alpha)
        eps = 1e-10

        ThX = self.xp.real((self.Theta_PM[None].conj() * self.X_TM[:, None]).sum(axis=-1))  # F x T x P
        Chf = self.xp.exp((1j / c) * ThX) + eps  # F x T x P

        self.X_TP = -2 * self.xp.log(self.xp.abs(self.X_TP))

    def compute_Lexp_est2(self):

        """
        Compute empirical Levy_exponent of X := -ln (chf.(X)) available for all alpha-stable distributions
        """
        eps = 1e-10

        ThX = self.xp.real((self.Theta_PM[None].conj() * self.X_TM[:, None]).sum(axis=-1))  # F x T x P
        Chf = self.xp.cos(ThX)  # F x T x P

        self.X_TP = - self.xp.log(self.xp.abs(Chf))

    def normalize(self):
        # self.W_NFK = self.W_NFK / phi_F[None, :, None]

        mu_N = (self.SM_NP).sum(axis=-1)
        self.SM_NP = self.SM_NP / mu_N[:, None]

        self.G_NP = (self.Psi_PP[None, ...] * self.SM_NP[:, None, :]).sum(axis=-1)

        mu_NF = (self.G_NP).sum(axis=-1)
        self.G_NP = self.G_NP / mu_NF[..., None]
        self.lambda_NT = self.lambda_NT * mu_N[:, None]

        self.reset_variable()

    def E_Step(self):
        # self.SM_NFP[0, :, 0] = 1.
        # self.SM_NFP[1, :, 1] = 1.
        # self.SM_NFP[0, :, 1] = 1e-3
        # self.SM_NFP[1, :, 0] = 1e-3
        # Base_path, id = os.path.split(os.path.split(self.DIR_PATH)[0])[0], os.path.split(self.DIR_PATH)[1]
        # File_sp = [os.path.join(Base_path, "s" + str(i_sp), id) for i_sp in range(1, self.n_source + 1)]
        #
        # lambda_true_NFT = np.zeros((self.n_source, self.n_freq, self.n_time)).astype(np.float)
        # for i_sp in range(self.n_source):
        #     tmp_true_s, fs = sf.read(File_sp[i_sp])
        #     stft_sp = np.asarray(librosa.core.stft(np.asfortranarray(tmp_true_s[:, 0]), n_fft=2048, hop_length=512))
        #     lambda_true_NFT[i_sp] = np.abs(stft_sp)
        # self.lambda_NFT = self.xp.array(lambda_true_NFT)
        self.Y_TP = (self.lambda_NT[..., None] * self.G_NP[:, None]).sum(axis=0)
        # Numerator
        ThTh_alpha = self.ThTh_PP * (self.xp.abs(self.ThTh_PP) ** (self.alpha - 2.))  # P x P'
        temp_num = ThTh_alpha[None] / (self.Y_TP[:, None, :] ** (2 * self.n_mic/self.alpha + 1.))  # T x P x P'

        # T x  P x P' x M
        Num = (temp_num[..., None] * self.Theta_PM[None, :, None]).sum(axis=-2)  # T x P x M

        # Denominator
        Den = ((self.Y_TP) ** (-2 * self.n_mic/self.alpha)).sum(axis=-1)  # T

        # first T P M M
        self.Xi_TMMP = self.n_mic * (self.Theta_PM[None, :, None] *
                                      (Num/Den[..., None, None])[..., None].conj()).transpose(0, 2, 3, 1)

        # N T M M P
        Mask_NTMM = (self.lambda_NT[..., None, None, None] *\
                      self.Xi_TMMP[None] *\
                      self.SM_NP[:, None, None, None]).sum(axis=-1)
        # Mask_NFTMM *= 2. * self.xp.pi ** (self.n_mic) / np.math.factorial(self.n_mic)
        # Mask_NFTMM /= self.n_Th
        self.Y_NTM = (Mask_NTMM * self.X_TM[None, :, None]).sum(axis=-1)


        # # F x P x P' x M
        # Num = (temp_num[..., None] * self.Theta_FPM[:, None]).sum(axis=-2)  # F x P x M
        #
        # # Denominator
        # Den = ((self.X_FP) ** (-2 * self.n_mic/self.alpha)).sum(axis=-1) # F
        #
        # self.Xi_FMMP = self.n_mic * (self.Theta_FPM[:, :, None] *
        #                              (Num/Den[:, None, None])[..., None].conj()).transpose(0, 3, 2, 1)
        #
        # # N F T M M P
        # Mask_NFTMM = (self.lambda_NFT[..., None, None, None] *\
        #               self.Xi_FMMP[None, :, None] *\
        #               self.SM_NFP[:, :, None, None, None]).sum(axis=-1)
        # Mask_NFTMM *= 2. * self.xp.pi ** (self.n_mic) / np.math.factorial(self.n_mic)
        # Mask_NFTMM /= self.n_Th
        # self.Y_NFTM = (Mask_NFTMM * self.X_FTM[None, :, :, None]).sum(axis=-1)

    def M_Step(self):
        self.update_PSD()
        self.update_SM()
        self.normalize()

    def save_parameter(self, fileName):
        if self.xp != np:
            lambda_NT = self.convert_to_NumpyArray(self.lambda_NT)
            SM_NP = self.convert_to_NumpyArray(self.SM_NP)
            lambda_true_NT = self.convert_to_NumpyArray(self.lambda_true_NT)
            Y_NTM = self.convert_to_NumpyArray(self.Y_NTM)
            Y_true_NTM = self.convert_to_NumpyArray(self.Y_true_NTM)
        np.savez(fileName, lambda_NT=lambda_NT,
                 lambda_true_NT=lambda_true_NT,
                 SM_NFP=SM_NP, Y_true_NTM=Y_true_NTM,
                 Y_NTM=Y_NTM)

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

        self.init_SMs()
        self.init_PSD()
        self.init_variable()
        self.make_filename_suffix()

        beta_div_array = []
        # sdr_array = []

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

        if save_likelihood and (it+1 == self.n_iteration):
            beta_div_array.append(self.calculate_beta_div())
            pic.dump(beta_div_array, open(save_path + "{}-likelihood-interval={}-{}.pic".format(self.method_name, interval_save_parameter, self.filename_suffix), "wb"))
            # pic.dump(sdr_array, open(save_path + "{}-sdr-interval={}-{}.pic".format(self.method_name, interval_save_parameter, self.filename_suffix), "wb"))
            # pic.dump(sir_array, open(save_path + "{}-sir-interval={}-{}.pic".format(self.method_name, interval_save_parameter, self.filename_suffix), "wb"))
            # pic.dump(sar_array, open(save_path + "{}-sar-interval={}-{}.pic".format(self.method_name, interval_save_parameter, self.filename_suffix), "wb"))
        if save_wav and ((it+1) % interval_save_parameter == 0) and ((it+1) != self.n_iteration):
            self.save_separated_signal(save_path+"{}-{}-{}".format(self.method_name, self.filename_suffix, it + 1))

        # separation of alpha-stable random vector
        self.E_Step()
        if save_parameter:
            self.save_parameter(save_path + "{}-parameters-{}.npz".format(self.method_name, self.filename_suffix))

    def calculate_beta_div(self):

        if self.beta == 0.:      # IS Divergence
            value = ((self.X_TP / (self.Y_TP + self.eps)) -
                     (self.xp.log(self.X_TP) - self.xp.log(self.Y_TP)) -
                     1.).sum()

        elif self.beta == 1.:    # KL Divergence
            value = (self.X_TP * (self.xp.log(self.X_TP) - self.xp.log(self.Y_TP)) +
                     (self.Y_TP - self.X_TP) -
                     1.).sum()
        else:
            value = ((self.beta) * (self.beta -1.)) ** (-1) * (
                     (self.X_TP ** (self.beta)) +
                     (self.beta - 1) * (self.Y_TP) ** (self.beta) -
                     (self.beta) * self.X_TP *
                     (self.Y_TP) ** (self.beta - 1.)
                     ).sum()
        return value


    def make_filename_suffix(self):
        self.filename_suffix = "M={}-S={}-it={}-init={}-rand={}".format(self.n_mic, self.n_source, self.n_iteration, self.init_SM, self.seed)

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
