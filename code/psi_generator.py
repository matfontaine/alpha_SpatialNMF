#! /usr/bin/env python3
# coding: utf-8

import numpy as np
import scipy.stats as st
import sys, os
import librosa
import soundfile as sf
import time
import pickle as pic

import sys
import progressbar
import h5py
try:
    FLAG_GPU_Available = True
except:
    print("---Warning--- You cannot use GPU acceleration because chainer or cupy is not installed")


if __name__ == "__main__":
    import argparse
    import pickle as pic
    import sys

    import glob as glob
    import os as os

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type= int, default=     0, help='GPU ID')
    parser.add_argument('--nfft', type=int, default=2048, help='number of fft window')
    parser.add_argument('--minF', type=float, default=1000, help='minimum frequency (in Hz)')
    parser.add_argument('--maxF', type=int, default=3000, help='maximum frequency (in Hz)')

    args = parser.parse_args()

    DIR_PATH = "/media/mafontai/SSD 2/data/speech_separation/wsj0/data/"
    SAVE_PATH = os.path.join(DIR_PATH, 'mix/')

    if args.gpu < 0:
        import numpy as xp
    else:
        import cupy as xp
        print("Use GPU " + str(args.gpu))
        xp.cuda.Device(args.gpu).use()
    # import ipdb; ipdb.set_trace()

    # randomly draw indices in grid

    maxF = int(args.maxF * args.nfft / 16000)
    minF = int(args.minF * args.nfft / 16000)
    #
    dF = maxF - minF
    ALPHA = [1., 1.2, 1.4, 1.6, 1.8]
    MIC = [10, 20, 30]
    f_model = h5py.File('/media/mafontai/SSD 2/data/speech_separation/RIRmodel.mat')

    A_model = f_model[u'a_model'].value
    P = A_model.shape[1] * A_model.shape[2]
    A_model  = xp.reshape(xp.array(A_model), (A_model.shape[0], P, A_model.shape[3]))

    for alpha in progressbar.progressbar(ALPHA):
        for M in MIC:
            A_FPM = xp.zeros((dF, P, M)).astype(xp.complex64)
            for m in range(M):
                A_FPM[..., m] = xp.fft.rfft(A_model[..., m], n=args.nfft, axis=0)[minF:maxF]
            A_FPM /= xp.linalg.norm(A_FPM, axis=-1, keepdims=True)
            A_FPM = xp.asnumpy(A_FPM)
            Psi_FPP = np.zeros((dF, P, P)).astype(xp.complex64)
            for m in range(M):
            # F x P x P
                Psi_FPP += A_FPM.conj()[..., None, m] * A_FPM[:, None, :, m]
            Psi_FPP = np.abs(Psi_FPP) ** (alpha)

            np.savez(SAVE_PATH + 'Psi-nfft={}-alpha={}-n_mic={}-range=({}-{}).npz'.format(args.nfft, alpha, M, args.minF, args.maxF), Psi_FPP=Psi_FPP)
            print('Psi-nfft={}-alpha={}-n_mic={}-range=({}-{}).npz saved !'.format(args.nfft, alpha, M, args.minF, args.maxF))
