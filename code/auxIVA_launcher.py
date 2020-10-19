#! /usr/bin/env python3
# coding: utf-8

import numpy as np
import scipy.stats as st
import sys, os
import librosa
import soundfile as sf
import time
import pickle as pic

import h5py
from alpha_ICA import alpha_ICA
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
    parser.add_argument(         '--gpu', type= int, default=     0, help='GPU ID')
    parser.add_argument(       '--n_fft', type= int, default=  2048, help='number of frequencies')
    parser.add_argument(    '--n_speaker', type= int, default=    2, help='number of sources')
    parser.add_argument(     '--n_basis', type= int, default=     8, help='number of basis')
    parser.add_argument(    '--init_SM', type= str, default="unit", help='unit')
    parser.add_argument( '--n_iteration', type= int, default=   100, help='number of iteration')
    parser.add_argument( '--n_inter', type= int, default=  200, help='number of intervals')
    parser.add_argument( '--determined',   dest='determined', action='store_true', help='put the determined case (M=J)')
    parser.add_argument( '--alpha',   dest='alpha', type=float, default=1.8,  help='Gaussian case (alpha=2)')
    parser.add_argument( '--id',   dest='id', type=int, default=0,  help='id number')
    parser.add_argument('--n_mic', type=int, default=20, help='number of microphones')
    parser.add_argument( '--beta',   dest='beta', type=float, default=0,  help='beta divergence')

    args = parser.parse_args()

    DIR_PATH = "/media/mafontai/SSD 2/data/speech_separation/wsj0/data/mix"
    EST_DIR = "/home/mafontai/Documents/project/git_project/speech_separation/alpha_ICA/results_" + str(args.n_speaker) + "reverb/"
    SAVE_PATH = os.path.join(EST_DIR, "alpha=%s" % str(args.alpha), "beta=%s/" % str(args.beta))


    if args.gpu < 0:
        import numpy as xp
    else:
        import cupy as xp
        print("Use GPU " + str(args.gpu))
        xp.cuda.Device(args.gpu).use()
    # import ipdb; ipdb.set_trace()
    file_path = os.path.join(SAVE_PATH,
                             "alpha_ICA-M={}-S={}-it={}-ID={}-N=0.wav".format(str(args.n_mic), str(args.n_speaker), str(200), str(args.id)))
    if os.path.exists(file_path):
        print("alpha_ICA => interval={}-M={}-S={}-it={}-ID={} already done.".format(str(args.n_inter), str(args.n_mic), str(args.n_speaker), str(200), str(args.id)))
        pass
    else:
        size = []
        for n in range(args.n_speaker):
            tmp_name = os.path.join(DIR_PATH, "audio-N={}-seed={}.wav".format(n, args.id))
            tmp_wav, fs = sf.read(tmp_name)
            size.append(tmp_wav.shape[0])

        sig_size = min(size)
        for n in range(args.n_speaker):
            tmp_name = os.path.join(DIR_PATH, "audio-N={}-seed={}.wav".format(n, args.id))
            tmp_wav, fs = sf.read(tmp_name)
            tmp_wav = tmp_wav[:sig_size-1].T
            for m in range(args.n_mic):
                tmp = librosa.core.stft(np.asfortranarray(tmp_wav[m]),
                                        n_fft=args.n_fft,
                                        hop_length=int(args.n_fft / 4))
                if m == 0 and n == 0:
                    spec = np.zeros([tmp.shape[0], tmp.shape[1], args.n_mic], dtype=np.complex)

                spec[:, :, m] += tmp

        separater = alpha_ICA(n_source=args.n_speaker,
                                      alpha=args.alpha, DIR_PATH=DIR_PATH,
                                      beta=args.beta, xp=xp, seed=args.id)
        separater.load_spectrogram(spec)
        separater.file_id = args.id

        if not os.path.exists(SAVE_PATH):
            os.makedirs(SAVE_PATH)
        separater.solve(n_iteration=args.n_iteration, save_likelihood=True,
                        save_parameter=True, save_wav=True,
                        save_path=SAVE_PATH,
                        interval_save_parameter=args.n_inter)
