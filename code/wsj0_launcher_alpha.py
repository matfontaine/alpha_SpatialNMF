#! /usr/bin/env python3
# coding: utf-8

import numpy as np
import scipy.stats as st
import sys, os
import librosa
import soundfile as sf
import time
import pickle as pic

from alpha_SpatialMNMF import alpha_SpatialMNMF
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
    parser.add_argument( '--seed',   dest='seed', type=int, default=0,  help='random seed for experiments')
    parser.add_argument('--data', type=str, default='dev', help='available: dev or test')
    parser.add_argument('--nb_file', type=int, default=1, help='nb of file to separate')
    parser.add_argument('--n_Th', type=int, default=180, help='number of sphere sampling')
    parser.add_argument( '--beta',   dest='beta', type=float, default=0,  help='beta divergence')

    args = parser.parse_args()

    if args.data == "dev":
        DIR_PATH = "/media/mafontai/SSD 2/data/speech_separation/spatialized_wsj0/" + str(args.n_speaker) + "speakers_anechoic/mix/"
        Name_file = glob.glob(os.path.join(DIR_PATH, "*.wav"))
        nb_file = args.nb_file
        EST_DIR = "/home/mafontai/Documents/project/git_project/speech_separation/alpha_SpatialMNMF/results_" + str(args.n_speaker) + "anechoic/"
    elif args.data == "test":
        DIR_PATH = "/media/mafontai/SSD 2/data/speech_separation/spatialized_wsj0/" + str(args.n_speaker) + "speakers_anechoic/mix/"
        Name_file = glob.glob(os.path.join(DIR_PATH, "*.wav"))
        nb_file = args.nb_file
        EST_DIR = "/home/mafontai/Documents/project/git_project/speech_separation/alpha_SpatialMNMF/results_" + str(args.n_speaker) + "anechoic/"

    SAVE_PATH = os.path.join(EST_DIR, args.data, "alpha=%s" % str(args.alpha), "beta=%s/" % str(args.beta))
    for id_file, name_file in enumerate(Name_file[:nb_file]):
        if args.gpu < 0:
            import numpy as xp
        else:
            import cupy as xp
            print("Use GPU " + str(args.gpu))
            xp.cuda.Device(args.gpu).use()
        # import ipdb; ipdb.set_trace()
        wav, fs = sf.read(name_file)
        wav = wav.T
        if args.determined:
            M = args.n_speaker
        else:
            M = 8
        file_path = os.path.join(SAVE_PATH,
                                 "alpha_SpatialMNMF-likelihood-interval={}-M={}-S={}-it={}-K={}-init={}-rand={}-ID={}.pic".format(str(args.n_inter), str(M), str(args.n_speaker), str(args.n_iteration), str(args.n_basis), str(args.init_SM), str(args.seed), str(id_file)))
        if os.path.exists(file_path):
            print("alpha_SpatialMNMF => interval={}-M={}-S={}-it={}-K={}-init={}-rand={}-ID={} already done.".format(str(args.n_inter), str(M), str(args.n_speaker), str(args.n_iteration), str(args.n_basis), str(args.init_SM), str(args.seed), str(id_file)))
            pass
        else:
            for m in range(M):
                tmp = librosa.core.stft(np.asfortranarray(wav[m]),
                                        n_fft=args.n_fft,
                                        hop_length=int(args.n_fft/4))
                if m == 0:
                    spec = np.zeros([tmp.shape[0],
                                     tmp.shape[1], M], dtype=np.complex)
                spec[:, :, m] = tmp
            separater = alpha_SpatialMNMF(n_source=args.n_speaker,
                                          alpha=args.alpha, DIR_PATH=name_file,
                                          beta=args.beta, n_basis=args.n_basis,
                                          xp=xp, init_SM=args.init_SM,
                                          seed=args.seed, n_Th=args.n_Th)
            separater.load_spectrogram(spec)
            separater.file_id = "%s" %str(id_file)

            if not os.path.exists(SAVE_PATH):
                os.makedirs(SAVE_PATH)
            separater.solve(n_iteration=args.n_iteration, save_likelihood=True,
                            save_parameter=True, save_wav=True,
                            save_path=SAVE_PATH,
                            interval_save_parameter=args.n_inter)
