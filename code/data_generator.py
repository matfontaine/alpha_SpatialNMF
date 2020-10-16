#! /usr/bin/env python3
# coding: utf-8

import numpy as np
import scipy.stats as st
import sys, os
import librosa
import soundfile as sf
import time
import pickle as pic

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
    parser.add_argument(         '--gpu', type= int, default=     0, help='GPU ID')
    parser.add_argument(    '--n_speaker', type= int, default=    2, help='number of sources')
    parser.add_argument( '--seed',   dest='seed', type=int, default=0,  help='random seed for experiments')
    parser.add_argument('--n_mic', type=int, default=20, help='number of microphones')

    args = parser.parse_args()

    DIR_PATH = "/media/mafontai/SSD 2/data/speech_separation/wsj0/data/"
    Name_file = glob.glob(os.path.join(DIR_PATH, "*.wav"))
    SAVE_PATH = os.path.join(DIR_PATH, 'mix/')


    if args.gpu < 0:
        import numpy as xp
    else:
        import cupy as xp
        print("Use GPU " + str(args.gpu))
        xp.cuda.Device(args.gpu).use()
    # import ipdb; ipdb.set_trace()

    # randomly draw indices in grid

    f_model = h5py.File('/media/mafontai/SSD 2/data/speech_separation/RIRmodel.mat')
    f_real = h5py.File('/media/mafontai/SSD 2/data/speech_separation/RIRreal.mat')

    A_TXYM_model = f_model[u'a_model'].value
    A_TXYM_real = f_real[u'a_real'].value
    if args.gpu >= 0:
        A_TXYM_real = xp.asarray(A_TXYM_real)
    fs = f_model[u'fs'].value
    pos_mics_real = f_model[u'pos_mics'].value
    xs_model = f_model[u'xs_model'].value
    ys_model = f_model[u'ys_model'].value
    xs_real = f_model[u'xs_model'].value
    ys_real = f_model[u'ys_model'].value

    y_index = [int(ys_real.size / 4), int(3 * ys_real.size / 4), int(ys_real.size / 4), int(3 * ys_real.size / 4)]
    x_index = [int(xs_real.size / 4), int(3 * xs_real.size / 4), int(xs_real.size / 4), int(3 * xs_real.size / 4)]
    # xs_realpos_index = rand_.randint(1, xs_real.size-2, size=args.n_speaker).astype(int)
    # ys_realpos_index = rand_.randint(1, ys_real.size-2, size=args.n_speaker).astype(int)

    for id in progressbar.progressbar(range(args.seed)):
        if id == 30:
            continue
        rand_ = np.random.RandomState(id)
        src_id = rand_.randint(0, len(Name_file) - 1., size=args.n_speaker).astype(int)
        for n in range(len(src_id)):
            filename = Name_file[src_id[n]]
            wav, fs = sf.read(filename)
            sig_room = xp.zeros((len(wav), args.n_mic)).astype(float)
            if args.gpu >= 0:
                wav = xp.asarray(wav)
            for m in range(args.n_mic):
                # if id == 30 and m == args.n_mic - 1:
                #     import ipdb; ipdb.set_trace()
                sig_room[:, m] = xp.convolve(A_TXYM_real[:, y_index[n], x_index[n], m], wav, mode='same')
            if args.gpu >= 0:
                sig_room = xp.asnumpy(sig_room)
            sf.write(filename[:-12] + "mix/audio-N={}-seed={}.wav".format(n, id), 4 * sig_room, fs)
