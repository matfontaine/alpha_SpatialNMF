#! /usr/bin/env python3
# coding: utf-8

from alpha_SpatialMNMF_toy import alpha_SpatialMNMF
try:
    FLAG_GPU_Available = True
except:
    print("---Warning--- You cannot use GPU acceleration because chainer or cupy is not installed")


if __name__ == "__main__":
    import argparse
    import os as os

    parser = argparse.ArgumentParser()
    parser.add_argument(         '--gpu', type= int, default=     0, help='GPU ID')
    parser.add_argument(       '--n_sample', type= int, default=  2048, help='number of samples')
    parser.add_argument(    '--n_speaker', type= int, default=    2, help='number of sources')
    parser.add_argument(    '--init_SM', type= str, default="unit", help='unit')
    parser.add_argument( '--n_iteration', type= int, default=   100, help='number of iteration')
    parser.add_argument( '--n_inter', type= int, default=  200, help='number of intervals')
    parser.add_argument( '--n_mic', type=int, default=2, help='number of channels')
    parser.add_argument( '--alpha',   dest='alpha', type=float, default=1.8,  help='Gaussian case (alpha=2)')
    parser.add_argument( '--seed',   dest='seed', type=int, default=0,  help='random seed for experiments')
    parser.add_argument('--data', type=str, default='dev', help='available: dev or test')
    parser.add_argument('--nb_file', type=int, default=1, help='nb of file to separate')
    parser.add_argument('--n_Th', type=int, default=180, help='number of sphere sampling')
    parser.add_argument( '--beta',   dest='beta', type=float, default=0,  help='beta divergence')

    args = parser.parse_args()
    EST_DIR = "/home/mafontai/Documents/project/git_project/speech_separation/alpha_SpatialMNMF/results_" + str(args.n_speaker) + "toy/"
    SAVE_PATH = os.path.join(EST_DIR, args.data, "alpha=%s" % str(args.alpha), "beta=%s/" % str(args.beta))
    if args.gpu < 0:
        import numpy as xp
    else:
        import cupy as xp
        print("Use GPU " + str(args.gpu))
        xp.cuda.Device(args.gpu).use()
    id_file = "test"
    file_path = os.path.join(SAVE_PATH,
                             "alpha_SpatialMNMF-likelihood-interval={}-M={}-S={}-it={}-init={}-rand={}-ID={}.pic".format(str(args.n_inter), str(args.n_mic), str(args.n_speaker), str(args.n_iteration), str(args.init_SM), str(args.seed), str(id_file)))
    if os.path.exists(file_path):
        print("alpha_SpatialMNMF => interval={}-M={}-S={}-it={}-init={}-rand={}-ID={} already done.".format(str(args.n_inter), str(args.n_mic), str(args.n_speaker), str(args.n_iteration), str(args.init_SM), str(args.seed), str(id_file)))
        pass
    else:
        separater = alpha_SpatialMNMF(n_source=args.n_speaker,
                                      alpha=args.alpha,
                                      beta=args.beta, n_sample=args.n_sample,
                                      n_mic=args.n_mic,
                                      xp=xp, init_SM=args.init_SM,
                                      seed=args.seed, n_Th=args.n_Th)
        separater.file_id = id_file

        if not os.path.exists(SAVE_PATH):
            os.makedirs(SAVE_PATH)
        separater.solve(n_iteration=args.n_iteration, save_likelihood=True,
                        save_parameter=True, save_wav=True,
                        save_path=SAVE_PATH,
                        interval_save_parameter=args.n_inter)
