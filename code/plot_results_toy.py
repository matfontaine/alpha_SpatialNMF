import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle as pic
import sys, os
import numpy as np
import math as math

HYP = 1.8
BETA = 0.0
it = 5000
S = 2
M = 2
K = 32
init = "ones"
EST_DIR_A = "/home/mafontai/Documents/project/git_project/speech_separation/alpha_SpatialMNMF/results_2toy/dev/"
SAVE_PATH_A = os.path.join(EST_DIR_A, "alpha=%s" % str(HYP), "beta=%s" % str(BETA))
file_path = os.path.join(SAVE_PATH_A, "alpha_SpatialMNMF-likelihood-interval=10-M={}-S={}-it={}-init={}-rand=1-ID=test.pic").format(str(M), str(S), str(it), init)
data_likelihood =  pic.load(open(file_path, 'rb'))
li_it = np.arange(1, it + 1, 10)

file_path = os.path.join(SAVE_PATH_A, "alpha_SpatialMNMF-parameters-M={}-S={}-it={}-init={}-rand=1-ID=test.npz").format(str(M), str(S), str(it), init)
file = np.load(file_path)

lambda_NT = file['lambda_NT']
lambda_true_NT = file['lambda_true_NT']
SM_NP = file['SM_NP']
SM_true_NP = file['SM_true_NP']
Y_true_NTM = file['Y_true_NTM']
Y_NTM = file['Y_NTM']


fig_width_pt = 400.6937  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0 / 72.27               # Convert pt to inch
golden_mean = (math.sqrt(5) - 1.0) / 2.0         # Aesthetic ratio
fig_width = fig_width_pt * inches_per_pt  # width in inches
fig_height = fig_width * golden_mean      # height in inches
fig_size = np.array([(S + 2) * fig_width, 3. * fig_height])


fig = plt.figure(tight_layout=True, figsize=fig_size)
gs = gridspec.GridSpec(nrows=S + 2, ncols=4)


# beta divergence
ax = fig.add_subplot(gs[0, :])
ax.plot(li_it, data_likelihood)
ax.set(xlabel='number of iterations', ylabel='beta_div', title='beta-divergence (beta={})'.format(BETA))

for n in range(S):
    ax = fig.add_subplot(gs[n+1, 0])
    ax.plot(lambda_true_NT[n])
    ax.set(xlabel='sample', ylabel='value', title='true lambda s{}'.format(n+1))

    ax = fig.add_subplot(gs[n+1, 1])
    ax.plot(lambda_NT[n])
    ax.set(xlabel='sample', ylabel='value', title='est lambda s{}'.format(n+1))

    ax = fig.add_subplot(gs[n+1, 2])
    ax.plot(SM_NP[n], label='est spatial measure s{}'.format(n+1))
    ax.plot(SM_true_NP[n], label='true spatial measure s{}'.format(n+1))
    ax.legend()
    ax.set(xlabel='directions', ylabel='value', title='spatial measure s{}'.format(n+1))

    ax = fig.add_subplot(gs[n+1, 3])
    ax.scatter(np.abs(Y_true_NTM[n, :, 0]), np.abs(Y_true_NTM[n, :, 1]), label='true x{}'.format(n+1))
    ax.scatter(np.abs(Y_NTM[n, :, 0]), np.abs(Y_NTM[n, :, 1]), label='est x{}'.format(n+1))
    ax.legend()
    ax.set(xlabel='1st component', ylabel='2nd component', title='true and est x{}'.format(n+1))


ax = fig.add_subplot(gs[-1, 0])
ax.plot(lambda_true_NT.sum(axis=0))
ax.set(xlabel='sample', ylabel='value', title='true lambda obs')

ax = fig.add_subplot(gs[-1, 1])
ax.plot(lambda_NT.sum(axis=0))
ax.set(xlabel='sample', ylabel='value', title='est lambda obs')

ax = fig.add_subplot(gs[-1, 2])
ax.plot(SM_true_NP.sum(axis=0), label='true obs')
ax.plot(SM_NP.sum(axis=0), label='est obs')
ax.legend()
ax.set(xlabel='directions', ylabel='value', title='spatial measure obs')

ax = fig.add_subplot(gs[-1, 3])
ax.scatter(np.abs(Y_true_NTM[..., 0]).sum(axis=0), np.abs(Y_true_NTM[..., 1]).sum(axis=0), label='true obs')
ax.scatter(np.abs(Y_NTM[..., 0]).sum(axis=0), np.abs(Y_NTM[..., 1]).sum(axis=0), label='est obs')
ax.legend()
ax.set(xlabel='1st component', ylabel='2nd component', title='true and est obs')


fig.align_labels()
fig.subplots_adjust(wspace=0.2, hspace=0.7)
plt.savefig("results_toy.png", bbox_inches='tight', dpi=300)
