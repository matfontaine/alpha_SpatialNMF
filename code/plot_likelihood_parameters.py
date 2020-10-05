import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle as pic
import sys, os
import numpy as np
import math as math

HYP = 1.4
BETA = 0.0
it = 500
S = 2
M = 2
K = 32
init = "circ"
EST_DIR_A = "/home/mafontai/Documents/project/git_project/speech_separation/alpha_SpatialMNMF/results_2anechoic/dev/"
SAVE_PATH_A = os.path.join(EST_DIR_A, "alpha=%s" % str(HYP), "beta=%s" % str(BETA))
file_path = os.path.join(SAVE_PATH_A, "alpha_SpatialMNMF-likelihood-interval=10-M={}-S={}-it={}-K={}-init={}-rand=1-ID=0.pic").format(str(M), str(S), str(it), str(K), init)
data_likelihood =  pic.load(open(file_path, 'rb'))
li_it = np.arange(1, it + 1, 10)

file_path = os.path.join(SAVE_PATH_A, "alpha_SpatialMNMF-parameters-M={}-S={}-it={}-K={}-init={}-rand=1-ID=0.npz").format(str(M), str(S), str(it), str(K), init)
file = np.load(file_path)

lambda_NFT = file['lambda_NFT']
lambda_true_NFT=file['lambda_true_NFT']
SM_NFP = file['SM_NFP']

fig_width_pt = 400.6937  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0 / 72.27               # Convert pt to inch
golden_mean = (math.sqrt(5) - 1.0) / 2.0         # Aesthetic ratio
fig_width = fig_width_pt * inches_per_pt  # width in inches
fig_height = fig_width * golden_mean      # height in inches
fig_size = np.array([(S + 2) * fig_width, 3. * fig_height])


fig = plt.figure(tight_layout=True, figsize=fig_size)
gs = gridspec.GridSpec(nrows=S + 2, ncols=3)




# beta divergence
ax = fig.add_subplot(gs[0, :])
ax.plot(li_it, data_likelihood)
ax.set(xlabel='number of iterations', ylabel='beta_div', title='beta-divergence (beta={})'.format(BETA))

for n in range(S):
    ax = fig.add_subplot(gs[n+1, 0])
    im = ax.imshow(np.log(lambda_true_NFT[n]), interpolation='nearest', origin='lower', aspect='auto')
    fig.colorbar(im, ax=ax)
    ax.set(xlabel='time frame', ylabel='frequency', title='true logPSD s{}'.format(n+1))

    ax = fig.add_subplot(gs[n+1, 1])
    im = ax.imshow(np.log(lambda_NFT[n]), interpolation='nearest', origin='lower', aspect='auto')
    fig.colorbar(im, ax=ax)
    ax.set(xlabel='time frame', ylabel='frequency', title='est logPSD s{}'.format(n+1))

    ax = fig.add_subplot(gs[n+1, 2])
    im = ax.imshow(SM_NFP[n], interpolation='nearest', origin='lower', aspect='auto')
    fig.colorbar(im, ax=ax)
    ax.set(xlabel='directions', ylabel='frequency', title='spatial measure s{}'.format(n+1))

ax = fig.add_subplot(gs[-1, 0])
im = ax.imshow(np.log(lambda_true_NFT.sum(axis=0)), interpolation='nearest', origin='lower', aspect='auto')
fig.colorbar(im, ax=ax)
ax.set(xlabel='time frame', ylabel='frequency', title='true logPSD x')

ax = fig.add_subplot(gs[-1, 1])
im = ax.imshow(np.log(lambda_NFT.sum(axis=0)), interpolation='nearest', origin='lower', aspect='auto')
fig.colorbar(im, ax=ax)
ax.set(xlabel='time frame', ylabel='frequency', title='est logPSD x')

ax = fig.add_subplot(gs[-1, 2])
im = ax.imshow(SM_NFP.sum(axis=0), interpolation='nearest', origin='lower', aspect='auto')
fig.colorbar(im, ax=ax)
ax.set(xlabel='directions', ylabel='frequency', title='spatial measure x')

fig.align_labels()
fig.subplots_adjust(wspace=0.2, hspace=0.7)
plt.savefig("results.png", bbox_inches='tight', dpi=300)
