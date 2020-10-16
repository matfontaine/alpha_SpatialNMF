import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.transforms import BlendedGenericTransform
import matplotlib.gridspec as gridspec
import pickle as pic
import sys, os
import numpy as np
import math as math

import pandas as pd
import seaborn as sns
HYP = [1.1, 1.2, 1.4, 1.6, 1.8, 1.9]
BETA = 0.0
it = 1000
rand = 100
S = 2
M = [2, 4]
init = "circ"
datas = pd.DataFrame(columns=['alpha', 'M', 'error_name', 'score'])
EST_DIR_A = "/home/mafontai/Documents/project/git_project/speech_separation/alpha_SpatialMNMF/results_" + str(S) + "toy/dev/"


sns.set_style("whitegrid")
# cste = 12
# params = {
#     'backend': 'ps',
#     'axes.labelsize': cste,
#     'font.size': cste,
#     'legend.fontsize': cste,
#     'xtick.labelsize': cste,
#     'ytick.labelsize': cste,
#     'text.usetex': True,
#     'font.family': 'serif',
#     'font.serif': 'ptmrr8re',
# }
#
# sns.set_style("whitegrid", {
#     'pgf.texsystem': 'xelatex',  # pdflatex, xelatex, lualatex
#     'text.usetex': True,
#     'font.family': 'serif',
#     'axes.labelsize': cste,
#     'legend.labelspacing':0,
#     'legend.borderpad':0,
#     'font.size': cste,
#     'legend.fontsize': cste,
#     'xtick.labelsize': cste,
#     'ytick.labelsize': cste,
#     'font.serif': [],
# })
# plt.rcParams.update(params)

fig_width_pt = 400.6937  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0 / 72.27               # Convert pt to inch
golden_mean = (math.sqrt(5) - 1.0) / 2.0         # Aesthetic ratio
fig_width = fig_width_pt * inches_per_pt  # width in inches
fig_height = fig_width * golden_mean      # height in inches
fig_size = np.array([2. * fig_width, 2. * fig_height])
fig_size2 = np.array([fig_width, fig_height + 0.042 * fig_height])
height = 5
capsize = 0.8
errwidth = 0
aspect = 1.


#
# for n_al, alpha in enumerate(HYP):
#     for n_m, mic in enumerate(M):
#         for id in range(2, rand - 10):
#             SAVE_PATH_A = os.path.join(EST_DIR_A, "alpha=%s" % str(alpha), "beta=%s" % str(BETA))
#             file_path = os.path.join(SAVE_PATH_A, "alpha_SpatialMNMF-parameters-M={}-S={}-it={}-init={}-rand={}-ID=test.npz").format(
#             str(mic), str(S), str(it), init, str(id))
#             file = np.load(file_path)
#             print('{}'.format(file_path))
#             lambda_NT = file['lambda_NT']
#             lambda_true_NT = file['lambda_true_NT']
#             SM_NP = file['SM_NP']
#             SM_true_NP = file['SM_true_NP']
#             Y_true_NTM = file['Y_true_NTM']
#             Y_NTM = file['Y_NTM']
#
#             SM_error = 100. * (np.abs(SM_true_NP-SM_NP).sum(axis=0).mean() / np.abs(SM_true_NP+ 1e-14).sum(axis=0).mean())
#             PSD_error = 100. * (np.abs(lambda_true_NT - lambda_NT).sum(axis=0).mean() / np.abs(lambda_true_NT + 1e-14).sum(axis=0).mean())
#             DATA_error = 100. * (np.abs(Y_true_NTM-Y_NTM).sum(axis=0).mean() / np.abs(Y_true_NTM + 1e-14).sum(axis=0).mean())
#
#             dict_SM_error = {'alpha':alpha,
#                         'M': mic,
#                         'error_name': "SM_error",
#                         'score': SM_error
#                             }
#
#             dict_PSD_error = {'alpha':alpha,
#                         'M': mic,
#                         'error_name': "PSD_error",
#                         'score': PSD_error
#                             }
#
#             dict_DATA_error = {'alpha':alpha,
#                         'M': mic,
#                         'error_name': "DATA_error",
#                         'score': DATA_error
#                             }
#             datas = datas.append(dict_SM_error,
#                      ignore_index=True)
#             datas = datas.append(dict_PSD_error,
#                      ignore_index=True)
#             datas = datas.append(dict_DATA_error,
#                      ignore_index=True)
#
# datas.to_pickle('./results_alphaSpatialtoy.pic')


datas = pd.read_pickle("./results_alphaSpatialtoy.pic")
fig_size = np.array([fig_width, (len(M)) * fig_height])
f, ax = plt.subplots(2, 1, figsize=fig_size)
for i_mic, mic in enumerate(M):
    tmp_data = datas.loc[(datas['M'] == mic)]
    tmp_data['score'] = tmp_data['score'].astype(float)
    # ax[i_src].yaxis.grid(True)
    g1 = sns.boxplot(x="alpha", y="score", hue='error_name',
                     palette="colorblind", data=tmp_data, ax=ax[i_mic],
                     showmeans=True, showfliers=False,
                     meanprops={"marker":"s","markerfacecolor":"white", "markeredgecolor":"black"})

    ax[i_mic].set(ylabel='relative absolute error (in %)', xlabel='alpha, M={}'.format(mic))

    if i_mic == 0:
        g1.legend_.remove()

    else:
        lgd = ax[i_mic].legend(
        loc='upper center',
        bbox_to_anchor=(0.5, 2.34),
        labelspacing=0,
        borderpad=0.2,
        bbox_transform=BlendedGenericTransform(f.transFigure, ax[i_mic].transAxes),
        ncol=3)

    # plt.gcf().autofmt_xdate()
    plt.savefig("separation_alphaSpatial.pdf", bbox_inches='tight', dpi=300)
    plt.savefig("separation_alphaSpatial.png", bbox_inches='tight', dpi=300)


plt.savefig("results_toy.png", bbox_inches='tight', dpi=300)
