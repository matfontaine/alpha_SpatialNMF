#version python 24/10



# import pylab as plt
import numpy as np
from basics import stft, wav, loadFile
from spatial_measure import sketch, compute_psi_old, compute_psi, nmf_old, nmf
from SRP import SRP_fast
from CLEAN import CLEAN
from numpy.linalg import norm
import tqdm as tqdm


######################### Parameters ###############################################################
J = 4 #number of sources
arrayK =[30]

minF = 1000
maxF = 3000
nb_group = 90 # linear subdivision

maxLength = 15
nfft = 2048 # number of window
overlap = 0.4 # 0.xx = xx% of overlap between windows
hop = float(nfft) * (1.0 - overlap)

alpha = 2.
beta = 0
eps = 1e-15
Lambda = 0
nb_it = 70

nb_simulations = 10

use_cython = True
old = True
##########################Load RIRs#################################################################

print()'Load RIRs')
K_model, A_model, c, fs, orientations_mics, pos_mics, size_room, spacing_model, xs_model, ys_model = \
loadFile.import_model('RIRmodel.mat')

K_real, A_real, c, delta_real, fs_real, orientations_mics_real, pos_mics_real, size_room_real, spacing_real, xs_real, \
ys_real =  loadFile.import_real('RIRreal.mat')
print ('Load Complete !'

#number of sources positions candidates
L = xs_model.size * ys_model.size

maxF = int(maxF * nfft / fs)
minF = int(minF * nfft / fs)

dF = maxF - minF
Fgroup = np.linspace(0, dF, num=nb_group + 1, dtype=np.int)

print "nb of sources: %d" % (J)
for K in arrayK:
    for it_mc in range(nb_simulations, nb_simulations +1):
        print "it MonteCarlo : %d" % (it_mc)
        ##########################Computation of the convolution################################################################

        '''draw sources positions'''
        rand_ = np.random.RandomState(it_mc)

        #randomly draw indices in grid
        xs_realpos_index = rand_.randint(1, xs_real.size-2, size = J).astype(int)
        ys_realpos_index = rand_.randint(1, ys_real.size-2, size = J).astype(int)

        #compute corresponding ground truth positions
        xs_realpos = xs_real[xs_realpos_index]
        ys_realpos = ys_real[ys_realpos_index]

        #get corresponding closest positions in the model grid
        xs_modelpos_index = [np.argmin((xs_model-xs_realpos[j]).flatten()**2) for j in range(J)]
        ys_modelpos_index = [np.argmin((ys_model-ys_realpos[j]).flatten()**2) for j in range(J)]

        '''generate observations'''
        #convolve sources by filters in the time domain
        size_sig = int(maxLength*fs)
        sig_room = np.zeros((J, size_sig, K)).astype(np.float32)

        for j in tqdm.tqdm(range(J), desc='Compute convolution'):
            FileName = "s%d.wav" % (j+1)
            (sig_temp, fs) = wav.wavread(FileName, maxLength)
            #sig_temp /= norm(sig_temp)  # Commented otherwise clean is fucked.
            for k in range(K):
                sig_room[j, :, k] = np.convolve(A_real[:, ys_realpos_index[j], xs_realpos_index[j], k], sig_temp[:, 0], mode='same')

        #compute STFT of observations
        X_t = np.sum(sig_room, axis=0) # this is our observation
        X = stft.stft(X_t, nfft, hop, real=True).astype(np.complex64)

        (F,T,K) = X.shape

        ############################################# Compute the propagation model ##########################################################


        print 'using %d frequencies'%dF

        ############################################# Compute the fft ##########################################################
        A = np.zeros((dF, ys_model.size, xs_model.size, K)).astype(np.complex64)
        for k in tqdm.tqdm(range(K), desc='Compute fft of A'):
            A[..., k] = np.fft.rfft(A_model[..., k],nfft, axis=0)[minF:maxF]

        A = np.reshape(A, (dF, L, K))

        # normalize RIR
        #Anorm = A
        Anorm = A/norm(A, axis=-1, keepdims=True)

        ########################################### Estimate directions with SRP ###################################
        print "1st method: Beamforming"

        SRP = np.zeros((L,), np.float64)
        SRP_hat = SRP_fast(SRP, X[minF:maxF,...].copy(), Anorm.copy(), dF, T, L, K)
        SRP_hat /= SRP_hat.max()


        ########################################### Estimate directions with spatial measure ###################################
        print "2nd method: alpha-stable"

        Uconj = np.conj(Anorm)  # sketching frequencies

        gamma = SRP_hat # init with all directions = 1
        gamma = np.reshape(SRP_hat, (L,1)).astype(np.float32)

        # A CALCULER UNE FOIS SEULEMENT POUR UNE CONFIG D ANTENNE DONNEE
        if it_mc == 0 or it_mc == nb_simulations:
              if old:
                Psi = compute_psi_old(Uconj.copy(),Anorm.copy(),alpha)

              else:
                Psi = compute_psi(Uconj.copy(), Anorm.copy(), alpha, Fgroup.copy())
                Psi = np.reshape(Psi,(nb_group * L, L)).astype(np.float32)



        I = sketch(X[minF:maxF, ...].copy(), Uconj.copy(), alpha)

        if (not old):
            I_slice = np.zeros((nb_group, L)).astype(np.float32)

            for index, f in enumerate(Fgroup):
                if index != nb_group:
                    range_batch = slice(Fgroup[index], Fgroup[index+1])
                    I_slice[index, :] = np.sum(I[range_batch, :], axis=0)
            I_slice = np.reshape(I_slice, (nb_group * L, 1)).astype(np.float32)

            gamma = nmf(I_slice.copy(), Psi.copy(), gamma.copy(), beta, Lambda, nb_it, eps)

        else:
            gamma = nmf_old(I.copy(), Psi.copy(), gamma.copy(), beta, Lambda, nb_it, eps)

        gamma /= gamma.max()


        # print "3rd method : CLEAN algorithm"
        #
        # P_CLEAN = CLEAN(J*2, X[minF:maxF,...], A, rho, dF, T, L, K)
        ########################################### display ###################################

        np.savez('results_others/MC_it%03dK%dJ%dalpha%d.npz' % (it_mc, K, J, alpha), gamma = gamma, SRP = SRP_hat, K=K,
                 xs_model=xs_model, ys_model=ys_model,
                 xs_modelpos_index=xs_modelpos_index, ys_modelpos_index=ys_modelpos_index,
                 F=F, T=T, J=J, alpha = alpha)
