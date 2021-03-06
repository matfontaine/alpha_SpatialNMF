N_ITER=200
N_INTER=10

for ALPHA in 1.4; do for BETA in 0.0; do for SOURCE in 2 3 4; do for MIC in 10 20 30; do for ID in $(seq 100)
do
  python ICA_launcher_alpha.py --gpu 0 --alpha ${ALPHA} --beta ${BETA}\
  --n_speaker ${SOURCE} --n_mic ${MIC} --id ${ID} --n_iteration ${N_ITER}\
  --n_inter ${N_INTER}

done; done; done; done; done
