N_ITER=5000
N_INTER=10
N_SAMPLE=200
N_THETA=60
INIT="ones"

for ALPHA in 1.8; do for N_MIC in 2; do for BETA in 0.0; do for SOURCE in 2; do for SEED in $(seq 1)
do
  python toy_launcher_alpha.py --gpu 0 --n_mic ${N_MIC} --n_sample ${N_SAMPLE} --alpha ${ALPHA}  --beta ${BETA} \
  --n_iteration ${N_ITER} --n_inter ${N_INTER} --init_SM ${INIT} --n_Th ${N_THETA} \
  --n_speaker ${SOURCE} --seed ${SEED}
done; done; done; done; done
