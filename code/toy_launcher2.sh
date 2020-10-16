N_ITER=1000
N_INTER=10
N_SAMPLE=10000
N_THETA=36
INIT="circ"

for ALPHA in 1.6 1.8 1.9; do for N_MIC in 2 4; do for BETA in 0.0; do for SOURCE in 2; do for SEED in $(seq 100);
do
  python toy_launcher_alpha.py --gpu 0 --n_mic ${N_MIC} --n_sample ${N_SAMPLE} --alpha ${ALPHA}  --beta ${BETA} \
  --n_iteration ${N_ITER} --n_inter ${N_INTER} --init_SM ${INIT} --n_Th ${N_THETA} \
  --n_speaker ${SOURCE} --seed ${SEED}
done; done; done; done; done
