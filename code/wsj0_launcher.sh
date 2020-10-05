N_ITER=500
N_INTER=10
N_THETA=2
INIT="circ"

for LATENT_DIM in 32; do for ALPHA in 1.4; do for BETA in 0; do for SOURCE in 2; do for SEED in $(seq 1)
do
  python wsj0_launcher_alpha.py --gpu 0 --n_basis ${LATENT_DIM} --alpha ${ALPHA}  --beta ${BETA} \
  --n_iteration ${N_ITER} --n_inter ${N_INTER} --init_SM ${INIT} --n_Th ${N_THETA} \
  --n_speaker ${SOURCE} --seed ${SEED} --determined
done; done; done; done; done
