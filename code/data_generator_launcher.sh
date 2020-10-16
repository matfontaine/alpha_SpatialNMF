for SOURCE in 4; do for SEED in 400; do for MIC in 30
do
  python data_generator.py --gpu 0 --n_speaker ${SOURCE} --seed ${SEED} --n_mic ${MIC}
done; done; done
