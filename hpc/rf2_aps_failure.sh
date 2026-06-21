#!/bin/sh
#BSUB -q hpc
#BSUB -J rf_aps
#BSUB -n 16
#BSUB -W 2:00
#BSUB -R "rusage[mem=2GB]"
#BSUB -R "span[hosts=1]"
#BSUB -u s245208@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -o rf2_aps_failure_%J.out
#BSUB -e rf2_aps_failure_%J.err

unset PYTHONHOME
unset PYTHONPATH
module load python3/3.11.9
export PATH=/appl9/python/3.11.9/bin:$PATH
cd ~/Fagprojekt
git fetch origin
git reset --hard origin/main
python3 -m pip install -e . --quiet

python3 -m src.training.tune_hyperparameters2 \
    --model rf \
    --dataset aps_failure

python3 -m src.training.run_experiments2 \
    --model rf \
    --dataset aps_failure \
    --seed 42 \
    --skip-preprocessing

find src/summary_results/results/rf/aps_failure* -name "model.joblib" -delete 2>/dev/null || true