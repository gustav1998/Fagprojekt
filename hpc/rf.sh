#!/bin/sh
#BSUB -q hpc
#BSUB -J rf
#BSUB -n 16
#BSUB -W 8:00
#BSUB -R "rusage[mem=4GB]"
#BSUB -R "span[hosts=1]"
#BSUB -u 245208@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -o rf_%J.out
#BSUB -e rf_%J.err

unset PYTHONHOME
unset PYTHONPATH
module load python3/3.11.9
export PATH=/appl9/python/3.11.9/bin:$PATH
cd ~/Fagprojekt
git fetch origin
git reset --hard origin/main
python3 -m pip install -e . --quiet

python3 -m src.training.tune_hyperparameters \
    --model rf \
    --skip-preprocessing

python3 -m src.training.run_experiments \
    --model rf \
    --seed 42 \
    --skip-preprocessing

git config user.email "<your-dtu-email>"
git config user.name "<your-name>"
git add src/summary_results/results/
git commit -m "RF results: all datasets"
git fetch origin
git rebase origin/main
git push
