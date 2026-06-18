#!/bin/sh
#BSUB -q gpuv100
#BSUB -J lr_cen
#BSUB -n 4
### -- select 1 gpu in exclusive process mode with MPS --
#BSUB -gpu "num=1:mode=exclusive_process:mps=yes"
#BSUB -W 1:00
#BSUB -R "rusage[mem=2GB]"
#BSUB -R "span[hosts=1]"
#BSUB -u s245208@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -o lr2_census_income_kdd_%J.out
#BSUB -e lr2_census_income_kdd_%J.err

unset PYTHONHOME
unset PYTHONPATH
module load python3/3.11.9
export PATH=/appl9/python/3.11.9/bin:$PATH
cd ~/Fagprojekt
git fetch origin
git reset --hard origin/main
python3 -m pip install -e . --quiet

python3 -m src.training.tune_hyperparameters2 \
    --model lr \
    --dataset census_income_kdd \
    --accelerator gpu \
    --num-workers 4

python3 -m src.training.run_experiments2 \
    --model lr \
    --dataset census_income_kdd \
    --seed 42 \
    --skip-preprocessing \
    --accelerator gpu \
    --num-workers 4

find src/summary_results/results/lr/census_income_kdd* -name "*.ckpt" -delete 2>/dev/null || true
find src/summary_results/results/tuning_census_income_kdd_lr* -name "*.ckpt" -delete 2>/dev/null || true
