#!/bin/sh
#BSUB -q gpuv100
#BSUB -J tr_cen_fold5
#BSUB -n 4
### -- select 1 gpu in exclusive process mode with MPS --
#BSUB -gpu "num=1:mode=exclusive_process:mps=yes"
#BSUB -W 2:00
#BSUB -R "rusage[mem=2GB]"
#BSUB -R "span[hosts=1]"
#BSUB -u s245208@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -o tr2_census_income_kdd_fold5_%J.out
#BSUB -e tr2_census_income_kdd_fold5_%J.err

unset PYTHONHOME
unset PYTHONPATH
module load python3/3.11.9
export PATH=/appl9/python/3.11.9/bin:$PATH
cd ~/Fagprojekt
git fetch origin
git reset --hard origin/main
python3 -m pip install -e . --quiet

python3 -m src.training.train2 \
    --dataset census_income_kdd \
    --model tr \
    --batch-size 512 \
    --num-workers 4 \
    --accelerator gpu \
    --seed 42 \
    --mode fold \
    --fold 5 \
    --epochs 60 \
    --learning-rate 0.0001 \
    --rank 32

find src/summary_results/results/tr/census_income_kdd_fold5_seed42 -name "*.ckpt" -delete 2>/dev/null || true