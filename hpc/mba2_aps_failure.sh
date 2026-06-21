#!/bin/sh
#BSUB -q gpuv100
#BSUB -J mba_aps
#BSUB -n 4
### -- select 1 gpu in exclusive process mode with MPS --
#BSUB -gpu "num=1:mode=exclusive_process:mps=yes"
#BSUB -W 8:00
#BSUB -R "rusage[mem=2GB]"
#BSUB -R "span[hosts=1]"
#BSUB -u s245208@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -o mba2_aps_failure_%J.out
#BSUB -e mba2_aps_failure_%J.err

unset PYTHONHOME
unset PYTHONPATH
module load python3/3.11.9
export PATH=/appl9/python/3.11.9/bin:$PATH
cd ~/Fagprojekt
git fetch origin
git reset --hard origin/main
python3 -m pip install -e . --quiet

python3 -m src.training.tune_hyperparameters2 \
    --model mba \
    --dataset aps_failure \
    --accelerator gpu \
    --num-workers 4

python3 -m src.training.run_experiments2 \
    --model mba \
    --dataset aps_failure \
    --seed 42 \
    --skip-preprocessing \
    --accelerator gpu \
    --num-workers 4

find src/summary_results/results/mba/aps_failure* -name "*.ckpt" -delete 2>/dev/null || true