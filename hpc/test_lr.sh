#!/bin/sh
#BSUB -q gpuv100
#BSUB -J test_lr
#BSUB -n 4
### -- select 1 gpu in exclusive process mode with MPS --
#BSUB -gpu "num=1:mode=exclusive_process:mps=yes"
#BSUB -W 1:00
#BSUB -R "rusage[mem=5GB]"
#BSUB -R "span[hosts=1]"
#BSUB -u 245208@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -o test_lr_%J.out
#BSUB -e test_lr_%J.err

unset PYTHONHOME
unset PYTHONPATH
module load python3/3.11.9
export PATH=/appl9/python/3.11.9/bin:$PATH
cd ~/Fagprojekt
git fetch origin
git reset --hard origin/main
python3 -m pip install "torch<2.5.0" --index-url https://download.pytorch.org/whl/cu118 --quiet
python3 -m pip install -e . --quiet

python3 -m src.training.tune_hyperparameters \
    --model lr \
    --dataset house_votes_84 \
    --skip-preprocessing \
    --accelerator gpu \
    --num-workers 2

python3 -m src.training.run_experiments \
    --model lr \
    --dataset house_votes_84 \
    --seed 1 --seed 2 --seed 3 --seed 4 --seed 5 --seed 42 \
    --skip-preprocessing \
    --accelerator gpu \
    --num-workers 2
