#!/bin/sh
#BSUB -q gpuv100
#BSUB -J test_mba_old
#BSUB -n 4
### -- select 1 gpu in exclusive process mode with MPS --
#BSUB -gpu "num=1:mode=exclusive_process:mps=yes"
#BSUB -W 2:00
#BSUB -R "rusage[mem=5GB]"
#BSUB -R "span[hosts=1]"
#BSUB -u 245208@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -o test_mba_old_%J.out
#BSUB -e test_mba_old_%J.err

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
    --model mba \
    --dataset connect_4 \
    --skip-preprocessing \
    --accelerator gpu \
    --num-workers 4 \
    --max-mba-order 3 \
    --max-mba-parameters 5000000 \
    --epochs 20

python3 -m src.training.run_experiments \
    --model mba \
    --dataset connect_4 \
    --seed 42 \
    --skip-preprocessing \
    --accelerator gpu \
    --num-workers 4 \
    --epochs 20

git config user.email "245208@dtu.dk"
git config user.name "Anya-Helle-Pritzl"
git add src/summary_results/results/
git commit -m "test: old MBA timing on connect_4"
git fetch origin
git rebase origin/main
git push
