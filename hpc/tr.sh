#!/bin/sh
#BSUB -q gpuv100
#BSUB -J tr
#BSUB -n 12
### -- select 1 gpu in exclusive process mode with MPS --
#BSUB -gpu "num=1:mode=exclusive_process:mps=yes"
#BSUB -W 24:00
#BSUB -R "rusage[mem=5GB]"
#BSUB -u 245208@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -o tr_%J.out
#BSUB -e tr_%J.err

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
    --model tr \
    --skip-preprocessing \
    --accelerator gpu \
    --num-workers 4

python3 -m src.training.run_experiments \
    --model tr \
    --seed 42 \
    --skip-preprocessing \
    --accelerator gpu \
    --num-workers 4

git config user.email "<your-dtu-email>"
git config user.name "<your-name>"
git add src/summary_results/results/
git commit -m "TR results: all datasets"
git fetch origin
git rebase origin/main
git push
