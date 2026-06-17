#!/bin/sh
#BSUB -q gpuv100
#BSUB -J test_mba2_new
#BSUB -n 4
### -- select 1 gpu in exclusive process mode with MPS --
#BSUB -gpu "num=1:mode=exclusive_process:mps=yes"
#BSUB -W 4:00
#BSUB -R "rusage[mem=5GB]"
#BSUB -R "span[hosts=1]"
#BSUB -u s245208@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -o test_mba2_new_%J.out
#BSUB -e test_mba2_new_%J.err

unset PYTHONHOME
unset PYTHONPATH
module load python3/3.11.9
export PATH=/appl9/python/3.11.9/bin:$PATH
cd ~/Fagprojekt
git fetch origin
git reset --hard origin/main
python3 -m pip install -e . --quiet

# preprocess connect_4 into the new tuning + fold split format
python3 -m src.data_pipeline.make_dataset2 \
    --dataset connect_4 \
    --representation both

# tune MBA2 on the tuning split (20% of data)
python3 -m src.training.tune_hyperparameters2 \
    --model mba \
    --dataset connect_4 \
    --skip-preprocessing \
    --accelerator gpu \
    --num-workers 4 \
    --epochs 20

# run all 5 CV folds with the best tuned params
python3 -m src.training.run_experiments2 \
    --model mba \
    --dataset connect_4 \
    --seed 42 \
    --skip-preprocessing \
    --accelerator gpu \
    --num-workers 4 \
    --epochs 20

git config user.email "245208@dtu.dk"
git config user.name "Anya-Helle-Pritzl"
git add -f src/summary_results/results/
git commit -m "test: new MBA2 timing on connect_4 (5-fold CV)"
git fetch origin
git rebase origin/main
git push
