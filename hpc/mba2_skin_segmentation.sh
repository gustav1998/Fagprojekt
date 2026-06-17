#!/bin/sh
#BSUB -q gpuv100
#BSUB -J mba2_skin_segmentation
#BSUB -n 4
### -- select 1 gpu in exclusive process mode with MPS --
#BSUB -gpu "num=1:mode=exclusive_process:mps=yes"
#BSUB -W 24:00
#BSUB -R "rusage[mem=5GB]"
#BSUB -R "span[hosts=1]"
#BSUB -u s245208@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -o mba2_skin_segmentation_%J.out
#BSUB -e mba2_skin_segmentation_%J.err

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
    --dataset skin_segmentation \
    --accelerator gpu \
    --num-workers 4

python3 -m src.training.run_experiments2 \
    --model mba \
    --dataset skin_segmentation \
    --seed 42 \
    --skip-preprocessing \
    --accelerator gpu \
    --num-workers 4

find src/summary_results/results/ -name "*.ckpt" -delete
git config user.email "s245208@dtu.dk"
git config user.name "Anya-Helle-Pritzl"
git add -f $(find src/summary_results/results/ -name "*.json")
git commit -m "MBA2 results: skin_segmentation"
for i in 1 2 3 4 5; do
    git fetch origin
    if git rebase origin/main && git push; then
        break
    fi
    git rebase --abort 2>/dev/null || true
    sleep 30
done
