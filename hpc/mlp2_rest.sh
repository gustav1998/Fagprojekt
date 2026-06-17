#!/bin/sh
#BSUB -q gpuv100
#BSUB -J mlp_rest
#BSUB -n 4
### -- select 1 gpu in exclusive process mode with MPS --
#BSUB -gpu "num=1:mode=exclusive_process:mps=yes"
#BSUB -W 24:00
#BSUB -R "rusage[mem=5GB]"
#BSUB -R "span[hosts=1]"
#BSUB -u s245208@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -o mlp2_rest_%J.out
#BSUB -e mlp2_rest_%J.err

unset PYTHONHOME
unset PYTHONPATH
module load python3/3.11.9
export PATH=/appl9/python/3.11.9/bin:$PATH
cd ~/Fagprojekt
git fetch origin
git reset --hard origin/main
python3 -m pip install -e . --quiet

python3 -m src.training.tune_hyperparameters2 \
    --model mlp \
    --dataset asia_lung \
    --dataset balance_scale \
    --dataset car_evaluation \
    --dataset cleveland \
    --dataset conf_ad \
    --dataset connect_4 \
    --dataset coronary \
    --dataset credit_approval \
    --dataset dmft \
    --dataset german_gss \
    --dataset hayesroth \
    --dataset house_votes_84 \
    --dataset krkopt \
    --dataset led7 \
    --dataset mofn \
    --dataset nursery \
    --dataset parity5p5 \
    --dataset ppd \
    --dataset primary_tumor \
    --dataset ptumor \
    --dataset secondary_mushroom \
    --dataset sensorless_drive \
    --dataset sensory \
    --dataset shuttle \
    --dataset three_of_nine \
    --dataset vehicle \
    --dataset xd6 \
    --accelerator gpu \
    --num-workers 4

python3 -m src.training.run_experiments2 \
    --model mlp \
    --dataset asia_lung \
    --dataset balance_scale \
    --dataset car_evaluation \
    --dataset cleveland \
    --dataset conf_ad \
    --dataset connect_4 \
    --dataset coronary \
    --dataset credit_approval \
    --dataset dmft \
    --dataset german_gss \
    --dataset hayesroth \
    --dataset house_votes_84 \
    --dataset krkopt \
    --dataset led7 \
    --dataset mofn \
    --dataset nursery \
    --dataset parity5p5 \
    --dataset ppd \
    --dataset primary_tumor \
    --dataset ptumor \
    --dataset secondary_mushroom \
    --dataset sensorless_drive \
    --dataset sensory \
    --dataset shuttle \
    --dataset three_of_nine \
    --dataset vehicle \
    --dataset xd6 \
    --seed 42 \
    --skip-preprocessing \
    --accelerator gpu \
    --num-workers 4

find src/summary_results/results/ -name "*.ckpt" -delete
git config user.email "s245208@dtu.dk"
git config user.name "Anya-Helle-Pritzl"
git add -f $(find src/summary_results/results/ -name "*.json")
git commit -m "MLP2 results: remaining 27 datasets"
for i in 1 2 3 4 5; do
    git fetch origin
    if git rebase origin/main && git push; then
        break
    fi
    git rebase --abort 2>/dev/null || true
    sleep 30
done
