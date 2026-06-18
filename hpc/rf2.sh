#!/bin/sh
#BSUB -q hpc
#BSUB -J rf
#BSUB -n 16
#BSUB -W 12:00
#BSUB -R "rusage[mem=4GB]"
#BSUB -R "span[hosts=1]"
#BSUB -u s245208@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -o rf2_%J.out
#BSUB -e rf2_%J.err

unset PYTHONHOME
unset PYTHONPATH
module load python3/3.11.9
export PATH=/appl9/python/3.11.9/bin:$PATH
cd ~/Fagprojekt
git fetch origin
git reset --hard origin/main
python3 -m pip install -e . --quiet

python3 -m src.training.tune_hyperparameters2 \
    --model rf \
    --dataset asia_lung \
    --dataset balance_scale \
    --dataset car_evaluation \
    --dataset census_income_kdd \
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
    --dataset phiusiil_phishing \
    --dataset ppd \
    --dataset primary_tumor \
    --dataset ptumor \
    --dataset rt_iot2022 \
    --dataset secondary_mushroom \
    --dataset sensorless_drive \
    --dataset sensory \
    --dataset shuttle \
    --dataset skin_segmentation \
    --dataset three_of_nine \
    --dataset vehicle \
    --dataset xd6

python3 -m src.training.run_experiments2 \
    --model rf \
    --dataset asia_lung \
    --dataset balance_scale \
    --dataset car_evaluation \
    --dataset census_income_kdd \
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
    --dataset phiusiil_phishing \
    --dataset ppd \
    --dataset primary_tumor \
    --dataset ptumor \
    --dataset rt_iot2022 \
    --dataset secondary_mushroom \
    --dataset sensorless_drive \
    --dataset sensory \
    --dataset shuttle \
    --dataset skin_segmentation \
    --dataset three_of_nine \
    --dataset vehicle \
    --dataset xd6 \
    --seed 42 \
    --skip-preprocessing

find src/summary_results/results/rf/ -name "model.joblib" -delete 2>/dev/null || true
