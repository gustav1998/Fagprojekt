#!/bin/sh
#BSUB -q gpuv100
#BSUB -J tt_iot
#BSUB -n 4
### -- select 1 gpu in exclusive process mode with MPS --
#BSUB -gpu "num=1:mode=exclusive_process:mps=yes"
#BSUB -W 24:00
#BSUB -R "rusage[mem=5GB]"
#BSUB -R "span[hosts=1]"
#BSUB -u s245208@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -o tt2_rt_iot2022_%J.out
#BSUB -e tt2_rt_iot2022_%J.err

unset PYTHONHOME
unset PYTHONPATH
module load python3/3.11.9
export PATH=/appl9/python/3.11.9/bin:$PATH
cd ~/Fagprojekt
git fetch origin
git reset --hard origin/main
python3 -m pip install -e . --quiet

python3 -m src.training.tune_hyperparameters2 \
    --model tt \
    --dataset rt_iot2022 \
    --accelerator gpu \
    --num-workers 4

python3 -m src.training.run_experiments2 \
    --model tt \
    --dataset rt_iot2022 \
    --seed 42 \
    --skip-preprocessing \
    --accelerator gpu \
    --num-workers 4

find src/summary_results/results/tt/rt_iot2022* -name "*.ckpt" -delete 2>/dev/null || true
find src/summary_results/results/tuning_rt_iot2022_tt* -name "*.ckpt" -delete 2>/dev/null || true
