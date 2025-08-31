#!/bin/bash
#SBATCH -p mcml-hgx-a100-80x4
#SBATCH --qos=mcml
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH -t 48:00:00

source $HOME/miniforge3/bin/activate
eval "$(mamba shell hook --shell bash)"

mamba activate ehealth

python ehealth/train.py
