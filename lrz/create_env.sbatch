#!/bin/bash
#SBATCH -p mcml-dgx-a100-40x8
#SBATCH --qos=mcml
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH -t 00:15:00

source $HOME/miniforge3/bin/activate
eval "$(mamba shell hook --shell bash)"

mamba create -n ehealth python=3.13.5 -y
mamba activate ehealth

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install transformers tensorboard simpleitk datasets protobuf==3.20.1 numpy pandas python-dotenv huggingface_hub peft trl evaluate scikit-learn pillow nltk rouge-score hf-xet bitsandbytes 
