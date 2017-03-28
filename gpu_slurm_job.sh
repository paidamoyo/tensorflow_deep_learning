#!/bin/bash
#SBATCH --job-name=train
#SBATCH --output=result.txt
#SBATCH -p gpu-common
#SBATCH --gres=gpu:1
#SBATCH -c 6
#SBATCH --mem=12g
uname -n && echo "The job has begun"
source .bashrc
pwd && cd research/tensorflow_deep_learning/ && echo  "directory change"
pyenv activate tensorflow && pip list && echo  "pyenv activated"
python train_auxiliary.py && echo  "The job completed"
