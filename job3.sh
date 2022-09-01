#!/bin/bash
#SBATCH --job-name=Unet1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH -o /home/%u/%j-%x-on-%N.out
#SBATCH -e /home/%u/%j-%x-on-%N.err
#SBATCH --mail-type=ALL
#Timelimit format: "hours:minutes:seconds" -- max is 24h
#SBATCH --time=24:00:00

export WORKON_HOME=/cluster/to50jego/.cache
export XDG_CACHE_DIR=/cluster/to50jego/.cache
export PYTHONUSERBASE=/cluster/to50jego/.python_packages
export PATH=/opt/miniconda/bin:$PATH

echo "Running on" $(hostname)

pip3 install --user -r /cluster/to50jego/Cell_segmentation/cluster_requirements.txt

python3 ./src/train.py
