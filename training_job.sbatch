#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=neutrino:cider-nu
#SBATCH --job-name=full0.9
#SBATCH --gpus 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=7:00:00
#SBATCH --output=dino_%j.txt
#SBATCH --error=dino_%j.txt

# Set container image path
export SINGULARITY_IMAGE_PATH=/sdf/group/neutrino/images/develop.sif

# Run your training script inside the container
singularity exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch \
    $SINGULARITY_IMAGE_PATH python hybrid_training.py