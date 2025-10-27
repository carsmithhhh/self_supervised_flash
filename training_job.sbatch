#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=mli:cider-ml
#SBATCH --job-name=50mask
#SBATCH --gpus 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=20:00:00
#SBATCH --output=dino_%j.txt
#SBATCH --error=dino_%j.txt

# Set container image path
export SINGULARITY_IMAGE_PATH=/sdf/group/neutrino/images/develop.sif

# Run your training script inside the container
singularity exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch \
    $SINGULARITY_IMAGE_PATH python vanilla_training.py