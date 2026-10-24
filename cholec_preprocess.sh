#!/bin/bash
#SBATCH --job-name=preprocess
#SBATCH --partition=all_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/homes/mgualtieri01/cvcs2026/SurgPilot/logs/preprocess_%j.out
#SBATCH --error=/homes/mgualtieri01/cvcs2026/SurgPilot/logs/preprocess_%j.err
#SBATCH --account=cvcs2026

# Carica il modulo Python del CINECA
module load python/3.11.15

# Attiva il tuo ambiente virtuale locale
source /homes/mgualtieri01/cvcs2026/SurgPilot/venv/bin/activate

# Lancia lo script Python di filtraggio
python src/datasets/cholecSegFilter.py