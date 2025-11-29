#!/bin/bash
#SBATCH --job-name=AIGC
#SBATCH --nodes=1
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=coursework
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G       
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_90
#SBATCH --mail-user=yiuctsoi2-c@my.cityu.edu.hk   # Optional

# ==================== Directory Setup ====================
cd $HOME/CS4487-course-project

# Create log directory if not exists
mkdir -p logs

# ==================== Environment ====================
source venv/bin/activate          # Activate your Python 3.9 venv
export PYTHONPATH=$PWD:$PYTHONPATH
pip install -r requirements.txt   # Install requirements

# ==================== Training Command ====================

# Stage 1: Train backbones (36 epochs)
python train.py

echo "Job finished at $(date)"