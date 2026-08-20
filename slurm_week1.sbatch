#!/bin/bash
#SBATCH --job-name=mmimdb_w1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%x-%j.out
# 사용:
#   1) 특징 추출(1회):  sbatch --export=ALL,STAGE=extract,RAW=~/Yechan/mmimdb_data/mmimdb slurm_week1.sbatch
#   2) 학습+sanity:     sbatch --export=ALL,STAGE=train slurm_week1.sbatch
# ⚠️ 파티션 필요시: #SBATCH --partition=<name>
set -e
source ~/.bashrc
conda activate mmimdb
cd "$SLURM_SUBMIT_DIR"

STAGE=${STAGE:-train}
RAW=${RAW:-~/Yechan/mmimdb_data/mmimdb}
FEAT=${FEAT:-feats}

if [ "$STAGE" = "extract" ]; then
  python extract_features.py --raw_root "$RAW" --out "$FEAT"
else
  python train_week1.py --mode real --feat_dir "$FEAT" --epochs 30 --seed 1 --out results/week1_s1.json
fi
echo "== DONE =="
