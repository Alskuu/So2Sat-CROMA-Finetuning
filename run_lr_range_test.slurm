#!/bin/bash
#SBATCH --job-name=croma_lr_test
#SBATCH --output=logs/croma_lr_test_%j.out
#SBATCH --error=logs/croma_lr_test_%j.err
#SBATCH --account=ubn26na@v100          # Adapte la clé de projet (ex: @v100 ou @a100)
#SBATCH --qos=qos_gpu-dev               # Partition Dev (max 2h sur V100)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1             # 1 seule tâche pour le Range Test (Mono-GPU)
#SBATCH --cpus-per-task=10              # 10 CPUs (allocation standard sur Jean Zay)
#SBATCH -C v100-32g                     # V100 32Go (ou a100 si tu utilises @a100)
#SBATCH --gres=gpu:1                    # 1 GPU réservé
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --signal=B:SIGTERM@120

set -euo pipefail

# --- 1. Nettoyage de l'environnement ---
module purge
conda deactivate 2>/dev/null || true
module load pytorch-gpu/py3/2.4.0

# --- 2. Configuration des dossiers ($SCRATCH pour les E/S rapides) ---
export DATA_DIR=$SCRATCH/data/So2Sat-LCZ42
export CKPT_DIR=$SCRATCH/checkpoints_lr_rt
export BACKBONE_CKPT=$WORK/CROMA_weights/CROMA_base.pt

# Création des dossiers de sortie et de logs
mkdir -p "$CKPT_DIR" logs

# --- 3. Lancement des LR Range Tests ---
echo "================================================="
echo " Démarrage du LR Range Test pour CROMA sur Jean Zay"
echo " Dataset source : $DATA_DIR"
echo "================================================="

# Run 1 : Batch Size 32
echo ">>> Lancement pour Batch Size = 32..."
python -u script_lr_range_test.py \
    --data-dir "$DATA_DIR" \
    --checkpoint-dir "$CKPT_DIR" \
    --backbone-checkpoint "$BACKBONE_CKPT" \
    --batch-size 32 \
    --num-workers 8 \
    --head-lr 1e-3 \
    --ratios "10.0,100.0" \
    --num-iters 800 \
    --mixed-precision bfloat16

# Run 2 : Batch Size 64
echo ">>> Lancement pour Batch Size = 64..."
python -u script_lr_range_test.py \
    --data-dir "$DATA_DIR" \
    --checkpoint-dir "$CKPT_DIR" \
    --backbone-checkpoint "$BACKBONE_CKPT" \
    --batch-size 64 \
    --num-workers 8 \
    --head-lr 1e-3 \
    --ratios "10.0,100.0" \
    --num-iters 800 \
    --mixed-precision bfloat16

echo "================================================="
echo " Range Tests terminés ! Fichiers de résultats dans :"
echo " $CKPT_DIR"
echo "================================================="