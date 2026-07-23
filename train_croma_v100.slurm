#!/bin/bash
#SBATCH --job-name=croma_ft
#SBATCH --output=logs/croma_ft_%j.out
#SBATCH --error=logs/croma_ft_%j.out
#SBATCH --qos=qos_gpu-dev                 # QoS max 2h sur du V100
#SBATCH --partition=gpu_p2                # Cas des 32 Go de VRAM
#SBATCH --nodes=1                       # augmente si tu veux du multi-noeud (le mapping RANK/WORLD_SIZE plus bas s'adapte automatiquement)
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=6                # 1/8 du noeud par GPU sur gpu_p5, ne pas changer même si ntasks-per-node < 8
#SBATCH --account=rsn@v100          # Adapte la clé de projet (ex: @v100 ou @a100)
#SBATCH --gres=gpu:2
#SBATCH --hint=nomultithread
#SBATCH --time=01:30:00                  # max autorisé sur la partition qos_gpu-dev
#SBATCH --signal=B:SIGTERM@180           # 120s de préavis avant kill -> laisse le temps de checkpointer proprement
#SBATCH --requeue                        # combiné au signal ci-dessus : le job est automatiquement resoumis après préemption

set -euo pipefail   # c'est une commande de sécurité en bash, si une commande échoue ou qu'une variable est mal définie : arrêt immédiat

module purge                           # efface toutes les bibliothèquese logicielles chargées par défaut pour démarrer sur une base propre
conda deactivate 2>/dev/null || true   # désactive conda et si on ne trouve déjà pas conda, no pb on continue, il n'y a pas de pipefail
module load pytorch-gpu/py3/2.4.0        

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29500

export DATA_DIR=$SCRATCH/data/LCZ42/                       
export CKPT_DIR=$WORK/So2Sat-CROMA-Finetuning/checkpoints/          
export BACKBONE_CKPT=$WORK/CROMA_weights/CROMA_base.pt 

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True # Gestion dynamique de la VRAM sous Linux pour éviter les OOM dus à des blocs non contigus

srun bash -c '
    export RANK=$SLURM_PROCID
    export LOCAL_RANK=$SLURM_LOCALID
    export WORLD_SIZE=$SLURM_NTASKS 
    python -u script_finetuning.py \
        --data-dir "$DATA_DIR" \
        --checkpoint-dir "$CKPT_DIR" \
        --backbone-checkpoint "$BACKBONE_CKPT" \
        --batch-size 64 \
        --num-workers 4
    # --resume est activé par défaut (cfg.resume=True) : si $CKPT_DIR/last.pt existe déjà
    # (ex: après un --requeue suite à une préemption), le run reprendra automatiquement.
'
