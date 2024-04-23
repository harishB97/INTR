#!/bin/bash

#SBATCH --account=imageomicswithanuj
#SBATCH --partition=a100_normal_q
#SBATCH --time=15-00:00:00 
#SBATCH --gres=gpu:1
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8
#SBATCH -o ./SLURM/slurm-%j.out


echo start load env and run python

module reset
module load Anaconda3/2020.11
source activate hpnet4
module reset
source activate hpnet4
which python

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
                            --nproc_per_node=1 \
                            --master_port 12345 \
                            --use_env main_hierINTR.py \
                            --finetune ~/projects/INTR/pretrained/detr-r50-e632da11.pth \
                            --dataset_path ~/data \
                            --dataset_name cub190_imgnet \
                            --num_queries 190 \
                            --phylo_config ~/projects/INTR/configs/cub190_phylogeny_disc4.yaml \
                            --output_dir 'output_HierINTR' \
                            --output_sub_dir '001_cub190-imgnet_disc=4' \
                            # --epochs 2

# DEBUGGING

# # Original cub
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 12345 --use_env main.py --finetune ~/projects/INTR/pretrained/detr-r50-e632da11.pth --dataset_path ~/data --dataset_name cub190_imgnet --num_queries 190
# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port 12345 --use_env main.py --finetune ~/projects/INTR/pretrained/detr-r50-e632da11.pth --dataset_path ~/data --dataset_name cub190_imgnet --num_queries 190

# # Original fish
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 12345 --use_env main.py --finetune ~/projects/INTR/pretrained/detr-r50-e632da11.pth --dataset_path ~/data --dataset_name fish38 --num_queries 38 --output_sub_dir checking

# # HierINTR cub190_imgnet
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 12345 --use_env main_hierINTR.py --finetune ~/projects/INTR/pretrained/detr-r50-e632da11.pth --dataset_path ~/data --dataset_name cub190_imgnet --num_queries 190 --phylo_config ~/projects/INTR/configs/cub190_phylogeny_disc4.yaml

# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port 12345 --use_env main_hierINTR.py --finetune ~/projects/INTR/pretrained/detr-r50-e632da11.pth --dataset_path ~/data --dataset_name cub190_imgnet --num_queries 190 --phylo_config ~/projects/INTR/configs/cub190_phylogeny_disc4.yaml
