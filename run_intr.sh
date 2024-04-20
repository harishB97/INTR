# #!/bin/bash

# #SBATCH --account=mabrownlab
# #SBATCH --partition=dgx_normal_q
# #SBATCH --time=1-00:00:00 
# #SBATCH --gres=gpu:1
# #SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8
# #SBATCH -o ./SLURM/slurm-%j.out


# echo start load env and run python

# module reset
# module load Anaconda3/2020.11
# source activate hpnet1
# module reset
# source activate hpnet1
# which python

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 12345 --use_env main.py --finetune ~/projects/INTR/pretrained --dataset_path ~/data --dataset_name cub190 --num_queries 190
