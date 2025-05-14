#!/bin/bash
#SBATCH -p allgriz
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=4
#SBATCH --time=00:35:00
#SBATCH --exclusive
#SBATCH -k
#SBATCH --output=logs/oumi.out
#SBATCH --job-name=oumi

echo Number of nodes: ${SLURM_NNODES}
ulimit -s unlimited

export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=1

cd /lus/scratch/arigazzi
source venvs/oumi/bin/activate

cd oumi

# Set MASTER_ADDR to the first allocated node and MASTER_PORT to 8000
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: ${head_node_ip}
export LOGLEVEL=INFO


echo ======== start ==============
date
echo ======== start ==============
srun --nodes=${SLURM_NNODES} --ntasks=${SLURM_NNODES} \
        --ntasks-per-node=1 \
        oumi distributed torchrun \
        --nnodes ${SLURM_NNODES} \
        --rdzv_id ${RANDOM} \
        --rdzv_backend c10d \
        --rdzv_endpoint ${head_node_ip}:29500 \
        -m oumi train  -c configs/recipes/smollm/sft/135m/quickstart_train.yaml
echo ======== end ==============
date
echo ======== end ==============