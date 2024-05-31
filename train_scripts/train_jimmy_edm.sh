#!/bin/bash
#SBATCH --account=juno --partition=juno --qos=normal
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=30G
#SBATCH --gres=gpu:a5000:1

#SBATCH --nodelist=juno2

#SBATCH --job-name="jimmy_edm_v5large"
#SBATCH --output=../outputs/slurm_logs/ap_%j.out

# only use the following if you want email notification
# SBATCH --mail-user=blazer2208@outlook.com
# SBATCH --mail-type=ALL

# list out some useful information (optional)

#wait for 30 seconds
cd ..

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR



# sample process (list hostnames of the nodes you've requested)
NPROCS=`srun --nodes=${SLURM_NNODES} bash -c 'hostname' |wc -l`
echo NPROCS=$NPROCS

source /juno/u/aadityap/.bashrc

conda activate robodiff-retry
# export CPATH=$CONDA_PREFIX/include

# echo $LD_LIBRARY_PATH

# python test_muj.py
export HYDRA_FULL_ERROR=1

dir="outputs/edm/jimmy_v5large"

if [ -d "$dir" ]; then
    rm -r "$dir"
fi

mkdir -p "$dir"

CUDA_VISIBLE_DEVICES=0 python train.py --config-dir=local_cfgs/ --config-name=edm_jimmy.yaml \
    training.seed=42 training.device=cuda:0 logging.name=edm_jimmy_v5large \
    training.output_dir="$dir" > outputs/slurm_logs/edm_jimmy_v5large.out 2>&1 


# done
echo "Done"