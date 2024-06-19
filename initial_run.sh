#!/bin/bash

##Submit this script with: sbatch thefilename

#SBATCH --time=24:00:00       # walltime
#SBATCH --nodes=1             # number of nodes
#SBATCH --nodelist=i8002      # place your job onto specific nodes
#SBATCH --gres=gpu:1          # how many GPUs to request
#SBATCH --ntasks=1            # limit to one node
#SBATCH --cpus-per-task=8     # number of processor cores (4 threads per CPU on ML nodes)es

#SBATCH --mem=40GB            # memory per CPU core
#SBATCH -J "SHD1_pos job"           # job name
# -A Charge resources used by this job to specified account
##SBATCH -A p_scads_rnn
#SBATCH -A p_largedata

#SBATCH -o /data/horse/beegfs/ws/syha597e-my_workspace/git-data/position-embeddings/event-stream-modeling-s5/slurm/slurm-%j.out     # save output message
#SBATCH -e /data/horse/beegfs/ws/syha597e-my_workspace/git-data/position-embeddings/event-stream-modeling-s5/slurm/slurm-%j.err     # save error messages

#SBATCH --mail-type=end
#SBATCH --mail-user=syed_quosain.haider@mailbox.tu-dresden.de

#wandb login --relogin
# Set the WANDB_API_KEY environment variable
export WANDB_API_KEY="7cf138723240158f313416d8f2d049400da611a2"

# Optionally, set the wandb project name
export WANDB_PROJECT="ssm-event-initial"

# Load any necessary modules
module --force purge
#module load release/23.04
module load development/24.04 
module load GCC/11.3.0
module load Python/3.10.4  # Ensure you load the module that corresponds to your Python version
#module load OpenMPI/4.1.4
#module load PyTorch/1.12.1-CUDA-11.7.0
#module load TensorFlow/2.9.1-CUDA-11.7.0
# Activate virtual environment (if applicable)
source /data/horse/beegfs/ws/syha597e-my_workspace/python_virtualenv/test_env/bin/activate

# Run the PyTorch script
#srun $(which python) /data/horse/beegfs/ws/syha597e-my_workspace/git-data/public-code/event-stream-modeling-s5/run_training.py logging=wandb system=hpc
#python /home/syha597e/git-data/public-code/event-stream-modeling-s5/run_training.py
python run_training.py task=spiking-heidelberg-digits logging=wandb system=hpc

#task=dvs-gesture
#task=spiking-speech-commands
#task=spiking-heidelberg-digits
