# Scalable Event-by-event Processing of Neuromorphic Sensory Signals With Deep State-Space Models
![Figure 1](docs/figures/figure1.png)
This is the official implementation of our paper [Scalable Event-by-event Processing of Neuromorphic Sensory Signals With Deep State-Space Models
](https://arxiv.org/abs/2404.18508).
A core motivation for this work was the irregular time-series example presented in the paper [Simplified State Space Layers for Sequence Modeling
](https://arxiv.org/abs/2208.04933). 
Therefore, this repository started as a fork of the [S5 repository](https://github.com/lindermanlab/S5).
We treat quite a general machine learning problem:
Modeling sequences that are irregularly sampled by a possibly large number of asynchronous sensors.
This problem is particularly present in the field of neuromorphic computing, where event-based sensors emit up to millions events per second from asynchronous channels.

We show how linear state-space models such as S5 can be tuned to effectively model asynchronous event-based sequences.
Our contributions are
- a **discretization method** derived from dirac delta pulses
- **time-invariant input normalization** constant to effectively learn on long event-streams
- formulating neuromorphic event-streams as a language modeling problem with **asynchronous tokens**
- Effectively model event-based vision **without frames and without CNNs** 

## Installation

## Reproducing experiments
We use the [hydra](https://hydra.cc/docs/intro/) package to manage configurations.
If you are not familiar with hydra, we recommend to read the [documentation](https://hydra.cc/docs/intro/).

### Run benchmark tasks
The basic command to run an experiment is
```bash
python run_training.py
```
This will default to running the Spiking Heidelberg Digits (SHD) dataset.
All benchmark tasks are defined by the configurations in `configs/tasks/`, and can be run by specifying the `task` argument.
E.g. to run the Spiking Speech Commands (SSC) task
```bash
python run_training.py task=spiking-speech-commands
```
or to run the DVS128 Gestures task
```bash
python run_training.py task=dvs-gesture
```

### Specify HPC system and logging
Many researchers operate on different HPC systems and perhaps log their experiments to multiple platforms.
Therefore, the user can specify configurations for 
- different systems (directories for reading data and saving outputs)
- logging methods (e.g. whether to log locally or to [wandb](https://wandb.ai/))

By default, the `configs/system/local.yaml` and `configs/logging/local.yaml` configurations are used, respectively.
We suggest to create new configs for the HPC systems and wandb projects you are using.

For example, to run the model on SSC with your custom wandb logging config and your custom HPC specification do
```bash
python run_training.py task=spiking-speech-commands logging=wandb system=hpc
```
where `configs/logging/wandb.yaml` should look like
```yaml
log_dir: ${output_dir}
interval: 1000
wandb: False
summary_metric: "Performance/Validation accuracy"
project: wandb_project_name
entity: wandb_entity_name
```
and `configs/system/hpc.yaml` should specify data and output directories
```yaml
# @package _global_

data_dir: my/fast/storage/location/data
output_dir: my/job/output/location/${task.name}/${oc.env:SLURM_JOB_ID}/${now:%Y-%m-%d-%H-%M-%S}
```
The string `${task.name}/${oc.env:SLURM_JOB_ID}/${now:%Y-%m-%d-%H-%M-%S}` will create subdirectories named by task, slurm job ID, and date,
which we found useful in practice.
This specification of the `output_dir` is not required though.

# Help and support
We are eager to help you with any questions or issues you might have. 
Please use the GitHub issue tracker for questions and issues.

## Citation
Please use the following when citing our work:
```
@inproceedings{
smith2023simplified,
title={Simplified State Space Layers for Sequence Modeling},
author={Jimmy T.H. Smith and Andrew Warrington and Scott Linderman},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=Ai8Hw3AXqks}
}
```

Please reach out if you have any questions.

-- The S5 authors.
