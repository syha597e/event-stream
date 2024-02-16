python run_train.py --C_init=lecun_normal --batchnorm=True \
  --bsz=16 --n_layers=6 --d_model=96 --ssm_size_base=128 --blocks=16 \
  --dataset=shd-classification --epochs=25 --warmup_end=3 --jax_seed=16416 --lr_factor=4 \
  --opt_config=noBCdecay --p_dropout=0.1 --ssm_lr_base=0.002 --weight_decay=0.04 --dir_name=$DATA_DIR \
  --USE_WANDB=True --wandb_entity=evnn-neurips --wandb_project=event-stream-classification \
  --activation_fn=half_glu2 --dt_min=1e-4