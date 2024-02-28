import argparse
from s5.utils.util import str2bool
from s5.train import train
from s5.dataloading import Datasets

if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	parser.add_argument("--USE_WANDB", type=str2bool, default=False,
						help="log with wandb?")
	parser.add_argument("--wandb_project", type=str, default=None,
						help="wandb project name")
	parser.add_argument("--wandb_entity", type=str, default=None,
						help="wandb entity name, e.g. username")
	parser.add_argument("--dir_name", type=str, default='./cache_dir',
						help="name of directory where data is cached")
	parser.add_argument("--dataset", type=str, choices=Datasets.keys(),
						default='mnist-classification',
						help="dataset name")

	# Model Parameters
	parser.add_argument("--n_layers", type=int, default=6,
						help="Number of layers in the network")
	parser.add_argument("--d_model", type=int, default=128,
						help="Number of features, i.e. H, "
							 "dimension of layer inputs/outputs")
	parser.add_argument("--ssm_size_base", type=int, default=256,
						help="SSM Latent size, i.e. P")
	parser.add_argument("--block_size", type=int, default=16,
						help="How many blocks, J, to initialize with")
	parser.add_argument("--C_init", type=str, default="trunc_standard_normal",
						choices=["trunc_standard_normal", "lecun_normal", "complex_normal"],
						help="Options for initialization of C: \\"
							 "trunc_standard_normal: sample from trunc. std. normal then multiply by V \\ " \
							 "lecun_normal sample from lecun normal, then multiply by V\\ " \
							 "complex_normal: sample directly from complex standard normal")
	parser.add_argument("--discretization", type=str, default="zoh", choices=["zoh", "bilinear", "dirac", "state_zoh"])
	parser.add_argument("--first_layer_discretization", type=str, default="dirac", choices=["zoh", "bilinear", "dirac", "state_zoh"],
						help="discretization for first layer")
	parser.add_argument("--mode", type=str, default="pool", choices=["pool", "last", "timepool"],
						help="options: (for classification tasks) \\" \
							 " pool: mean pooling \\" \
							 "last: take last element")
	parser.add_argument("--activation_fn", default="half_glu2", type=str,
						choices=["full_glu", "half_glu1", "half_glu2", "gelu"])
	parser.add_argument("--conj_sym", type=str2bool, default=True,
						help="whether to enforce conjugate symmetry")
	parser.add_argument("--clip_eigs", type=str2bool, default=False,
						help="whether to enforce the left-half plane condition")
	parser.add_argument("--dt_min", type=float, default=0.001,
						help="min value to sample initial timescale params from")
	parser.add_argument("--dt_max", type=float, default=0.1,
						help="max value to sample initial timescale params from")
	parser.add_argument("--pooling_stride", type=int, default=1,
						help="stride for event pooling")
	parser.add_argument("--pooling_every_n_layers", type=int, default=99999,
						help="Apply stride every n layers")
	parser.add_argument("--pooling_mode", type=str, default="last",
						choices=["last", "avgpool", "timepool"],
						help="Select mode to pool events")
	parser.add_argument("--state_expansion_factor", type=int, default=1,
						help="Expands the state in pooling layers by this factor")

	# Data options
	parser.add_argument("--max_events_per_sample", type=int, default=None,
						help="Default number of events padded to in each sequence")
	parser.add_argument("--time_jitter", type=float, default=100,
						help="time jitter in micro seconds - important to reduce the overlap of events")
	parser.add_argument("--noise", type=int, default=0,
						help="Number of randomly added events per sample")
	parser.add_argument("--drop_event", type=float, default=0.0,
						help="probability of dropping an event")
	parser.add_argument("--time_skew", type=float, default=1.01,
						help="time skew factor (must be larger than 1)")
	# SHD only
	parser.add_argument("--validate_on_test", action="store_true",
						help="whether to validate on SHD test set or create a custom validation set by random split")
	# DVS only
	parser.add_argument("--refractory_period", type=int, default=0,
						help="refractory period in micro seconds")
	parser.add_argument("--downsampling", type=int, default=1,
						help="resolution of the DVS sensor")

	# Optimization Parameters
	parser.add_argument("--prenorm", type=str2bool, default=True,
						help="True: use prenorm, False: use postnorm")
	parser.add_argument("--batchnorm", type=str2bool, default=True,
						help="True: use batchnorm, False: use layernorm")
	parser.add_argument("--bn_momentum", type=float, default=0.95,
						help="batchnorm momentum")
	parser.add_argument("--bsz", type=int, default=64,
						help="batch size")
	parser.add_argument("--epochs", type=int, default=100,
						help="max number of epochs")
	parser.add_argument("--early_stop_patience", type=int, default=1000,
						help="number of epochs to continue training when val loss plateaus")
	parser.add_argument("--ssm_lr_base", type=float, default=1e-3,
						help="initial ssm learning rate")
	parser.add_argument("--lr_factor", type=float, default=1,
						help="global learning rate = lr_factor*ssm_lr_base")
	parser.add_argument("--dt_global", type=str2bool, default=False,
						help="Treat timescale parameter as global parameter or SSM parameter")
	parser.add_argument("--lr_min", type=float, default=0,
						help="minimum learning rate")
	parser.add_argument("--cosine_anneal", type=str2bool, default=True,
						help="whether to use cosine annealing schedule")
	parser.add_argument("--warmup_end", type=int, default=1,
						help="epoch to end linear warmup")
	parser.add_argument("--lr_patience", type=int, default=1000000,
						help="patience before decaying learning rate for lr_decay_on_val_plateau")
	parser.add_argument("--reduce_factor", type=float, default=1.0,
						help="factor to decay learning rate for lr_decay_on_val_plateau")
	parser.add_argument("--p_dropout", type=float, default=0.0,
						help="probability of dropout")
	parser.add_argument("--weight_decay", type=float, default=0.05,
						help="weight decay value")
	parser.add_argument("--opt_config", type=str, default="standard", choices=['standard',
																			   'BandCdecay',
																			   'BfastandCdecay',
																			   'noBCdecay'],
						help="Opt configurations: \\ " \
			   "standard:       no weight decay on B (ssm lr), weight decay on C (global lr) \\" \
	  	       "BandCdecay:     weight decay on B (ssm lr), weight decay on C (global lr) \\" \
	  	       "BfastandCdecay: weight decay on B (global lr), weight decay on C (global lr) \\" \
	  	       "noBCdecay:      no weight decay on B (ssm lr), no weight decay on C (ssm lr) \\")
	parser.add_argument("--jax_seed", type=int, default=1919,
						help="seed randomness")

	train(parser.parse_args())
