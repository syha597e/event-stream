import torch
from pathlib import Path
import os
from typing import Callable, Optional, TypeVar, Dict, Tuple, List, Union
import tonic
from functools import partial
from torch.nn.functional import pad
import numpy as np

DEFAULT_CACHE_DIR_ROOT = Path('./cache_dir/')

DataLoader = TypeVar('DataLoader')
InputType = [str, Optional[int], Optional[int]]


class Data:
	def __init__(
			self,
			train_loader: DataLoader,
			val_loader: DataLoader,
			test_loader: DataLoader,
			aux_loaders: Dict,
			n_classes: int,
			train_pad_length: int,
			test_pad_length: int,
			in_dim: int,
			train_size: int
):
		self.train_loader = train_loader
		self.val_loader = val_loader
		self.test_loader = test_loader
		self.aux_loaders = aux_loaders
		self.n_classes = n_classes
		self.train_pad_length = train_pad_length
		self.test_pad_length = test_pad_length
		self.in_dim = in_dim
		self.train_size = train_size


# Custom loading functions must therefore have the template.
dataset_fn = Callable[[str, Optional[int], Optional[int]], Data]


# Example interface for making a loader.
def custom_loader(cache_dir: str,
				  bsz: int = 50,
				  seed: int = 42) -> Data:
	...


def make_data_loader(dset,
					 dobj,
					 seed: int,
					 batch_size: int=128,
					 shuffle: bool=True,
					 drop_last: bool=True,
					 collate_fn: callable=None):
	"""

	:param dset: 			(PT dset):		PyTorch dataset object.
	:param dobj (=None): 	(AG data): 		Dataset object, as returned by A.G.s dataloader.
	:param seed: 			(int):			Int for seeding shuffle.
	:param batch_size: 		(int):			Batch size for batches.
	:param shuffle:         (bool):			Shuffle the data loader?
	:param drop_last: 		(bool):			Drop ragged final batch (particularly for training).
	:return:
	"""

	# Create a generator for seeding random number draws.
	if seed is not None:
		rng = torch.Generator()
		rng.manual_seed(seed)
	else:
		rng = None

	if dobj is not None:
		assert collate_fn is None
		collate_fn = dobj._collate_fn

	# Generate the dataloaders.
	return torch.utils.data.DataLoader(dataset=dset, collate_fn=collate_fn, batch_size=batch_size, shuffle=shuffle,
									   drop_last=drop_last, generator=rng)


def event_stream_collate_fn(batch, resolution, max_time=None):
	# x are inputs, y are targets, z are aux data
	x, y, *z = zip(*batch)
	assert len(z) == 0

	# set tonic specific items
	timesteps = [torch.tensor(e['t'].copy()) for e in x]

	# process tokens for single input dim (e.g. audio)
	if len(resolution) == 1:
		tokens = [torch.tensor(e['x']).int() for e in x]
	elif len(resolution) == 2:
		tokens = [torch.tensor(e['x'] * e['y'] + np.prod(resolution) * e['p'].astype(np.int32)).int() for e in x]
	else:
		raise ValueError('resolution must contain 1 or 2 elements')

	# drop of events later than max_time
	if max_time is not None:
		mask = [e <= max_time for e in timesteps]
		timesteps = [e[m] for e, m in zip(timesteps, mask)]
		tokens = [e[m] for e, m in zip(tokens, mask)]

	# pad time steps with final time-step -> this is a bit of a hack to make the integration time steps 0
	lengths = torch.tensor([len(e) for e in timesteps], dtype=torch.long)
	max_length = lengths.max().item()
	timesteps = torch.stack([pad(e, (0, max_length - len(e)), 'constant', e[-1]) for e in timesteps])

	# timesteps are in micro seconds... transform to milliseconds
	timesteps = timesteps / 1000

	# Hack: apply random time jitter to avoid the same timestep occouring multiple times
	timesteps = timesteps + torch.rand_like(timesteps) * 0.1
	ind = torch.argsort(timesteps, dim=1)
	timesteps = torch.gather(timesteps, dim=1, index=ind)

	# pad tokens with -1, which results in a zero vector with jax.nn.one_hot
	tokens = torch.stack([pad(e, (0, max_length - len(e)), 'constant', -1) for e in tokens])
	tokens = torch.gather(tokens, dim=1, index=ind)

	y = torch.tensor(y).int()
	return tokens, y, {'timesteps': timesteps, 'lengths': lengths}


def event_stream_dataloader(train_data, val_data, test_data, bsz, collate_fn, rng, shuffle_training=True, max_time=None):
	def dataloader(dset, shuffle, time_limit=None):
		return torch.utils.data.DataLoader(
			dset,
			batch_size=bsz,
			drop_last=True,
			collate_fn=partial(collate_fn, max_time=time_limit),
			shuffle=shuffle,
			generator=rng,
			num_workers=6
		)
	train_loader = dataloader(train_data, shuffle=shuffle_training, time_limit=max_time)
	val_loader = dataloader(val_data, shuffle=False)
	test_loader = dataloader(test_data, shuffle=False)
	return train_loader, val_loader, test_loader


def create_events_shd_classification_dataset(
		cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT,
		bsz: int = 50,
		seed: int = 42,
		**kwargs
) -> Data:
	"""
	creates a view of the spiking heidelberg digits dataset

	:param cache_dir:		(str):		where to store the dataset
	:param bsz:				(int):		Batch size.
	:param seed:			(int)		Seed for shuffling data.
	"""
	print("[*] Generating Spiking Heidelberg Digits Classification Dataset")

	if seed is not None:
		rng = torch.Generator()
		rng.manual_seed(seed)
	else:
		rng = None

	train_data = tonic.datasets.SHD(save_to=cache_dir, train=True)
	val_length = int(0.1 * len(train_data))
	train_data, val_data = torch.utils.data.random_split(
		train_data,
		lengths=[len(train_data) - val_length, val_length],
		generator=rng
	)
	test_data = tonic.datasets.SHD(save_to=cache_dir, train=False)

	train_loader, val_loader, test_loader = event_stream_dataloader(
		train_data, val_data, test_data,
		collate_fn=partial(event_stream_collate_fn, resolution=(700,)),
		bsz=bsz, rng=rng, shuffle_training=True
	)

	return Data(
		train_loader, val_loader, test_loader, aux_loaders={},
		n_classes=20, in_dim=700, train_pad_length=16384, test_pad_length=16384, train_size=len(train_data)
	)

def create_events_ssc_classification_dataset(
		cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT,
		bsz: int = 50,
		seed: int = 42,
		**kwargs
) -> Data:
	"""
	creates a view of the spiking speech commands dataset

	:param cache_dir:		(str):		where to store the dataset
	:param bsz:				(int):		Batch size.
	:param seed:			(int)		Seed for shuffling data.
	"""
	print("[*] Generating Spiking Speech Commands Classification Dataset")

	if seed is not None:
		rng = torch.Generator()
		rng.manual_seed(seed)
	else:
		rng = None

	train_data = tonic.datasets.SSC(save_to=cache_dir, split='train')
	val_data = tonic.datasets.SSC(save_to=cache_dir, split='valid')
	test_data = tonic.datasets.SSC(save_to=cache_dir, split='test')

	train_loader, val_loader, test_loader = event_stream_dataloader(
		train_data, val_data, test_data,
		collate_fn=partial(event_stream_collate_fn, resolution=(700,)),
		bsz=bsz, rng=rng, shuffle_training=True
	)

	return Data(
		train_loader, val_loader, test_loader, aux_loaders={},
		n_classes=35, in_dim=700, train_pad_length=20480, test_pad_length=20480, train_size=len(train_data)
	)


def create_events_dvs_gesture_classification_dataset(
		cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT,
		bsz: int = 50,
		seed: int = 42,
		pad: int = 262144,
		max_time: int = None
) -> Data:
	"""
	creates a view of the DVS Gesture dataset

	:param cache_dir:		(str):		where to store the dataset
	:param bsz:				(int):		Batch size.
	:param seed:			(int)		Seed for shuffling data.
	"""
	print("[*] Generating DVS Gesture Classification Dataset")

	assert isinstance(pad, int), "default_num_events must be an integer"
	assert pad > 0, "default_num_events must be a positive integer"

	if seed is not None:
		rng = torch.Generator()
		rng.manual_seed(seed)
	else:
		rng = None

	train_data = tonic.datasets.DVSGesture(save_to=cache_dir, train=True)
	val_length = int(0.1 * len(train_data))
	train_data, val_data = torch.utils.data.random_split(
		train_data,
		lengths=[len(train_data) - val_length, val_length],
		generator=rng
	)
	test_data = tonic.datasets.DVSGesture(save_to=cache_dir, train=False)

	train_loader, val_loader, test_loader = event_stream_dataloader(
		train_data, val_data, test_data,
		collate_fn=partial(event_stream_collate_fn, resolution=(128, 128)),
		bsz=bsz, rng=rng, shuffle_training=False, max_time=max_time
	)

	return Data(
		train_loader, val_loader, test_loader, aux_loaders={},
		n_classes=11, in_dim=128 * 128 * 2, train_pad_length=pad, test_pad_length=1595392, train_size=len(train_data)
	)

def create_speechcommands35_classification_dataset(
		cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT,
		bsz: int = 50,
		seed: int = 42,
		**kwargs
) -> Data:
	"""
	AG inexplicably moved away from using a cache dir...  Grumble.
	The `cache_dir` will effectively be ./raw_datasets/speech_commands/0.0.2 .

	See abstract template.
	"""
	print("[*] Generating SpeechCommands35 Classification Dataset")
	from s5.dataloaders.basic import SpeechCommands
	name = 'sc'

	dir_name = f'./raw_datasets/speech_commands/0.0.2/'
	os.makedirs(dir_name, exist_ok=True)

	kwargs = {
		'all_classes': True,
		'sr': 1  # Set the subsampling rate.
	}
	dataset_obj = SpeechCommands(name, data_dir=dir_name, **kwargs)
	dataset_obj.setup()
	trn_loader = make_data_loader(dataset_obj.dataset_train, dataset_obj, seed=seed, batch_size=bsz)
	val_loader = make_data_loader(dataset_obj.dataset_val, dataset_obj, seed=seed, batch_size=bsz, drop_last=False, shuffle=False)
	tst_loader = make_data_loader(dataset_obj.dataset_test, dataset_obj, seed=seed, batch_size=bsz, drop_last=False, shuffle=False)

	N_CLASSES = dataset_obj.d_output
	SEQ_LENGTH = dataset_obj.dataset_train.tensors[0].shape[1]
	IN_DIM = 1
	TRAIN_SIZE = dataset_obj.dataset_train.tensors[0].shape[0]

	# Also make the half resolution dataloader.
	kwargs['sr'] = 2
	dataset_obj = SpeechCommands(name, data_dir=dir_name, **kwargs)
	dataset_obj.setup()
	val_loader_2 = make_data_loader(dataset_obj.dataset_val, dataset_obj, seed=seed, batch_size=bsz, drop_last=False, shuffle=False)
	tst_loader_2 = make_data_loader(dataset_obj.dataset_test, dataset_obj, seed=seed, batch_size=bsz, drop_last=False, shuffle=False)

	aux_loaders = {
		'valloader2': val_loader_2,
		'testloader2': tst_loader_2,
	}

	return Data(
		trn_loader, val_loader, tst_loader, aux_loaders=aux_loaders,
		n_classes=N_CLASSES, in_dim=IN_DIM, train_pad_length=SEQ_LENGTH, test_pad_length=SEQ_LENGTH, train_size=TRAIN_SIZE
	)


Datasets = {
	# Speech.
	"speech35-classification": create_speechcommands35_classification_dataset,

	# Events
	"shd-classification": create_events_shd_classification_dataset,
	"ssc-classification": create_events_ssc_classification_dataset,
	"dvs-gesture-classification": create_events_dvs_gesture_classification_dataset,
}
