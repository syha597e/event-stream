import torch
from pathlib import Path
from typing import Callable, Optional, TypeVar, Dict, Tuple, List, Union
import tonic
from functools import partial
import numpy as np
from s5.transform import CropEvents, Identity

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
			prototypical_sequence_length: int,
			in_dim: int,
			num_patches: int,
			train_size: int
):
		self.train_loader = train_loader
		self.val_loader = val_loader
		self.test_loader = test_loader
		self.aux_loaders = aux_loaders
		self.n_classes = n_classes
		self.prototypical_sequence_length = prototypical_sequence_length
		self.num_patches = num_patches
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


def event_stream_collate_fn(batch, resolution, pad_unit, patch_ids=None):
	# x are inputs, y are targets, z are aux data
	x, y, *z = zip(*batch)
	assert len(z) == 0

	# set tonic specific items
	timesteps = [np.diff(e['t']) for e in x]

	# NOTE: since timesteps are deltas, their length is L - 1, and we have to remove the last token in the following

	# process tokens for single input dim (e.g. audio)
	if len(resolution) == 1:
		tokens = [e['x'][:-1].astype(np.int32) for e in x]
	elif len(resolution) == 2:
		tokens = [(e['x'][:-1] * e['y'][:-1] + np.prod(resolution) * e['p'][:-1].astype(np.int32)).astype(np.int32) for e in x]

		if patch_ids is not None:
			# collect tokens and timesteps for each patch
			patch_tokens = [[tok[patch_ids[tok % np.prod(resolution)] == p] for p in np.unique(patch_ids)] for tok in tokens]
			patch_timesteps = [[timestep[patch_ids[tok % np.prod(resolution)] == p] for p in np.unique(patch_ids)] for tok, timestep in zip(tokens, timesteps)]
			max_len = np.max(np.array([[len(p) for p in tok] for tok in patch_tokens]))

			# pad tokens and timesteps
			tokens = [np.stack([np.pad(p, (0, max_len - len(p)), mode='constant', constant_values=-1) for p in tok]) for tok in patch_tokens]
			timesteps = [np.stack([np.pad(p, (0, max_len - len(p)), mode='constant', constant_values=0) for p in timestep]) for timestep in patch_timesteps]
	else:
		raise ValueError('resolution must contain 1 or 2 elements')

	# get padding lengths
	lengths = np.array([e.shape[-1] for e in timesteps], dtype=np.int32)
	pad_length = (lengths.max() // pad_unit) * pad_unit + pad_unit

	# need only the full length for masking in classification model
	lengths = np.array([e.size for e in timesteps], dtype=np.int32)

	if patch_ids is None:
		# pad tokens with -1, which results in a zero vector with embedding look-ups
		tokens = np.stack([np.pad(e, (0, pad_length - len(e)), mode='constant', constant_values=-1) for e in tokens])
		timesteps = np.stack([np.pad(e, (0, pad_length - len(e)), mode='constant', constant_values=0) for e in timesteps])
	else:
		# pad tokens with -1, which results in a zero vector with embedding look-ups
		tokens = np.stack([np.pad(e, ((0, 0), (0, pad_length - e.shape[1])), mode='constant', constant_values=-1) for e in tokens])
		timesteps = np.stack([np.pad(e, ((0, 0), (0, pad_length - e.shape[1])), mode='constant', constant_values=0) for e in timesteps])

	# timesteps are in micro seconds... transform to milliseconds
	timesteps = timesteps / 1000

	y = np.array(y)

	return tokens, y, timesteps, lengths


def event_stream_dataloader(train_data, val_data, test_data, bsz, collate_fn, rng, shuffle_training=True):
	def dataloader(dset, shuffle):
		return torch.utils.data.DataLoader(
			dset,
			batch_size=bsz,
			drop_last=True,
			collate_fn=collate_fn,
			shuffle=shuffle,
			generator=rng,
			num_workers=6
		)
	train_loader = dataloader(train_data, shuffle=shuffle_training)
	val_loader = dataloader(val_data, shuffle=False)
	test_loader = dataloader(test_data, shuffle=False)
	return train_loader, val_loader, test_loader


def create_events_shd_classification_dataset(
		cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT,
		bsz: int = 50,
		seed: int = 42,
		time_jitter: float = 100,
		noise: int = 100,
		drop_event: float = 0.1,
		time_skew: float = 1.1,
		validate_on_test: bool = False,
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

	transforms = tonic.transforms.Compose([
		tonic.transforms.DropEvent(p=drop_event),
		tonic.transforms.TimeSkew(coefficient=(1 / time_skew, time_skew), offset=0),
		tonic.transforms.TimeJitter(std=time_jitter, clip_negative=False, sort_timestamps=True),
		tonic.transforms.UniformNoise(sensor_size=(128, 128, 2), n=(0, noise))
	])

	train_data = tonic.datasets.SHD(save_to=cache_dir, train=True, transform=transforms)

	if validate_on_test:
		print("WARNING: Using test set for validation")
		val_data = tonic.datasets.SHD(save_to=cache_dir, train=False)
	else:
		val_length = int(0.1 * len(train_data))
		train_data, val_data = torch.utils.data.random_split(
			train_data,
			lengths=[len(train_data) - val_length, val_length],
			generator=rng
		)
	test_data = tonic.datasets.SHD(save_to=cache_dir, train=False)

	# choose a minimal padding unit of 8192, which covers half the sequences.
	# The rest will be padded to multiples of this value
	pad_unit = 8192
	train_loader, val_loader, test_loader = event_stream_dataloader(
		train_data, val_data, test_data,
		collate_fn=partial(event_stream_collate_fn, resolution=(700,), pad_unit=pad_unit),
		bsz=bsz, rng=rng, shuffle_training=True
	)

	return Data(
		train_loader, val_loader, test_loader, aux_loaders={},
		n_classes=20, in_dim=700, prototypical_sequence_length=pad_unit, train_size=len(train_data),
		num_patches=0
	)


def create_events_ssc_classification_dataset(
		cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT,
		bsz: int = 50,
		seed: int = 42,
		time_jitter: float = 100,
		noise: int = 100,
		drop_event: float = 0.1,
		time_skew: float = 1.1,
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

	transforms = tonic.transforms.Compose([
		tonic.transforms.DropEvent(p=drop_event),
		tonic.transforms.TimeSkew(coefficient=(1 / time_skew, time_skew), offset=0),
		tonic.transforms.TimeJitter(std=time_jitter, clip_negative=False, sort_timestamps=True),
		tonic.transforms.UniformNoise(sensor_size=(128, 128, 2), n=(0, noise))
	])

	train_data = tonic.datasets.SSC(save_to=cache_dir, split='train', transform=transforms)
	val_data = tonic.datasets.SSC(save_to=cache_dir, split='valid')
	test_data = tonic.datasets.SSC(save_to=cache_dir, split='test')

	# choose a minimal padding unit of 8192, which covers half the sequences.
	# The rest will be padded to multiples of this value
	pad_unit = 8192
	train_loader, val_loader, test_loader = event_stream_dataloader(
		train_data, val_data, test_data,
		collate_fn=partial(event_stream_collate_fn, resolution=(700,), pad_unit=pad_unit),
		bsz=bsz, rng=rng, shuffle_training=True
	)

	return Data(
		train_loader, val_loader, test_loader, aux_loaders={},
		n_classes=35, in_dim=700, prototypical_sequence_length=pad_unit, train_size=len(train_data),
		num_patches=0
	)


def get_patch_ids(sensor_size, stages):
	# look up table for patched SSM
	num_patches = (2 ** stages) ** 2
	ids_per_patch = sensor_size[0] * sensor_size[1] // num_patches
	N = sensor_size[0] // (2 ** stages)
	base = np.arange(N * N).reshape(N, N)
	for _ in range(stages):
		kernel = np.arange(4).repeat(N * N).reshape(4, N, N)
		kernel = N * N * kernel + base[None, ...]
		kernel = kernel.reshape(2, 2, N, N)
		base = np.transpose(kernel, (0, 2, 1, 3)).reshape(2 * N, 2 * N)
		N = 2 * N
	return base.flatten() // ids_per_patch, num_patches


def create_events_dvs_gesture_classification_dataset(
		cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT,
		bsz: int = 50,
		seed: int = 42,
		crop_events: int = None,
		time_jitter: float = 100,
		noise: int = 100,
		drop_event: float = 0.1,
		time_skew: float = 1.1,
		downsampling: int = 1,
		stages: int = 0,
		**kwargs
) -> Data:
	"""
	creates a view of the DVS Gesture dataset

	:param cache_dir:		(str):		where to store the dataset
	:param bsz:				(int):		Batch size.
	:param seed:			(int)		Seed for shuffling data.
	"""
	print("[*] Generating DVS Gesture Classification Dataset")

	assert time_skew >= 1, "time_skew must be greater than 1"

	if seed is not None:
		rng = torch.Generator()
		rng.manual_seed(seed)
	else:
		rng = None

	orig_sensor_size = (128, 128, 2)
	new_sensor_size = (128 // downsampling, 128 // downsampling, 2)
	train_transforms = [
		tonic.transforms.DropEvent(p=drop_event),
		tonic.transforms.TimeJitter(std=time_jitter, clip_negative=False, sort_timestamps=True),
		tonic.transforms.SpatialJitter(sensor_size=orig_sensor_size, var_x=1, var_y=1, clip_outliers=True),
		tonic.transforms.Downsample(spatial_factor=downsampling) if downsampling > 1 else Identity(),
		tonic.transforms.DropEventByArea(sensor_size=new_sensor_size, area_ratio=(0.05, 0.2)),
		tonic.transforms.UniformNoise(sensor_size=new_sensor_size, n=(0, noise))
	]
	if crop_events is not None:
		train_transforms.append(CropEvents(crop_events))
	if time_skew > 0:
		train_transforms.append(tonic.transforms.TimeSkew(coefficient=(1 / time_skew, time_skew), offset=0))

	train_transforms = tonic.transforms.Compose(train_transforms)

	train_data = tonic.datasets.DVSGesture(save_to=cache_dir, train=True, transform=train_transforms)
	val_length = int(0.1 * len(train_data))
	train_data, val_data = torch.utils.data.random_split(
		train_data,
		lengths=[len(train_data) - val_length, val_length],
		generator=rng
	)

	test_transforms = tonic.transforms.Compose([
		tonic.transforms.Downsample(spatial_factor=downsampling) if downsampling > 1 else Identity(),
	])
	test_data = tonic.datasets.DVSGesture(save_to=cache_dir, train=False, transform=test_transforms)

	if stages > 0:
		patch_ids, num_patches = get_patch_ids(sensor_size=new_sensor_size, stages=stages)
	else:
		patch_ids, num_patches = None, 0

	# TODO: 1. num workers > 0 2. pass num patches as argument 3. align num patches and num levels
	# choose a minimal padding unit of 8192, which covers half the sequences.
	# The rest will be padded to multiples of this value
	pad_unit = 2 ** 19 if stages == 0 else 2 ** 16
	train_loader, val_loader, test_loader = event_stream_dataloader(
		train_data, val_data, test_data,
		collate_fn=partial(event_stream_collate_fn, resolution=new_sensor_size[:2], pad_unit=pad_unit, patch_ids=patch_ids),
		bsz=bsz, rng=rng, shuffle_training=True
	)

	return Data(
		train_loader, val_loader, test_loader, aux_loaders={},
		n_classes=11, in_dim=128 * 128 * 2, prototypical_sequence_length=pad_unit if crop_events is None else crop_events,
		train_size=len(train_data), num_patches=num_patches
	)


Datasets = {
	"shd-classification": create_events_shd_classification_dataset,
	"ssc-classification": create_events_ssc_classification_dataset,
	"dvs-gesture-classification": create_events_dvs_gesture_classification_dataset,
}
