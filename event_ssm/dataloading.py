import torch
from pathlib import Path
from typing import Callable, Optional, TypeVar, Dict, Tuple, List, Union
import tonic
from functools import partial
from torch.nn.functional import pad
import numpy as np
import jax
from event_ssm.transform import CropEvents, Identity

DEFAULT_CACHE_DIR_ROOT = Path('./cache_dir/')

DataLoader = TypeVar('DataLoader')
InputType = [str, Optional[int], Optional[int]]


class Data:
	def __init__(
			self,
			n_classes: int,
			num_embeddings: int,
			train_size: int
):
		self.n_classes = n_classes
		self.num_embeddings = num_embeddings
		self.train_size = train_size


def event_stream_collate_fn(batch, resolution, pad_unit):
	# x are inputs, y are targets, z are aux data
	x, y, *z = zip(*batch)
	assert len(z) == 0

	# integration time steps are the difference between two consequtive time stamps
	timesteps = [np.diff(e['t']) for e in x]

	# NOTE: since timesteps are deltas, their length is L - 1, and we have to remove the last token in the following

	# process tokens for single input dim (e.g. audio)
	if len(resolution) == 1:
		tokens = [e['x'][:-1].astype(np.int32) for e in x]
	elif len(resolution) == 2:
		tokens = [(e['x'][:-1] * e['y'][:-1] + np.prod(resolution) * e['p'][:-1].astype(np.int32)).astype(np.int32) for e in x]
	else:
		raise ValueError('resolution must contain 1 or 2 elements')

	# get padding lengths
	lengths = np.array([len(e) for e in timesteps], dtype=np.int32)
	pad_length = (lengths.max() // pad_unit) * pad_unit + pad_unit

	# pad tokens with -1, which results in a zero vector with embedding look-ups
	tokens = np.stack(
		[np.pad(e, (0, pad_length - len(e)), mode='constant', constant_values=-1) for e in tokens])
	timesteps = np.stack(
		[np.pad(e, (0, pad_length - len(e)), mode='constant', constant_values=0) for e in timesteps])

	# timesteps are in micro seconds... transform to milliseconds
	timesteps = timesteps / 1000

	return tokens, np.array(y), timesteps, lengths


def event_stream_dataloader(train_data, val_data, test_data, batch_size, eval_batch_size, collate_fn, rng, num_workers=0, shuffle_training=True):
	def dataloader(dset, bsz, shuffle, drop_last):
		return torch.utils.data.DataLoader(
			dset,
			batch_size=bsz,
			drop_last=drop_last,
			collate_fn=collate_fn,
			shuffle=shuffle,
			generator=rng,
			num_workers=num_workers
		)
	train_loader = dataloader(train_data, bsz=batch_size, shuffle=shuffle_training, drop_last=True)
	val_loader = dataloader(val_data, bsz=eval_batch_size, shuffle=False, drop_last=False)
	test_loader = dataloader(test_data, bsz=eval_batch_size, shuffle=False, drop_last=False)
	return train_loader, val_loader, test_loader


def create_events_shd_classification_dataset(
		cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT,
		batch_size: int = 32,
		eval_batch_size: int = 64,
		num_workers: int = 0,
		seed: int = 42,
		time_jitter: float = 100,
		noise: int = 100,
		drop_event: float = 0.1,
		time_skew: float = 1.1,
		pad_unit: int = 8192,
		validate_on_test: bool = False,
		**kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader, Data]:
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

	sensor_size = (700, 1, 1)

	transforms = tonic.transforms.Compose([
		tonic.transforms.DropEvent(p=drop_event),
		tonic.transforms.TimeSkew(coefficient=(1 / time_skew, time_skew), offset=0),
		tonic.transforms.TimeJitter(std=time_jitter, clip_negative=False, sort_timestamps=True),
		tonic.transforms.UniformNoise(sensor_size=sensor_size, n=(0, noise))
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

	train_loader, val_loader, test_loader = event_stream_dataloader(
		train_data, val_data, test_data,
		collate_fn=partial(event_stream_collate_fn, resolution=(700,), pad_unit=pad_unit),
		batch_size=batch_size, eval_batch_size=eval_batch_size,
		rng=rng, num_workers=num_workers, shuffle_training=True
	)
	data = Data(
		n_classes=20, num_embeddings=700, train_size=len(train_data)
	)
	return train_loader, val_loader, test_loader, data

def create_events_ssc_classification_dataset(
		cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT,
		batch_size: int = 32,
		eval_batch_size: int = 64,
		num_workers: int = 0,
		seed: int = 42,
		time_jitter: float = 100,
		noise: int = 100,
		drop_event: float = 0.1,
		time_skew: float = 1.1,
		pad_unit: int = 8192,
		**kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader, Data]:
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

	sensor_size = (700, 1, 1)

	transforms = tonic.transforms.Compose([
		tonic.transforms.DropEvent(p=drop_event),
		tonic.transforms.TimeSkew(coefficient=(1 / time_skew, time_skew), offset=0),
		tonic.transforms.TimeJitter(std=time_jitter, clip_negative=False, sort_timestamps=True),
		tonic.transforms.UniformNoise(sensor_size=sensor_size, n=(0, noise))
	])

	train_data = tonic.datasets.SSC(save_to=cache_dir, split='train', transform=transforms)
	val_data = tonic.datasets.SSC(save_to=cache_dir, split='valid')
	test_data = tonic.datasets.SSC(save_to=cache_dir, split='test')

	train_loader, val_loader, test_loader = event_stream_dataloader(
		train_data, val_data, test_data,
		collate_fn=partial(event_stream_collate_fn, resolution=(700,), pad_unit=pad_unit),
		batch_size=batch_size, eval_batch_size=eval_batch_size,
		rng=rng, num_workers=num_workers, shuffle_training=True
	)

	data = Data(
		n_classes=35, num_embeddings=700, train_size=len(train_data)
	)
	return train_loader, val_loader, test_loader, data



def create_events_dvs_gesture_classification_dataset(
		cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT,
		batch_size: int = 32,
		eval_batch_size: int = 64,
		num_workers: int = 0,
		seed: int = 42,
		crop_events: int = None,
		time_jitter: float = 100,
		noise: int = 100,
		drop_event: float = 0.1,
		time_skew: float = 1.1,
		downsampling: int = 1,
		pad_unit: int = 2 ** 19,
		**kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader, Data]:
	"""
	creates a view of the DVS Gesture dataset

	:param cache_dir:		(str):		where to store the dataset
	:param bsz:				(int):		Batch size.
	:param seed:			(int)		Seed for shuffling data.
	"""
	print("[*] Generating DVS Gesture Classification Dataset")

	assert isinstance(crop_events, int), "default_num_events must be an integer"
	assert crop_events > 0, "default_num_events must be a positive integer"
	assert time_skew > 1, "time_skew must be greater than 1"

	if seed is not None:
		rng = torch.Generator()
		rng.manual_seed(seed)
	else:
		rng = None

	orig_sensor_size = (128, 128, 2)
	new_sensor_size = (128 // downsampling, 128 // downsampling, 2)
	train_transforms = [
		tonic.transforms.DropEvent(p=drop_event),
		tonic.transforms.TimeSkew(coefficient=(1 / time_skew, time_skew), offset=0),
		tonic.transforms.TimeJitter(std=time_jitter, clip_negative=False, sort_timestamps=True),
		tonic.transforms.SpatialJitter(sensor_size=orig_sensor_size, var_x=1, var_y=1, clip_outliers=True),
		tonic.transforms.Downsample(spatial_factor=downsampling) if downsampling > 1 else Identity(),
		tonic.transforms.DropEventByArea(sensor_size=new_sensor_size, area_ratio=(0.05, 0.2)),
		tonic.transforms.UniformNoise(sensor_size=new_sensor_size, n=(0, noise))
	]
	if crop_events is not None:
		train_transforms.append(CropEvents(crop_events))

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

	train_loader, val_loader, test_loader = event_stream_dataloader(
		train_data, val_data, test_data,
		collate_fn=partial(event_stream_collate_fn, resolution=new_sensor_size[:2], pad_unit=pad_unit if crop_events is None else crop_events),
		batch_size=batch_size, eval_batch_size=eval_batch_size,
		rng=rng, num_workers=num_workers, shuffle_training=True
	)

	data = Data(
		n_classes=11, num_embeddings=np.prod(new_sensor_size), train_size=len(train_data)
	)
	return train_loader, val_loader, test_loader, data


Datasets = {
	"shd-classification": create_events_shd_classification_dataset,
	"ssc-classification": create_events_ssc_classification_dataset,
	"dvs-gesture-classification": create_events_dvs_gesture_classification_dataset,
}
