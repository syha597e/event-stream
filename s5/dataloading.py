import torch
from pathlib import Path
from typing import Callable, Optional, TypeVar, Dict, Tuple, List, Union
import tonic
from functools import partial
from torch.nn.functional import pad
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
	timesteps = torch.stack([pad(e, (0, max_length - l), 'constant', e[-1]) for e, l in zip(timesteps, lengths)])

	# timesteps are in micro seconds... transform to milliseconds
	timesteps = timesteps / 1000

	# pad tokens with -1, which results in a zero vector with jax.nn.one_hot
	tokens = torch.stack([pad(e, (0, max_length - l), 'constant', -1) for e, l in zip(tokens, lengths)])

	y = torch.tensor(y).int()

	tokens = tokens.numpy()
	y = y.numpy()
	timesteps = timesteps.numpy()
	lengths = lengths.numpy()

	return tokens, y, timesteps, lengths


def event_frame_collate_fn(batch, time_window):
	# x are inputs, y are targets, z are aux data
	x, y, *z = zip(*batch)
	assert len(z) == 0

	# set tonic specific items
	timesteps = [torch.arange(0, e.shape[0]*time_window, time_window, dtype=torch.long) for e in x]

	# pad time steps with final time-step -> this is a bit of a hack to make the integration time steps 0
	lengths = torch.tensor([len(e) for e in timesteps], dtype=torch.long)

	max_length = lengths.max().item()
 
	timesteps = torch.stack([pad(e, (0, max_length - l), 'constant', e[-1]) for e, l in zip(timesteps, lengths)])

	tokens = [e.squeeze() for e in x]
	
	# timesteps are in micro seconds... transform to milliseconds
	timesteps = timesteps / 1000

	# pad tokens with -1, which results in a zero vector with jax.nn.one_hot
	tokens = np.stack([np.pad(e, ((0, max_length - l), (0,0)), 'constant', constant_values=0) for e, l in zip(tokens, lengths)])

	y = torch.tensor(y).int()

	# tokens = tokens.numpy()
	y = y.numpy()
	timesteps = timesteps.numpy()
	lengths = lengths.numpy()

	return tokens, y, timesteps, lengths

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
		collate_fn=partial(event_stream_collate_fn, resolution=(700,)),
		batch_size=batch_size, eval_batch_size=eval_batch_size,
		rng=rng, num_workers=num_workers, shuffle_training=True
	)

	return Data(
		train_loader, val_loader, test_loader, aux_loaders={},
		n_classes=20, in_dim=700, train_pad_length=16384, test_pad_length=16384, train_size=len(train_data)
	)

def create_frame_shd_classification_dataset(
		cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT,
		batch_size: int = 32,
		eval_batch_size: int = 64,
		num_workers: int = 0,
		seed: int = 42,
		time_jitter: float = 100,
		noise: int = 100,
		drop_event: float = 0.1,
		time_skew: float = 1.1,
		validate_on_test: bool = False,
		time_window: int = 10000,
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

	train_transforms = tonic.transforms.Compose([
		tonic.transforms.DropEvent(p=drop_event),
		tonic.transforms.TimeSkew(coefficient=(1 / time_skew, time_skew), offset=0),
		tonic.transforms.TimeJitter(std=time_jitter, clip_negative=False, sort_timestamps=True),
		# tonic.transforms.UniformNoise(sensor_size=(128, 128, 2), n=(0, noise)),
    	tonic.transforms.ToFrame(sensor_size=tonic.datasets.SHD.sensor_size, time_window=time_window, include_incomplete=True)
	])
	eval_transforms =  tonic.transforms.Compose([
    	tonic.transforms.ToFrame(sensor_size=tonic.datasets.SHD.sensor_size, time_window=time_window, include_incomplete=True)
	])
	train_data = tonic.datasets.SHD(save_to=cache_dir, train=True, transform=train_transforms)

	if validate_on_test:
		print("WARNING: Using test set for validation")
		val_data = tonic.datasets.SHD(save_to=cache_dir, train=False, transform=eval_transforms)
	else:
		val_length = int(0.1 * len(train_data))
		train_data, val_data = torch.utils.data.random_split(
			train_data,
			lengths=[len(train_data) - val_length, val_length],
			generator=rng
		)
	test_data = tonic.datasets.SHD(save_to=cache_dir, train=False, transform=eval_transforms)

	train_loader, val_loader, test_loader = event_stream_dataloader(
		train_data, val_data, test_data,
		collate_fn=partial(event_frame_collate_fn, time_window=time_window),
		batch_size=batch_size, eval_batch_size=eval_batch_size, rng=rng, shuffle_training=True
	)

	return Data(
		train_loader, val_loader, test_loader, aux_loaders={},
		n_classes=20, in_dim=700, train_pad_length=512*10000//time_window, test_pad_length=512*10000//time_window, train_size=len(train_data)
	)


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
		collate_fn=partial(event_stream_collate_fn, resolution=(700,)),
		batch_size=batch_size, eval_batch_size=eval_batch_size,
		rng=rng, num_workers=num_workers, shuffle_training=True
	)

	return Data(
		train_loader, val_loader, test_loader, aux_loaders={},
		n_classes=35, in_dim=700, train_pad_length=20480, test_pad_length=20480, train_size=len(train_data)
	)

def create_frame_ssc_classification_dataset(
		cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT,
		batch_size: int = 32,
		eval_batch_size: int = 64,
		num_workers: int = 0,
		seed: int = 42,
		time_jitter: float = 100,
		noise: int = 100,
		drop_event: float = 0.1,
		time_skew: float = 1.1,
		time_window: int = 10000,
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
		# tonic.transforms.UniformNoise(sensor_size=(128, 128, 2), n=(0, noise))
		tonic.transforms.ToFrame(sensor_size=tonic.datasets.SSC.sensor_size, time_window=time_window, include_incomplete=True)
	])
	eval_transforms =  tonic.transforms.Compose([
    	tonic.transforms.ToFrame(sensor_size=tonic.datasets.SSC.sensor_size, time_window=time_window, include_incomplete=True)
	])
	train_data = tonic.datasets.SSC(save_to=cache_dir, split='train', transform=transforms)
	val_data = tonic.datasets.SSC(save_to=cache_dir, split='valid',transform=eval_transforms)
	test_data = tonic.datasets.SSC(save_to=cache_dir, split='test',transform=eval_transforms)

	train_loader, val_loader, test_loader = event_stream_dataloader(
		train_data, val_data, test_data,
		collate_fn=partial(event_frame_collate_fn, time_window=time_window),
		batch_size=batch_size, eval_batch_size=eval_batch_size, num_workers=num_workers, rng=rng, shuffle_training=True
	)

	return Data(
		train_loader, val_loader, test_loader, aux_loaders={},
		n_classes=35, in_dim=700, train_pad_length=512*10000//time_window, test_pad_length=512*10000//time_window, train_size=len(train_data)
	)
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
		**kwargs
) -> Data:
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
		collate_fn=partial(event_stream_collate_fn, resolution=new_sensor_size[:2]),
		batch_size=batch_size, eval_batch_size=eval_batch_size,
		rng=rng, num_workers=num_workers, shuffle_training=True
	)

	return Data(
		train_loader, val_loader, test_loader, aux_loaders={},
		n_classes=11, in_dim=np.prod(new_sensor_size), train_pad_length=crop_events, test_pad_length=1595392, train_size=len(train_data)
	)

Datasets = {
	"shd-classification": create_events_shd_classification_dataset,
	"ssc-classification": create_events_ssc_classification_dataset,
	"dvs-gesture-classification": create_events_dvs_gesture_classification_dataset,
	"shd-frame-classification": create_frame_shd_classification_dataset,
	"ssc-frame-classification": create_frame_ssc_classification_dataset,
}
