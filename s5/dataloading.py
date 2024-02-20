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
ReturnType = Tuple[DataLoader, DataLoader, DataLoader, Dict, int, int, int, int]

# Custom loading functions must therefore have the template.
dataset_fn = Callable[[str, Optional[int], Optional[int]], ReturnType]


# Example interface for making a loader.
def custom_loader(cache_dir: str,
				  bsz: int = 50,
				  seed: int = 42) -> ReturnType:
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
		seed: int = 42
) -> ReturnType:
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

	aux_loaders = {}
	N_CLASSES = 20
	SEQ_LENGTH = 16384  # if sequence length is longer than the inputs seq len, will pad in function `prep_batch`
	IN_DIM = 700
	TRAIN_SIZE = len(train_data)

	return train_loader, val_loader, test_loader, aux_loaders, N_CLASSES, SEQ_LENGTH, IN_DIM, TRAIN_SIZE


def create_events_ssc_classification_dataset(
		cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT,
		bsz: int = 50,
		seed: int = 42
) -> ReturnType:
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

	aux_loaders = {}
	N_CLASSES = 35
	SEQ_LENGTH = 20480  # choose a multiple of 1024 that is just larger than the longest sequence
	IN_DIM = 700
	TRAIN_SIZE = len(train_data)

	return train_loader, val_loader, test_loader, aux_loaders, N_CLASSES, SEQ_LENGTH, IN_DIM, TRAIN_SIZE


def create_events_dvs_gesture_classification_dataset(
		cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT,
		bsz: int = 50,
		seed: int = 42
) -> ReturnType:
	"""
	creates a view of the DVS Gesture dataset

	:param cache_dir:		(str):		where to store the dataset
	:param bsz:				(int):		Batch size.
	:param seed:			(int)		Seed for shuffling data.
	"""
	print("[*] Generating DVS Gesture Classification Dataset")

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
		collate_fn=partial(event_stream_collate_fn, resolution=(128, 128), max_time=1e6),
		bsz=bsz, rng=rng, shuffle_training=False
	)

	aux_loaders = {}
	N_CLASSES = 11
	SEQ_LENGTH = 2 * 131072  # if sequence length is longer than the inputs seq len, will pad in function `prep_batch`
	IN_DIM = 128 * 128 * 2
	TRAIN_SIZE = len(train_data)

	return train_loader, val_loader, test_loader, aux_loaders, N_CLASSES, SEQ_LENGTH, IN_DIM, TRAIN_SIZE


def create_lra_imdb_classification_dataset(cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT,
										   bsz: int = 50,
										   seed: int = 42) -> ReturnType:
	"""

	:param cache_dir:		(str):		Not currently used.
	:param bsz:				(int):		Batch size.
	:param seed:			(int)		Seed for shuffling data.
	:return:
	"""
	print("[*] Generating LRA-text (IMDB) Classification Dataset")
	from s5.dataloaders.lra import IMDB
	name = 'imdb'

	dataset_obj = IMDB('imdb', )
	dataset_obj.cache_dir = Path(cache_dir) / name
	dataset_obj.setup()

	trainloader = make_data_loader(dataset_obj.dataset_train, dataset_obj, seed=seed, batch_size=bsz)
	testloader = make_data_loader(dataset_obj.dataset_test, dataset_obj, seed=seed, batch_size=bsz, drop_last=False, shuffle=False)
	valloader = None

	N_CLASSES = dataset_obj.d_output
	SEQ_LENGTH = dataset_obj.l_max
	IN_DIM = 135  # We should probably stop this from being hard-coded.
	TRAIN_SIZE = len(dataset_obj.dataset_train)

	aux_loaders = {}

	return trainloader, valloader, testloader, aux_loaders, N_CLASSES, SEQ_LENGTH, IN_DIM, TRAIN_SIZE


def create_lra_listops_classification_dataset(cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT,
											  bsz: int = 50,
											  seed: int = 42) -> ReturnType:
	"""
	See abstract template.
	"""
	print("[*] Generating LRA-listops Classification Dataset")
	from s5.dataloaders.lra import ListOps
	name = 'listops'
	dir_name = './raw_datasets/lra_release/lra_release/listops-1000'

	dataset_obj = ListOps(name, data_dir=dir_name)
	dataset_obj.cache_dir = Path(cache_dir) / name
	dataset_obj.setup()

	trn_loader = make_data_loader(dataset_obj.dataset_train, dataset_obj, seed=seed, batch_size=bsz)
	val_loader = make_data_loader(dataset_obj.dataset_val, dataset_obj, seed=seed, batch_size=bsz, drop_last=False, shuffle=False)
	tst_loader = make_data_loader(dataset_obj.dataset_test, dataset_obj, seed=seed, batch_size=bsz, drop_last=False, shuffle=False)

	N_CLASSES = dataset_obj.d_output
	SEQ_LENGTH = dataset_obj.l_max
	IN_DIM = 20
	TRAIN_SIZE = len(dataset_obj.dataset_train)

	aux_loaders = {}

	return trn_loader, val_loader, tst_loader, aux_loaders, N_CLASSES, SEQ_LENGTH, IN_DIM, TRAIN_SIZE


def create_lra_path32_classification_dataset(cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT,
											 bsz: int = 50,
											 seed: int = 42) -> ReturnType:
	"""
	See abstract template.
	"""
	print("[*] Generating LRA-Pathfinder32 Classification Dataset")
	from s5.dataloaders.lra import PathFinder
	name = 'pathfinder'
	resolution = 32
	dir_name = f'./raw_datasets/lra_release/lra_release/pathfinder{resolution}'

	dataset_obj = PathFinder(name, data_dir=dir_name, resolution=resolution)
	dataset_obj.cache_dir = Path(cache_dir) / name
	dataset_obj.setup()

	trn_loader = make_data_loader(dataset_obj.dataset_train, dataset_obj, seed=seed, batch_size=bsz)
	val_loader = make_data_loader(dataset_obj.dataset_val, dataset_obj, seed=seed, batch_size=bsz, drop_last=False, shuffle=False)
	tst_loader = make_data_loader(dataset_obj.dataset_test, dataset_obj, seed=seed, batch_size=bsz, drop_last=False, shuffle=False)

	N_CLASSES = dataset_obj.d_output
	SEQ_LENGTH = dataset_obj.dataset_train.tensors[0].shape[1]
	IN_DIM = dataset_obj.d_input
	TRAIN_SIZE = dataset_obj.dataset_train.tensors[0].shape[0]

	aux_loaders = {}

	return trn_loader, val_loader, tst_loader, aux_loaders, N_CLASSES, SEQ_LENGTH, IN_DIM, TRAIN_SIZE


def create_lra_pathx_classification_dataset(cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT,
											bsz: int = 50,
											seed: int = 42) -> ReturnType:
	"""
	See abstract template.
	"""
	print("[*] Generating LRA-PathX Classification Dataset")
	from s5.dataloaders.lra import PathFinder
	name = 'pathfinder'
	resolution = 128
	dir_name = f'./raw_datasets/lra_release/lra_release/pathfinder{resolution}'

	dataset_obj = PathFinder(name, data_dir=dir_name, resolution=resolution)
	dataset_obj.cache_dir = Path(cache_dir) / name
	dataset_obj.setup()

	trn_loader = make_data_loader(dataset_obj.dataset_train, dataset_obj, seed=seed, batch_size=bsz)
	val_loader = make_data_loader(dataset_obj.dataset_val, dataset_obj, seed=seed, batch_size=bsz, drop_last=False, shuffle=False)
	tst_loader = make_data_loader(dataset_obj.dataset_test, dataset_obj, seed=seed, batch_size=bsz, drop_last=False, shuffle=False)

	N_CLASSES = dataset_obj.d_output
	SEQ_LENGTH = dataset_obj.dataset_train.tensors[0].shape[1]
	IN_DIM = dataset_obj.d_input
	TRAIN_SIZE = dataset_obj.dataset_train.tensors[0].shape[0]

	aux_loaders = {}

	return trn_loader, val_loader, tst_loader, aux_loaders, N_CLASSES, SEQ_LENGTH, IN_DIM, TRAIN_SIZE


def create_lra_image_classification_dataset(cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT,
											seed: int = 42,
											bsz: int=128) -> ReturnType:
	"""
	See abstract template.

	Cifar is quick to download and is automatically cached.
	"""

	print("[*] Generating LRA-listops Classification Dataset")
	from s5.dataloaders.basic import CIFAR10
	name = 'cifar'

	kwargs = {
		'grayscale': True,  # LRA uses a grayscale CIFAR image.
	}

	dataset_obj = CIFAR10(name, data_dir=cache_dir, **kwargs)  # TODO - double check what the dir here does.
	dataset_obj.setup()

	trn_loader = make_data_loader(dataset_obj.dataset_train, dataset_obj, seed=seed, batch_size=bsz)
	val_loader = make_data_loader(dataset_obj.dataset_val, dataset_obj, seed=seed, batch_size=bsz, drop_last=False, shuffle=False)
	tst_loader = make_data_loader(dataset_obj.dataset_test, dataset_obj, seed=seed, batch_size=bsz, drop_last=False, shuffle=False)

	N_CLASSES = dataset_obj.d_output
	SEQ_LENGTH = 32 * 32
	IN_DIM = 1
	TRAIN_SIZE = len(dataset_obj.dataset_train)

	aux_loaders = {}

	return trn_loader, val_loader, tst_loader, aux_loaders, N_CLASSES, SEQ_LENGTH, IN_DIM, TRAIN_SIZE


def create_lra_aan_classification_dataset(cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT,
										  bsz: int = 50,
										  seed: int = 42, ) -> ReturnType:
	"""
	See abstract template.
	"""
	print("[*] Generating LRA-AAN Classification Dataset")
	from s5.dataloaders.lra import AAN
	name = 'aan'

	dir_name = './raw_datasets/lra_release/lra_release/tsv_data'

	kwargs = {
		'n_workers': 1,  # Multiple workers seems to break AAN.
	}

	dataset_obj = AAN(name, data_dir=dir_name, **kwargs)
	dataset_obj.cache_dir = Path(cache_dir) / name
	dataset_obj.setup()

	trn_loader = make_data_loader(dataset_obj.dataset_train, dataset_obj, seed=seed, batch_size=bsz)
	val_loader = make_data_loader(dataset_obj.dataset_val, dataset_obj, seed=seed, batch_size=bsz, drop_last=False, shuffle=False)
	tst_loader = make_data_loader(dataset_obj.dataset_test, dataset_obj, seed=seed, batch_size=bsz, drop_last=False, shuffle=False)

	N_CLASSES = dataset_obj.d_output
	SEQ_LENGTH = dataset_obj.l_max
	IN_DIM = len(dataset_obj.vocab)
	TRAIN_SIZE = len(dataset_obj.dataset_train)

	aux_loaders = {}

	return trn_loader, val_loader, tst_loader, aux_loaders, N_CLASSES, SEQ_LENGTH, IN_DIM, TRAIN_SIZE


def create_speechcommands35_classification_dataset(cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT,
												   bsz: int = 50,
												   seed: int = 42) -> ReturnType:
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

	return trn_loader, val_loader, tst_loader, aux_loaders, N_CLASSES, SEQ_LENGTH, IN_DIM, TRAIN_SIZE


def create_cifar_classification_dataset(cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT,
										seed: int = 42,
										bsz: int=128) -> ReturnType:
	"""
	See abstract template.

	Cifar is quick to download and is automatically cached.
	"""

	print("[*] Generating CIFAR (color) Classification Dataset")
	from s5.dataloaders.basic import CIFAR10
	name = 'cifar'

	kwargs = {
		'grayscale': False,  # LRA uses a grayscale CIFAR image.
	}

	dataset_obj = CIFAR10(name, data_dir=cache_dir, **kwargs)
	dataset_obj.setup()

	trn_loader = make_data_loader(dataset_obj.dataset_train, dataset_obj, seed=seed, batch_size=bsz)
	val_loader = make_data_loader(dataset_obj.dataset_val, dataset_obj, seed=seed, batch_size=bsz, drop_last=False, shuffle=False)
	tst_loader = make_data_loader(dataset_obj.dataset_test, dataset_obj, seed=seed, batch_size=bsz, drop_last=False, shuffle=False)

	N_CLASSES = dataset_obj.d_output
	SEQ_LENGTH = 32 * 32
	IN_DIM = 3
	TRAIN_SIZE = len(dataset_obj.dataset_train)

	aux_loaders = {}

	return trn_loader, val_loader, tst_loader, aux_loaders, N_CLASSES, SEQ_LENGTH, IN_DIM, TRAIN_SIZE


def create_mnist_classification_dataset(cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT,
																				seed: int = 42,
																				bsz: int=128) -> ReturnType:
	"""
	See abstract template.

	Cifar is quick to download and is automatically cached.
	"""

	print("[*] Generating MNIST Classification Dataset")
	from s5.dataloaders.basic import MNIST
	name = 'mnist'

	kwargs = {
		'permute': False
	}

	dataset_obj = MNIST(name, data_dir=cache_dir, **kwargs)
	dataset_obj.setup()

	trn_loader = make_data_loader(dataset_obj.dataset_train, dataset_obj, seed=seed, batch_size=bsz)
	val_loader = make_data_loader(dataset_obj.dataset_val, dataset_obj, seed=seed, batch_size=bsz, drop_last=False, shuffle=False)
	tst_loader = make_data_loader(dataset_obj.dataset_test, dataset_obj, seed=seed, batch_size=bsz, drop_last=False, shuffle=False)

	N_CLASSES = dataset_obj.d_output
	SEQ_LENGTH = 28 * 28
	IN_DIM = 1
	TRAIN_SIZE = len(dataset_obj.dataset_train)
	aux_loaders = {}
	return trn_loader, val_loader, tst_loader, aux_loaders, N_CLASSES, SEQ_LENGTH, IN_DIM, TRAIN_SIZE


def create_pmnist_classification_dataset(cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT,
																				seed: int = 42,
																				bsz: int=128) -> ReturnType:
	"""
	See abstract template.

	Cifar is quick to download and is automatically cached.
	"""

	print("[*] Generating permuted-MNIST Classification Dataset")
	from s5.dataloaders.basic import MNIST
	name = 'mnist'

	kwargs = {
		'permute': True
	}

	dataset_obj = MNIST(name, data_dir=cache_dir, **kwargs)
	dataset_obj.setup()

	trn_loader = make_data_loader(dataset_obj.dataset_train, dataset_obj, seed=seed, batch_size=bsz)
	val_loader = make_data_loader(dataset_obj.dataset_val, dataset_obj, seed=seed, batch_size=bsz, drop_last=False, shuffle=False)
	tst_loader = make_data_loader(dataset_obj.dataset_test, dataset_obj, seed=seed, batch_size=bsz, drop_last=False, shuffle=False)

	N_CLASSES = dataset_obj.d_output
	SEQ_LENGTH = 28 * 28
	IN_DIM = 1
	TRAIN_SIZE = len(dataset_obj.dataset_train)
	aux_loaders = {}
	return trn_loader, val_loader, tst_loader, aux_loaders, N_CLASSES, SEQ_LENGTH, IN_DIM, TRAIN_SIZE


Datasets = {
	# Other loaders.
	"mnist-classification": create_mnist_classification_dataset,
	"pmnist-classification": create_pmnist_classification_dataset,
	"cifar-classification": create_cifar_classification_dataset,

	# LRA.
	"imdb-classification": create_lra_imdb_classification_dataset,
	"listops-classification": create_lra_listops_classification_dataset,
	"aan-classification": create_lra_aan_classification_dataset,
	"lra-cifar-classification": create_lra_image_classification_dataset,
	"pathfinder-classification": create_lra_path32_classification_dataset,
	"pathx-classification": create_lra_pathx_classification_dataset,

	# Speech.
	"speech35-classification": create_speechcommands35_classification_dataset,

	# Events
	"shd-classification": create_events_shd_classification_dataset,
	"ssc-classification": create_events_ssc_classification_dataset,
	"dvs-gesture-classification": create_events_dvs_gesture_classification_dataset,
}
