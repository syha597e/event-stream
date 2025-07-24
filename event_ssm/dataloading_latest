import torch
from pathlib import Path
from typing import Callable, Optional, TypeVar, Dict, Tuple, List, Union
import tonic
from functools import partial
import numpy as np
import time

from event_ssm.transform import Identity, Roll, Rotate, Scale, DropEventChunk, Jitter1D, OneHotLabels, cut_mix_augmentation

_POS_EMB_CACHE: Dict[Tuple[int, int], np.ndarray] = {}


# Positional Encoding Implementation
class PositionalEncoding:
    def __init__(self, d_model, positions):
        self.d_model = d_model
        self.positions = positions
        self.pe = self._create_positional_encoding()

    def _create_positional_encoding(self):
        pe = np.zeros((self.positions, self.d_model), dtype=np.float32)
        position =np.arange(0, self.positions, dtype=np.float32)[:, None]
        div_term = np.exp(np.arange(0, self.d_model, 2) * (-np.log(10000.0) / self.d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        return pe

    def __call__(self, idx):
        return self.pe[idx]
    
def _preprocess_event(
    events: Dict[str, np.ndarray],
    resolution,
    no_time_information: bool,
    bin_size_ms: Optional[np.float32],
    pos_embeddings: Optional[Callable[[np.ndarray], np.ndarray]],
    sum_tokens_in_bins: bool,
):
    """Return token and timestep arrays for a single event sample."""

    t = events["t"].astype(np.int32)

    if bin_size_ms == 'None':
        bin_size_ms = None
    # breakpoint()
    if bin_size_ms is not None:
        bin_size_us=int(bin_size_ms* 1000.0) # convert to microseconds
        t = ((t // (bin_size_us))* bin_size_us).astype(np.int32)


    # breakpoint()
    if len(resolution) == 1:
        tok = events["x"].astype(np.int32)
    elif len(resolution) == 2:
        tok = (
            events["x"] * resolution[1]
            + events["y"]
            + np.prod(resolution) * events["p"].astype(np.int32)
        ).astype(np.int32)
    else:
        raise ValueError("resolution must contain 1 or 2 elements")

    if sum_tokens_in_bins:
        if bin_size_ms is None:
            emb = pos_embeddings(tok).astype(np.float32)
            tok = emb[:-1]
            
            if no_time_information:
                dt = np.ones_like(t[:-1])
            else:
                dt = np.diff(t)
            # breakpoint()
        # token_emb=PositionalEncoding(positions=700, d_model=96)
        # assert bin_size_ms is not None
        else:
            uniq, inv = np.unique(t, return_inverse=True)
            emb = pos_embeddings(tok).astype(np.float32) 
            agg = np.zeros((len(uniq), emb.shape[1]), dtype=np.float32)
            np.add.at(agg, inv, emb)
            tok = agg[:-1]
            # breakpoint()
    
            if no_time_information:
                dt = np.ones_like(uniq[:-1])
            else:
                dt = np.diff(uniq)
    else:
        if no_time_information:
            dt = np.ones_like(t[:-1])
        else:
            dt = np.diff(t)
        tok = tok[:-1]
    # padded_tokens = np.pad(tok, ((0, pad_length - len(tok)), (0, 0)), constant_values=0)
    # breakpoint()
    return tok, dt


import atexit
from multiprocessing import shared_memory
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

class InMemoryCachedDataset(torch.utils.data.Dataset):
    def __init__(self, raw_dataset, process_fn):
        self.tokens = []
        self.timesteps = []
        self.labels = []
        self.lengths = []
        self.raw_dataset = raw_dataset

        for i in range(len(raw_dataset)):
            events, label = raw_dataset[i]
            tokens_i, dt_i = process_fn(events)
            self.tokens.append(tokens_i.astype(np.float32))
            self.timesteps.append(dt_i.astype(np.float32))
            self.labels.append(label)
            self.lengths.append(len(dt_i))

        self.lengths = np.array(self.lengths, dtype=np.int32)
        max_len = self.lengths.max()
        pad_unit = max(1, 2 ** int(np.ceil(np.log2(max_len))))
        self.pad_length = ((max_len + pad_unit - 1) // pad_unit) * pad_unit

        # Pad everything once
        D = self.tokens[0].shape[1]
        self.tokens = np.stack([
            np.pad(t, ((0, self.pad_length - len(t)), (0, 0)), constant_values=0)
            for t in self.tokens
        ])
        self.timesteps = np.stack([
            np.pad(dt, (0, self.pad_length - len(dt)), constant_values=0)
            for dt in self.timesteps
        ])
        # self.labels = np.array(self.labels, dtype=np.int32)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return self.tokens[idx], self.timesteps[idx], self.labels[idx], self.lengths[idx]

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


def event_stream_collate_fn(batch, resolution, pad_unit, cut_mix=0.0, no_time_information: bool = False, sum_tokens_in_bins: bool=False, preload_tokens: bool = False,):

    # x are inputs, y are targets, z are aux data

    # x, t, y, *z = zip(*batch)
    tokens, dt, labels,lengths, *z = zip(*batch)
    # x, y, *z = zip(*batc)
    assert len(z) == 0
    batch_size_one = len(tokens) == 1
    # breakpoint()


    # preprocessed = isinstance(x[0], dict) and "tokens" in x[0]
    if preload_tokens:
        # tokens = [e["tokens"] for e in x]
        # timesteps = [e["timesteps"] for e in x]

        # tokens=torch.stack(x).numpy()
        # timesteps=torch.stack(t).numpy()
        # y = torch.stack(y).numpy()

        # lengths = np.array([len(e) for e in dt], dtype=np.int32)
        tokens = np.stack(tokens)
        dt = np.stack(dt)
        labels = np.stack(labels)
        lengths=np.array(lengths)

        # tokens = torch.stack(tokens).numpy()
        # breakpoint()

  
    timesteps = dt / 1000
    
return tokens, labels, timesteps, lengths


def event_stream_dataloader(train_data, val_data, test_data, batch_size, eval_batch_size, train_collate_fn, eval_collate_fn, rng, num_workers=0, shuffle_training=True):
    def dataloader(dset, bsz, collate_fn, shuffle, drop_last):
        return torch.utils.data.DataLoader(
            dset,
            batch_size=bsz,
            drop_last=drop_last,
            collate_fn=collate_fn,
            shuffle=shuffle,
            generator=rng,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            # pin_memory=True,
            prefetch_factor=8,
        )
    train_loader = dataloader(train_data, batch_size, train_collate_fn, shuffle=shuffle_training, drop_last=True)
    val_loader = dataloader(val_data, eval_batch_size, eval_collate_fn, shuffle=False, drop_last=True)
    test_loader = dataloader(test_data, eval_batch_size, eval_collate_fn, shuffle=False, drop_last=False)
    return train_loader, val_loader, test_loader


def create_events_shd_classification_dataset(
        cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT,
        per_device_batch_size: int = 32,
        per_device_eval_batch_size: int = 64,
        world_size: int = 1,
        num_workers: int = 0,
        seed: int = 42,
        time_jitter: float = 100,
        spatial_jitter: float = 1.0,
        max_drop_chunk: float = 0.1,
        noise: int = 100,
        drop_event: float = 0.1,
        time_skew: float = 1.1,
        cut_mix: float = 0.5,
        pad_unit: int = 8192,
        d_model: int = 96,

        bin_size_ms: float = None,
        sum_tokens_in_bins: bool = True,
        preload_tokens: bool = True,

        validate_on_test: bool = False,
        no_time_information: bool = False,
        **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader, Data]:
    """

   Create the Spiking Heidelberg Digits dataset loaders.

    Optional parameters allow precomputation of token embeddings. When an
    ``embedding_table`` is supplied and ``sum_tokens_in_bins`` is ``True`` the
    collate function sums embeddings of all events that fall into the same
    ``bin_size_ms`` window. Setting ``positional_embedding_dim`` adds sinusoidal
    encodings of this size to the returned batch. ``embedding_dim`` controls the
    size of a randomly initialised table when ``embedding_table`` is not
    provided.  ``preload_tokens`` caches tokenised samples in memory to reduce
    data loading overhead.
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
        DropEventChunk(p=0.3, max_drop_size=max_drop_chunk),
        Jitter1D(sensor_size=sensor_size, var=spatial_jitter),
        tonic.transforms.TimeSkew(coefficient=(1 / time_skew, time_skew), offset=0),
        tonic.transforms.TimeJitter(std=time_jitter, clip_negative=False, sort_timestamps=True),
        tonic.transforms.UniformNoise(sensor_size=sensor_size, n=(0, noise)),
        # BinTimestamps(bin_ms=bin_size_ms)
    ])
    target_transforms = OneHotLabels(num_classes=20)

    train_data = tonic.datasets.SHD(save_to=cache_dir, train=True, transform=transforms, target_transform=target_transforms)
    val_data = tonic.datasets.SHD(save_to=cache_dir, train=True, target_transform=target_transforms)
    test_data = tonic.datasets.SHD(save_to=cache_dir, train=False, target_transform=target_transforms)

    # create validation set
    if validate_on_test:
        print("[*] WARNING: Using test set for validation")
        val_data = tonic.datasets.SHD(save_to=cache_dir, train=False, target_transform=target_transforms)
    else:
        val_length = int(0.1 * len(train_data))
        indices = torch.randperm(len(train_data), generator=rng)
        train_data = torch.utils.data.Subset(train_data, indices[:-val_length])
        val_data = torch.utils.data.Subset(val_data, indices[-val_length:])
    
    pos_embeddings = PositionalEncoding(d_model=d_model, positions=700) if sum_tokens_in_bins else None

    process_fn = partial(
        _preprocess_event,
        resolution=(700,),
        no_time_information=no_time_information,
        bin_size_ms=bin_size_ms,
        pos_embeddings=pos_embeddings,
        sum_tokens_in_bins=sum_tokens_in_bins,
    )
    import time
    import tracemalloc
    import psutil
    import os
    def log_memory_usage(tag="[Memory]"):
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / 1024**2
        print(f"{tag} Memory RSS: {mem_mb:.2f} MB")


    if preload_tokens:
        # shm_log_path="/data/horse/ws/syha597e-pw_workspace/syha597e-my_workspace/git-data/position-embeddings/event-stream-modeling-s5/shm_blocks.txt"
        
        print("[*] Preloading tokens into memory...")
        start_time = time.time()
        tracemalloc.start()
        # Start profiling    
        log_memory_usage("[Before]")


        train_data = InMemoryCachedDataset(train_data, process_fn)
        val_data = InMemoryCachedDataset(val_data, process_fn)
        test_data = InMemoryCachedDataset(test_data, process_fn)

        
        log_memory_usage("[After]")
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(f"[⏱] Preload Time: {time.time() - start_time:.2f}s")
        print(f"[📦] Memory Usage: Current = {current / 1e6:.2f} MB; Peak = {peak / 1e6:.2f} MB")
        # breakpoint()

    collate_fn = partial(
        event_stream_collate_fn,
        resolution=(700,),
        pad_unit=pad_unit,
        cut_mix=cut_mix,
        no_time_information=no_time_information,
        # bin_size_ms=bin_size_ms,
        # pos_embeddings=pos_embeddings,
        sum_tokens_in_bins=sum_tokens_in_bins,
        preload_tokens=preload_tokens,
    )
    train_loader, val_loader, test_loader = event_stream_dataloader(
        train_data, val_data, test_data,
        train_collate_fn=partial(collate_fn, cut_mix=cut_mix),
        eval_collate_fn=collate_fn,
        batch_size=per_device_batch_size * world_size, eval_batch_size=per_device_eval_batch_size * world_size,
        rng=rng, num_workers=num_workers, shuffle_training=True
    )
    data = Data(
        n_classes=20, num_embeddings=700, train_size=len(train_data)
    )
    # breakpoint()
    return train_loader, val_loader, test_loader, data

Datasets = {
    "shd-classification": create_events_shd_classification_dataset,
}
