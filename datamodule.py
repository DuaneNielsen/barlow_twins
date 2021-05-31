from abc import ABC
from math import ceil, floor
from typing import Union, Optional, Any, List

import pytorch_lightning as pl
import torch
import torch.utils
from rich.progress import track
from torch.utils.data import DataLoader
from argparse import ArgumentParser

import buffer_h5 as b5
from sampler import SamplerFactory
import numpy as np
from torchvision import transforms
from transforms import GaussianBlur
import h5py as h5
from time import time
import math

class H5NextStateReward(b5.Buffer, torch.utils.data.Dataset, ABC):
    REWARD_NEG: int = 0
    REWARD_POS: int = 1

    def __init__(self):
        super().__init__()
        self.transforms = None
        self.classes = []
        self.image_shape = None
        self.image_dtype = None
        self.class_index = None
        self.reward_causality_distance = 5

    def load(self, filename, mode='r', cache_bytes=1073741824, cache_slots=100000, cache_w0=0.0, reward_causality_distance=5):
        super().load(filename, mode, cache_bytes, cache_slots, cache_w0)
        self.reward_causality_distance = reward_causality_distance
        self.classes = np.zeros(self.steps, dtype=np.int64)
        initials = self.episodes[:self.num_episodes]
        initials_mask = np.zeros(self.steps, dtype=np.uint8)
        initials_mask[initials] = 1
        initials_mask = initials_mask.astype(np.bool_)
        reward_pos_mask = self.reward[:self.steps] > 0.0
        self.image_shape = *self.f['/replay/raw'].shape[1:3], self.f['/replay/raw'].shape[3]*2
        self.image_dtype = self.f['/replay/raw'].dtype

        """
        When we hit a reward, flag the states that lead up to it as rewarding
        but not if they are from a different trajectory!
        """
        counter = 0
        for i in track(reversed(range(self.steps)), total=self.steps, description='[blue]indexing'):
            r = reward_pos_mask[i]
            initial = initials_mask[i]
            if r:
                counter = reward_causality_distance + 1
            if counter > 0:
                reward_pos_mask[i] = True
                if initial:
                    counter = 0
                else:
                    counter -= 1

        rewards_pos, = np.where(reward_pos_mask)
        rewards_neg, = np.where(reward_pos_mask == False)
        self.classes = reward_pos_mask.astype(dtype=np.int64)
        self.class_index = [rewards_neg, rewards_pos]

    def make_stat_table(self):
        table = super().make_stat_table()
        table.add_row("Reward causality distance", f'{self.reward_causality_distance}')
        table.add_row("Labeled reward_pos", f"{np.count_nonzero(self.classes)}")
        return table

    def __len__(self):
        return self.n_gram_len(gram_len=1)

    def __getitem__(self, item):
        raw = self.raw[item]
        grad = self.replay['grad'][item]
        image = np.concatenate((raw, grad), axis=2)
        x = self.transforms(image)
        label = self.classes[item]
        return x, label

    @property
    def num_classes(self) -> int:
        """
        Return:
            10
        """
        return 2

    @property
    def name_classes(self) -> List[str]:
        """
        Return:
            10
        """
        return ['non-reward', 'reward']


class PolicyActionLabels(b5.Buffer, torch.utils.data.Dataset, ABC):
    def __init__(self):
        super().__init__()
        self.transforms = None
        self.classes = None
        self.class_index = None
        self.image_shape = None
        self.image_dtype = None

    def build_class_index(self):
        idx = [[] for _ in range(self.num_classes)]
        for i in range(self.steps):
            idx[self.action[i]].append(i)
        return idx

    def load(self, filename, mode='r', cache_bytes=1073741824, cache_slots=100000, cache_w0=0.0, reward_causality_distance=5):
        super().load(filename, mode, cache_bytes, cache_slots, cache_w0)
        self.classes = self.action[:]
        self.class_index = self.build_class_index()
        self.image_shape = *self.f['/replay/raw'].shape[1:3], self.f['/replay/raw'].shape[3]*2
        self.image_dtype = self.f['/replay/raw'].dtype

    def make_stat_table(self):
        table = super().make_stat_table()
        return table

    def __len__(self):
        return self.n_gram_len(gram_len=1)

    def __getitem__(self, item):
        raw = self.raw[item]
        grad = self.replay['grad'][item]
        image = np.concatenate((raw, grad), axis=2)
        x = self.transforms(image)
        action = self.classes[item]
        return x, action

    @property
    def num_classes(self) -> int:
        """
        Return:
            10
        """
        return 4

    @property
    def name_classes(self) -> List[str]:
        """
        Return:
            10
        """
        return ['NOP', 'FIRE', 'RIGHT', 'LEFT']


def split_index(class_index):
    """

    :param class_index: nested lists of indexes of each class
    ie for classes
    labels = [1, 2, 0, 1, 2, 0, 0]
    class_index = [[2, 5, 6], [0, 3], [1, 4]]
    :return: test and validation splits with same proportion of clesses, in same format as class_index
    """
    train, val = [], []
    for i in range(len(class_index)):
        class_i = class_index[i]
        train.append(class_i[:ceil(len(class_i) * 0.8)])
        val.append(class_i[floor(len(class_i) * 0.8):])
    return train, val


def write_samples(group, dataset, class_dict, class_index, image_shape, batch_size, num_batches, compression,
                  compression_opts):

    sampler = SamplerFactory().get(
        class_idxs=class_index,
        batch_size=batch_size,
        n_batches=num_batches,
        alpha=0.5,
        kind='fixed'
    )

    len = batch_size * num_batches

    group.create_dataset('image',
                     shape=(len, *dataset.image_shape),
                     chunks=(batch_size, *dataset.image_shape),
                     dtype=dataset.image_dtype,
                     compression=compression,
                     compression_opts=compression_opts,
                     shuffle=False
                     )
    group.create_dataset('label',
                     shape=(len,),
                     chunks=(batch_size,),
                     dtype=h5.enum_dtype(class_dict, basetype=np.int64),
                     compression=compression,
                     compression_opts=compression_opts,
                     shuffle=False
                     )

    for i, (image, cls) in track(enumerate(DataLoader(dataset, batch_sampler=sampler, num_workers=0)),
                                 total=num_batches, description=f'[red] writing {group.name}'):
        offset = i * batch_size
        group['image'][offset:offset + batch_size] = image.numpy()
        group['label'][offset:offset + batch_size] = cls.numpy()


def write_balanced_splits(dataset, dest_filename, class_dict, batch_size, num_batches, compression, compression_opts):

    train_class_index, val_class_index = split_index(dataset.class_index)
    f = h5.File(dest_filename, mode='w')
    train = f.create_group('train')
    write_samples(train, dataset, class_dict, train_class_index, dataset.image_shape, batch_size, num_batches, compression,
                  compression_opts)
    val = f.create_group('val')
    write_samples(val, dataset, class_dict, val_class_index, dataset.image_shape, batch_size, num_batches//10, compression,
                  compression_opts)


class H5ImageLabelIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, filename, group, transform, batch_size):
        """

        :param filename:
        :param group: 'train' or 'val' will load train or validation set respectively
        :param transform:
        :param batch_size:
        """
        super().__init__()
        self.filename = filename
        self.group = group
        self.f = h5.File(self.filename, mode='r')
        self.offset = 0
        self.batch_size = batch_size
        self.transform = transform

    def __iter__(self):
        return self

    def __next__(self):
        g = self.f[self.group]

        start = self.offset
        end = self.offset + self.batch_size
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            iter_start = start
            iter_end = end
        else:  # in a worker process
            # split workload
            per_worker = int(math.ceil((end - start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, end)

        start = time()
        images = g['image'][iter_start:iter_end]
        labels = g['label'][iter_start:iter_end]
        load = time()
        images = list(zip(*[self.transform(image) for image in images]))
        x = tuple([torch.stack(items) for items in images])
        transform = time()
        #print(f'start: {iter_start} end:{iter_end} load: {load-start} transform: {transform-load}')
        self.offset += self.batch_size
        if self.offset > len(g['image']):
            raise StopIteration
        return x, labels

    def __len__(self):
        # lightning expects this to be the number of batches per epoch
        # for some reason I can't understand
        g = self.f[self.group]
        return len(g['label']) //16

    @property
    def num_classes(self) -> int:
        """
        Return:
            10
        """
        return len(self.name_classes)

    @property
    def name_classes(self) -> List[str]:
        with h5.File(self.filename, mode='r') as f:
            g = f[self.group]
            return list(g['label'].dtype.metadata['enum'])


class H5ImageLabelDataset(torch.utils.data.Dataset):
    def __init__(self, filename, group, transform, batch_size):
        """

        :param filename:
        :param group: 'train' or 'val' will load train or validation set respectively
        :param transform:
        :param batch_size: not used
        """
        super().__init__()
        self.filename = filename
        self.f = h5.File(filename, mode='r')
        self.g = self.f[group]
        self.offset = 0
        self.transform = transform

    def __getitem__(self, item):
        start = time()
        image = self.g['image'][item]
        label = self.g['label'][item]
        fetch = time()
        x = self.transform(image)
        transform = time()
        #print(fetch-start, transform-fetch)
        return x, label

    def __len__(self):
        return len(self.g['label'])

    @property
    def num_classes(self) -> int:
        """
        Return:
            10
        """
        return len(self.name_classes)

    @property
    def name_classes(self) -> List[str]:
        """
        Return:
            10
        """
        return list(self.g['label'].dtype.metadata['enum'])


class AtariDataModule(pl.LightningDataModule):
    def __init__(self, filename, train_transforms, val_transforms, test_transforms,
                 val_split: Union[int, float] = 0.2,
                 num_workers: int = 0,
                 normalize: bool = False,
                 batch_size: int = 32,
                 seed: int = 42,
                 shuffle: bool = False,
                 pin_memory: bool = False,
                 drop_last: bool = False,
                 batches_per_epoch: int = 1024
                 ):
        super().__init__(train_transforms, val_transforms, test_transforms)
        self.buffer = None
        self.train_set = None
        self.val_set = None
        self.val_split = val_split
        self.num_workers = num_workers
        self.normalize = normalize
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.seed = seed
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.val_sampler = None
        self.train_sampler = None

        self.train_set = H5ImageLabelIterableDataset(filename, 'train', train_transforms, batch_size=batch_size)
        #self.train_set = H5ImageLabelDataset(filename, 'train', train_transforms, batch_size=batch_size)
        self.train_set.transforms = self.train_transforms

        self.val_set = H5ImageLabelIterableDataset(filename, 'val', val_transforms, batch_size=batch_size)
        #self.val_set = H5ImageLabelDataset(filename, 'val', val_transforms, batch_size=batch_size)
        self.val_set.transforms = self.val_transforms


    def _collate(self, worker_data):
        xb, yb, zb, labels = [], [], [], []
        for (x, y, z), l in worker_data:
            xb.append(x)
            yb.append(y)
            zb.append(z)
            labels.append(l)
        xb = torch.cat(xb, dim=0)
        yb = torch.cat(yb, dim=0)
        zb = torch.cat(zb, dim=0)
        labels = torch.from_numpy(np.concatenate(labels, axis=0))
        return (xb, yb, zb), labels

    def _data_loader(self, dataset: torch.utils.data.Dataset) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.num_workers,
            num_workers=self.num_workers,
            pin_memory=False,
            shuffle=False,
            collate_fn=self._collate,
            prefetch_factor=1,
        )

    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        """ The train dataloader """
        return self._data_loader(self.train_set)

    def val_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The val dataloader """
        return self._data_loader(self.val_set)

    @property
    def num_classes(self) -> int:
        """
        Return:
            10
        """
        return self.train_set.num_classes

    @property
    def name_classes(self) -> List[str]:
        """
        Return:
            10
        """
        return self.train_set.name_classes


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--source_filename', required=True)
    parser.add_argument('--batch_size', required=True, type=int)
    parser.add_argument('--num_batches', required=True, type=int)
    parser.add_argument('--dest_filename', required=True)
    parser.add_argument('--compression_opts', type=int, default=6)
    parser.add_argument('--dataset', choices=['reward', 'action'], required=True)
    args = parser.parse_args()

    def identity(x):
        return x

    if args.dataset == 'reward':
        ds = H5NextStateReward()
        ds.transforms = identity
        ds.load(args.source_filename)

    elif args.dataset == 'action':
        ds = PolicyActionLabels()
        ds.transforms = identity
        ds.load(args.source_filename)
    else:
        raise Exception

    class_dict = dict(zip(ds.name_classes, range(len(ds.name_classes))))
    write_balanced_splits(ds, dest_filename=args.dest_filename, class_dict=class_dict,
                          batch_size=args.batch_size, num_batches=args.num_batches,
                          compression='gzip', compression_opts=args.compression_opts)