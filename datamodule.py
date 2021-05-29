from abc import ABC
from math import ceil, floor
from typing import Union, Optional, Any, List

import pytorch_lightning as pl
import torch
import torch.utils
from rich.progress import track
from torch.utils.data import DataLoader

import buffer_h5 as b5
from sampler import SamplerFactory
import numpy as np
from torchvision import transforms
from transforms import GaussianBlur


class H5NextStateReward(b5.Buffer, torch.utils.data.Dataset, ABC):
    REWARD_NEG: int = 0
    REWARD_POS: int = 1

    def __init__(self):
        super().__init__()
        self.transforms = None
        self.classes = []
        self.rewards_pos = []
        self.rewards_neg = []
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

        self.rewards_pos, = np.where(reward_pos_mask)
        self.rewards_neg, = np.where(reward_pos_mask == False)
        self.classes = reward_pos_mask.astype(dtype=np.int64)

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
        self.filename = filename
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

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_set = H5NextStateReward()
        self.train_set.load(self.filename)
        self.train_set.print_stats()
        self.train_set.transforms = self.train_transforms

        self.val_set = H5NextStateReward()
        self.val_set.load(self.filename)
        self.val_set.transforms = self.val_transforms

        def split_index(index):
            return index[:ceil(len(index) * 0.8)], index[floor(len(index) * 0.8):]

        train_rewards_pos, val_rewards_pos = split_index(self.train_set.rewards_pos)
        train_rewards_nonpos, val_rewards_nonpos = split_index(self.train_set.rewards_neg)

        self.train_sampler = SamplerFactory().get(
            class_idxs=[train_rewards_nonpos, train_rewards_pos],
            batch_size=self.batch_size,
            n_batches=self.batches_per_epoch,
            alpha=0.5,
            kind='fixed'
        )
        self.val_sampler = SamplerFactory().get(
            class_idxs=[train_rewards_nonpos, train_rewards_pos],
            batch_size=self.batch_size,
            n_batches=self.batches_per_epoch // 10,
            alpha=0.5,
            kind='fixed'
        )

    def _data_loader(self, dataset: torch.utils.data.Dataset,
                     batch_sampler,
                     shuffle: bool = False) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory
        )

    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        """ The train dataloader """
        return self._data_loader(self.train_set, batch_sampler=self.train_sampler, shuffle=self.shuffle)

    def val_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The val dataloader """
        return self._data_loader(self.val_set, batch_sampler=self.val_sampler, shuffle=self.shuffle)