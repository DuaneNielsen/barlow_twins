from abc import ABC
from math import ceil, floor
from statistics import mean
from typing import Union, Optional, Any, List

import pytorch_lightning as pl
import rich.table
import torch
import torch.utils
from rich.progress import track
from rich import print
from torch.utils.data import DataLoader

import buffer_h5 as b5
from rl import OnDiskReplayBuffer
from sampler import SamplerFactory
import time
import numpy as np


class RefactoredNextStateReward(OnDiskReplayBuffer, torch.utils.data.Dataset, ABC):
    REWARD_NEG: int = 0
    REWARD_POS: int = 1

    def __init__(self):
        super().__init__()
        self.transforms = None
        self.classes = []
        self.rewards_pos = []
        self.rewards_neg = []

    def load(self, filename, print_stats=False, **kwargs):
        super().load(filename)

        if kwargs['reward_class'] == 'monte_carlo':
            """
            Classify states with a discounted monte_carlo value greater than threshold as "good"
            """

            monte_carlo_values = self.compute_monte_carlo_values(discount=kwargs['reward_monte_carlo_discount'])
            for i, v in enumerate(monte_carlo_values):
                if v < kwargs['reward_threshold']:
                    self.rewards_neg.append(i)
                    self.classes.append(self.REWARD_NEG)
                else:
                    self.rewards_pos.append(i)
                    self.classes.append(self.REWARD_POS)

        elif kwargs['reward_class'] == 'positive_reward':
            """
            Classify only states with positive reward as "good"
            """
            self.rewards_pos = self.transitions.get_where_list('reward > 0')
            self.rewards_neg = self.transitions.get_where_list('reward <= 0')
            self.classes = [0] * sum(len(self.rewards_pos) + len(self.rewards_neg))
            for i in self.rewards_pos:
                self.classes[i] = self.REWARD_POS

        elif kwargs['reward_class'] == 'distance':
            """
            Classify states within a causality distance of a reward transition as "good"
            """
            self.classes = [0] * len(self.transitions)
            rewards_pos = self.transitions.get_where_list('reward > 0')
            for offset in rewards_pos:
                for j in reversed(range(kwargs['reward_causality_distance'])):
                    if offset - j < 0:
                        break
                    if self.transitions[offset -j]['done']:
                        break
                    self.rewards_pos.append(offset - j)
                    self.classes[offset - j] = self.REWARD_POS

            for i, cls in enumerate(self.classes):
                if cls != self.REWARD_POS:
                    self.rewards_neg.append(i)

        if print_stats:
            self.print_stats()
        return self

    def __getitem__(self, item):
        trans = self.transitions[item]
        s_p = self.states[trans['next_state']]
        s_p = self.transforms(s_p)
        r = self.classes[item]
        return s_p, r

    def print_stats(self):
        table = rich.table.Table(title=f"{self.__class__.__name__}")
        table.add_column("Stat", justify="right", style="cyan", no_wrap=True)
        table.add_column("Title", style="magenta")
        table.add_row("File", f'{self.fileh.filename}')
        table.add_row("Episodes", f"{len(self.episodes)}")
        epi_lengths = [(end - start).item() for start, end in self.fileh.root.replay.Episodes[:]]
        table.add_row("Mean episode len", f"{mean(epi_lengths)}")
        table.add_row("Transitions", f"{len(self)}")
        table.add_row("Transitions with + reward", f"{len(self.transitions.get_where_list('reward > 0'))}")
        table.add_row("Transitions with - reward", f"{len(self.transitions.get_where_list('reward < 0'))}")
        table.add_row("Transitions with 0 reward", f"{len(self.transitions.get_where_list('reward == 0'))}")
        table.add_row("Transitions labeled 0", f"{len(self.rewards_neg)}")
        table.add_row("Transitions labeled 1", f"{len(self.rewards_pos)}")
        print(table)

    def compute_monte_carlo_values(self, discount=0.99):
        """ computes an estimate of the value of a state local to the episode"""
        buffer_values = []
        for epi in self.episodes:
            transitions = list(self.transitions[epi])
            values = []
            value = 0
            for trsn in reversed(transitions):
                values.append(value)
                value = value * discount + trsn['reward']
            values = reversed(values)
            buffer_values += values

        return buffer_values


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

    def load(self, filename, mode='r'):
        super().load(filename, mode)
        self.classes = np.zeros(self.steps, dtype=np.int64)
        initials = self.episodes[:self.num_episodes]
        initials_mask = np.zeros(self.steps, dtype=np.uint8)
        initials_mask[initials] = 1
        initials_mask = initials_mask.astype(np.bool_)
        reward_pos = self.reward[:self.steps] > 0.0

        """
        When we hit a reward, flag the states that lead up to it as rewarding
        but not if they are from a different trajectory!
        """
        counter = 0
        for i in track(reversed(range(self.steps)), total=self.steps):
            r = reward_pos[i]
            initial = initials_mask[i]
            if r:
                counter = self.reward_causality_distance
            if counter > 0:
                reward_pos[i] = True
                if initial:
                    counter = 0
                else:
                    counter -= 1

        self.rewards_pos, = np.where(reward_pos)
        self.rewards_neg, = np.where(reward_pos == False)
        self.classes = reward_pos.astype(dtype=np.int64)

    def make_stat_table(self):
        table = super().make_stat_table()
        table.add_row("Labeled reward_pos", f"{np.count_nonzero(self.classes)}")
        return table

    def __len__(self):
        return self.n_gram_len(gram_len=1)

    def __getitem__(self, item):
        image = self.transforms(self.raw[item])
        label = self.classes[item]
        return image, label


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
        self.seed = seed
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.val_sampler = None
        self.train_sampler = None

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
            n_batches=1024,
            alpha=0.5,
            kind='fixed'
        )
        self.val_sampler = SamplerFactory().get(
            class_idxs=[train_rewards_nonpos, train_rewards_pos],
            batch_size=self.batch_size,
            n_batches=1024,
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