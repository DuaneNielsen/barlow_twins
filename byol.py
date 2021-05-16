import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pl_bolts.callbacks.ssl_online import SSLOnlineEvaluator
from pl_bolts.datamodules import CIFAR10DataModule, ImagenetDataModule, STL10DataModule
from pl_bolts.models.self_supervised.simclr import SimCLREvalDataTransform, SimCLRTrainDataTransform
from pytorch_lightning.loggers.wandb import WandbLogger

from argparse import ArgumentParser
from copy import deepcopy
from typing import Any

import torch
import torch.nn as nn
import torch.hub
from torch.nn import functional as F
from torch.optim import Adam
from torchvision.transforms import Compose, Resize
from pathlib import Path

from pl_bolts.callbacks.byol_updates import BYOLMAWeightUpdate
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pl_bolts.utils.self_supervised import torchvision_ssl_encoder
from rl import OnDiskReplayBuffer
from torch.utils.data import random_split
import random
import itertools
import torchvision.transforms
import numpy as np
from math import ceil, floor
from rich.progress import track
import os
import tables as tb


class FixNineYearOldCodersJunk(nn.Module):
    def __init__(self, dunce_net):
        super().__init__()
        self.dunce_net = dunce_net

    def forward(self, x):
        return self.dunce_net(x)[0]


class MLP(nn.Module):

    def __init__(self, input_dim=2048, hidden_size=4096, output_dim=256):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_size, bias=False),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, output_dim, bias=True),
        )

    def forward(self, x):
        x = self.model(x)
        return x


class SiameseArm(nn.Module):

    def __init__(self, encoder=None):
        super().__init__()

        self.encoder = encoder
        # Projector
        self.projector = MLP()
        # Predictor
        self.predictor = MLP(input_dim=256)

    def forward(self, x):
        y = self.encoder(x)
        z = self.projector(y)
        h = self.predictor(z)
        return y, z, h


class AtariVision(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.SELU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.SELU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv2[0].weight.data.mul_(2 ** -0.5)
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.SELU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv3[0].weight.data.mul_(3 ** -0.5)
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.SELU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv4[0].weight.data.mul_(4 ** -0.5)
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.SELU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv5[0].weight.data.mul_(5 ** -0.5)
        self.conv6 = nn.Sequential(
            nn.Conv2d(1024, 2048, kernel_size=3, stride=1, padding=1, bias=False),
            nn.SELU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv6[0].weight.data.mul_(6 ** -0.5)

        self.bias = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(5)])

    def forward(self, state):
        l1 = self.conv_stem(state)
        l2 = self.conv2(l1) + self.bias[0]
        l3 = self.conv3(l2) + self.bias[1]
        l4 = self.conv4(l3) + self.bias[2]
        l5 = self.conv5(l4) + self.bias[3]
        l6 = self.conv6(l5) + self.bias[4]
        return l6.flatten(start_dim=1)


class BYOL(pl.LightningModule):
    """
    PyTorch Lightning implementation of `Bootstrap Your Own Latent (BYOL)
    <https://arxiv.org/pdf/2006.07733.pdf>`_

    Paper authors: Jean-Bastien Grill, Florian Strub, Florent Altché, Corentin Tallec, Pierre H. Richemond, \
    Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Daniel Guo, Mohammad Gheshlaghi Azar, \
    Bilal Piot, Koray Kavukcuoglu, Rémi Munos, Michal Valko.

    Model implemented by:
        - `Annika Brundyn <https://github.com/annikabrundyn>`_

    .. warning:: Work in progress. This implementation is still being verified.

    TODOs:
        - verify on CIFAR-10
        - verify on STL-10
        - pre-train on imagenet

    Example::

        model = BYOL(num_classes=10)

        dm = CIFAR10DataModule(num_workers=0)
        dm.train_transforms = SimCLRTrainDataTransform(32)
        dm.val_transforms = SimCLREvalDataTransform(32)

        trainer = pl.Trainer()
        trainer.fit(model, datamodule=dm)

    Train::

        trainer = Trainer()
        trainer.fit(model)

    CLI command::

        # cifar10
        python byol_module.py --gpus 1

        # imagenet
        python byol_module.py
            --gpus 8
            --dataset imagenet2012
            --data_dir /path/to/imagenet/
            --meta_dir /path/to/folder/with/meta.bin/
            --batch_size 32
    """

    def __init__(
        self,
        num_classes,
        learning_rate: float = 0.2,
        weight_decay: float = 1.5e-6,
        input_height: int = 32,
        batch_size: int = 32,
        num_workers: int = 0,
        warmup_epochs: int = 10,
        max_epochs: int = 1000,
        encoder: nn.Module = None,
        **kwargs
    ):
        """
        Args:
            datamodule: The datamodule
            learning_rate: the learning rate
            weight_decay: optimizer weight decay
            input_height: image input height
            batch_size: the batch size
            num_workers: number of workers
            warmup_epochs: num of epochs for scheduler warm up
            max_epochs: max epochs for scheduler
        """
        super().__init__()
        self.save_hyperparameters()

        self.online_network = SiameseArm(encoder=encoder)
        self.target_network = deepcopy(self.online_network)
        self.weight_callback = BYOLMAWeightUpdate()

    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        # Add callback for user automatically since it's key to BYOL weight update
        self.weight_callback.on_train_batch_end(self.trainer, self, outputs, batch, batch_idx, dataloader_idx)

    def forward(self, x):
        y, _, _ = self.online_network(x)
        return y

    def shared_step(self, batch, batch_idx):
        imgs, y = batch
        img_1, img_2 = imgs[:2]

        # Image 1 to image 2 loss
        y1, z1, h1 = self.online_network(img_1)
        with torch.no_grad():
            y2, z2, h2 = self.target_network(img_2)
        loss_a = -2 * F.cosine_similarity(h1, z2).mean()

        # Image 2 to image 1 loss
        y1, z1, h1 = self.online_network(img_2)
        with torch.no_grad():
            y2, z2, h2 = self.target_network(img_1)
        # L2 normalize
        loss_b = -2 * F.cosine_similarity(h1, z2).mean()

        # Final loss
        total_loss = loss_a + loss_b

        return loss_a, loss_b, total_loss

    def training_step(self, batch, batch_idx):
        loss_a, loss_b, total_loss = self.shared_step(batch, batch_idx)

        # log results
        self.log_dict({'1_2_loss': loss_a, '2_1_loss': loss_b, 'train_loss': total_loss})

        return total_loss

    def validation_step(self, batch, batch_idx):
        loss_a, loss_b, total_loss = self.shared_step(batch, batch_idx)

        # log results
        self.log_dict({'1_2_loss': loss_a, '2_1_loss': loss_b, 'val_loss': total_loss})

        return total_loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=self.hparams.warmup_epochs, max_epochs=self.hparams.max_epochs
        )
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--online_ft', action='store_true', help='run online finetuner')
        parser.add_argument('--dataset', type=str, default='cifar10',
                            choices=['cifar10', 'cifar10_upsampled', 'imagenet2012', 'stl10', 'atari'])

        (args, _) = parser.parse_known_args()

        # Data
        parser.add_argument('--data_dir', type=str, default='.')
        parser.add_argument('--num_workers', default=8, type=int)
        parser.add_argument('--filename', type=str)


        # optim
        parser.add_argument('--batch_size', type=int, default=256)
        parser.add_argument('--learning_rate', type=float, default=1e-3)
        parser.add_argument('--weight_decay', type=float, default=1.5e-6)
        parser.add_argument('--warmup_epochs', type=float, default=10)

        parser.add_argument('--vision_model', type=str, default='resnet-50')
        parser.add_argument('--vision_model_from_pytorch_hub', type=str, nargs=2, default=None)
        parser.add_argument('--pretrained', action='store_true', default=False)

        # Model
        parser.add_argument('--meta_dir', default='.', type=str, help='path to meta.bin for imagenet')

        return parser


if __name__ == '__main__':

    seed_everything(1234)

    parser = ArgumentParser()

    # trainer args
    parser = pl.Trainer.add_argparse_args(parser)

    # model args
    parser = BYOL.add_model_specific_args(parser)
    parser.add_argument('--input_size', type=int, default=32)
    args = parser.parse_args()

    # pick data
    dm = None

    # init default datamodule
    if args.dataset == 'cifar10':
        dm = CIFAR10DataModule.from_argparse_args(args)
        dm.train_transforms = SimCLRTrainDataTransform(args.input_size)
        dm.val_transforms = SimCLREvalDataTransform(args.input_size)
        args.num_classes = dm.num_classes

    elif args.dataset == 'stl10':
        dm = STL10DataModule.from_argparse_args(args)
        dm.train_dataloader = dm.train_dataloader_mixed
        dm.val_dataloader = dm.val_dataloader_mixed

        (c, h, w) = dm.size()
        dm.train_transforms = SimCLRTrainDataTransform(h)
        dm.val_transforms = SimCLREvalDataTransform(h)
        args.num_classes = dm.num_classes

    elif args.dataset == 'imagenet2012':
        dm = ImagenetDataModule.from_argparse_args(args, image_size=196)
        (c, h, w) = dm.size()
        dm.train_transforms = SimCLRTrainDataTransform(h)
        dm.val_transforms = SimCLREvalDataTransform(h)
        args.num_classes = dm.num_classes

    elif args.dataset == 'atari':

        class NextStateReward(OnDiskReplayBuffer, torch.utils.data.Dataset):
            def __init__(self, transforms):
                super().__init__()
                self.transforms = transforms

            def make_transition(self, trans):
                s_p = self.states[trans['next_state']]
                s_p = self.transforms(s_p)
                r = trans['reward']
                r = 0 if r == 0 else 1
                return s_p, r

            @staticmethod
            def load_splits(filename):
                """
                returns non overlapping random splits of the dataset
                filename: the filename
                :param lengths: list of lengths of each split, [10, 4, 3] 10 is length of split 0, etc...
                :return: None
                """
                assert os.path.isfile(filename), f"{filename} does not exist"
                fileh = tb.open_file(filename, mode='r')
                length = len(fileh.root.replay.Transitions)
                test_len = ceil(length/2)
                val_len = floor(length/2)
                indices = np.random.permutation(length)

                train_transforms = torchvision.transforms.Compose([
                    torchvision.transforms.ToPILImage(),
                    torchvision.transforms.Resize(size=args.input_size),
                    SimCLRTrainDataTransform(h)
                ])
                val_transforms = torchvision.transforms.Compose([
                    torchvision.transforms.ToPILImage(),
                    torchvision.transforms.Resize(size=args.input_size),
                    SimCLREvalDataTransform(h)
                ])

                train_set = NextStateReward(train_transforms)
                train_set.fileh = fileh
                train_set.split = indices[0:test_len]

                val_set = NextStateReward(val_transforms)
                val_set.fileh = fileh

                # create a balanced set of reward positive and no reward
                val_set.split = indices[test_len:]
                reward_pos, reward_zero = [], []
                for i in val_set.split:
                    reward = val_set.transitions[i]['reward']
                    if reward == 0:
                        reward_zero += [i]
                    if reward > 0:
                        reward_pos += [i]
                reward_zero = reward_zero[:len(reward_pos)]
                val_set.split = list(itertools.chain(*zip(reward_pos, reward_zero)))

                return train_set, val_set

        (h, w, c) = (219, 160, 3)

        train_set, val_set = NextStateReward.load_splits(args.filename)

        dm = pl.LightningDataModule.from_datasets(
            train_dataset=train_set,
            val_dataset=val_set,
            test_dataset=None,
            batch_size=args.batch_size,
            num_workers=0
        )

        dm.num_classes = 2
        args.num_classes = 2

    encoder = None
    if args.vision_model == 'atari':
        encoder = AtariVision()
    elif args.vision_model == 'resnet-50':
        encoder = torchvision_ssl_encoder('resnet50')
        encoder = FixNineYearOldCodersJunk(encoder)
    else:
        encoder = torch.hub.load(repo_or_dir='rwightman/gen-efficientnet-pytorch', model=args.vision_model, pretrained=args.pretrained)
        if encoder.classifier.in_features == 2048:
            encoder.classifier = nn.Identity()
        else:
            encoder.classifier = nn.Linear(encoder.classifier.in_features, 2048, bias=False)

    model = BYOL(encoder=encoder, **args.__dict__)

    save_dir = f'./{args.vision_model}-{args.input_size}'
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    wandb_logger = WandbLogger(project="byol",  save_dir=save_dir, log_model=False)
    # finetune in real-time
    online_eval = SSLOnlineEvaluator(dataset=args.dataset, z_dim=2048, num_classes=dm.num_classes)
    # DEFAULTS used by the Trainer
    trainer = pl.Trainer.from_argparse_args(args, max_steps=300000, callbacks=[online_eval])
    trainer.logger = wandb_logger
    trainer.fit(model, datamodule=dm)