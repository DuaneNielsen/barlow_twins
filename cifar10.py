import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader
import torchvision
from torchvision import transforms
from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import confusion_matrix
from torch.nn.functional import cross_entropy
from typing import Optional
import wandb
from pytorch_lightning.loggers import WandbLogger
from torch.nn.functional import upsample_bilinear


class AtariVision(nn.Module):
    def __init__(self, feature_size=512):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.SELU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.SELU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv2[0].weight.data.mul_(2 ** -0.5)
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.SELU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv3[0].weight.data.mul_(3 ** -0.5)
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.SELU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv4[0].weight.data.mul_(4 ** -0.5)
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.SELU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv5[0].weight.data.mul_(5 ** -0.5)
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, feature_size, kernel_size=3, stride=1, padding=1, bias=False),
            nn.SELU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv6[0].weight.data.mul_(6 ** -0.5)

        self.bias = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(5)])

        self.classifier = nn.Linear(feature_size, 10, bias=True)

    def forward(self, state):
        state = upsample_bilinear(state, (88, 88))
        l1 = self.conv1(state)
        l2 = self.conv2(l1) + self.bias[0]
        l3 = self.conv3(l2) + self.bias[1]
        l4 = self.conv4(l3) + self.bias[2]
        l5 = self.conv5(l4) + self.bias[3]
        l6 = self.conv6(l5) + self.bias[4]
        return torch.softmax(self.classifier(l6.flatten(start_dim=1)), dim=1)


class ENetCIFAR10(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if 'atari' in config.model:
            self.model = AtariVision(feature_size=512)
        else:
            self.model = torch.hub.load('rwightman/gen-efficientnet-pytorch', config.model, pretrained=True)
        class_in_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(nn.Linear(class_in_features, 10, bias=True), nn.Softmax(dim=1))
        self.batch_size = config.batch_size
        self.learning_rate = config.lr
        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        # get all the parameters that are not the classifier
        not_classifer = [param for name, param in filter(lambda key: 'classifier' not in key[0], self.model.named_parameters())]
        return torch.optim.Adam([
            {"params": self.model.classifier.parameters(), "lr": self.learning_rate},
            {"params": not_classifer, "lr": self.learning_rate/10.0}
        ])

    def training_step(self, batch, batch_idx):
        x, labels = batch
        predictions = self.model(x)
        loss = cross_entropy(predictions, labels)

        self.train_acc(predictions, labels)
        self.log('train/loss', loss, on_step=False, on_epoch=True)
        self.log('train/acc', self.train_acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, labels = batch
        y = self.model(x)
        predictions = torch.argmax(y, dim=1)
        self.valid_acc(predictions, labels)
        self.log('valid/acc', self.valid_acc, on_step=False, on_epoch=True)
        return {'predictions': predictions, 'labels': labels}

    def validation_epoch_end(self, outputs):
        preds = torch.cat([tmp['predictions'] for tmp in outputs])
        targets = torch.cat([tmp['labels'] for tmp in outputs])
        confusion = confusion_matrix(preds, targets, num_classes=10)
        confusion_table = wandb.Table(data=confusion.tolist(), columns=['plane', 'car', 'bird', 'cat', 'deer',
                        'dog', 'frog', 'horse', 'ship', 'truck'])
        self.logger.experiment.log({'confusion': confusion_table})

    def test_step(self, batch, batch_idx):
        x, labels = batch
        y = self.model(x)
        predictions = torch.argmax(y, dim=1)
        self.test_acc(predictions, labels)
        self.log('test/acc', self.test_acc, on_step=False, on_epoch=True)
        return {'predictions': predictions, 'labels': labels}


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer',
                        'dog', 'frog', 'horse', 'ship', 'truck')

    def setup(self, stage: Optional[str] = None):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.trainset = torchvision.datasets.CIFAR10(
            root='~/data', train=True, download=True, transform=transform_train)
        self.testset = torchvision.datasets.CIFAR10(
            root='~/data', train=False, download=True, transform=transform_test)

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.config.batch_size, shuffle=True, pin_memory=True,
                          num_workers=config.workers)

    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=100, shuffle=False, pin_memory=True, num_workers=config.workers)

    def val_dataloader(self):
        return DataLoader(self.testset, batch_size=100, shuffle=False, pin_memory=True, num_workers=config.workers)


if __name__ == '__main__':
    args = ArgumentParser()
    pl.Trainer.add_argparse_args(args)
    args.add_argument('--model', type=str, default='efficientnet_b0')
    args.add_argument('--batch_size', type=int, default=128)
    args.add_argument('--workers', type=int, default=2)
    args.add_argument('--lars', action='store_true', default=False)
    args.add_argument('--seed', type=int)
    args.add_argument('--lr', type=float, default=1e-4)
    config = args.parse_args()

    model = ENetCIFAR10(config)

    wandb_logger = WandbLogger(project="efficientnet-cifar10", config=config)
    trainer = pl.Trainer.from_argparse_args(config)
    trainer.logger = wandb_logger
    dm = CIFAR10DataModule(config)
    trainer.fit(model=model, datamodule=dm)
    trainer.test(model=model, datamodule=dm)
