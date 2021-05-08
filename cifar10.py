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


class ENetCIFAR10(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = torch.hub.load('rwightman/gen-efficientnet-pytorch', 'efficientnet_b0', pretrained=True)
        self.classifier = nn.Sequential(nn.Linear(1000, 10, bias=True), nn.Softmax())
        self.batch_size = config.batch_size
        self.learning_rate = config.lr
        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
            return torch.optim.Adam(self.classifier.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        x, labels = batch
        with torch.no_grad():
            hidden = self.model(x)
        predictions = self.classifier(hidden)
        loss = cross_entropy(predictions, labels)

        self.train_acc(predictions, labels)
        self.log('train/loss', loss, on_step=False, on_epoch=True)
        self.log('train/acc', self.train_acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, labels = batch
        y = self.classifier(self.model(x))
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
        y = self.classifier(self.model(x))
        predictions = torch.argmax(y, dim=1)
        self.test_acc(predictions, labels)
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

        if self.config.dev:
            self.trainset = Subset(self.trainset, list(range(1000)))
            self.testset = Subset(self.testset, list(range(1000)))

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.config.batch_size, shuffle=True, pin_memory=True,
                          num_workers=config.workers)

    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=100, shuffle=False, pin_memory=True, num_workers=config.workers)

    def val_dataloader(self):
        return DataLoader(self.testset, batch_size=100, shuffle=False, pin_memory=True, num_workers=config.workers)


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('--epochs', type=int, default=100)
    args.add_argument('--batch_size', type=int, default=128)
    args.add_argument('--workers', type=int, default=2)
    args.add_argument('--lars', action='store_true', default=False)
    args.add_argument('--dev', action='store_true', default=False)
    args.add_argument('--device', type=str)
    args.add_argument('--seed', type=int)
    args.add_argument('--lr', type=float, default=1e-4)
    config = args.parse_args()

    model = ENetCIFAR10(config)

    wandb_logger = WandbLogger(project="efficientnet-cifar10", config=config)
    trainer = pl.Trainer(gpus=1, check_val_every_n_epoch=1, logger=wandb_logger)
    dm = CIFAR10DataModule(config)
    trainer.fit(model=model, datamodule=dm)
    trainer.test(model=model, datamodule=dm)
