from typing import Optional, Sequence, Tuple, Union

import torch
import pytorch_lightning as pl
from pytorch_lightning import Callback, LightningModule, Trainer
from torch import device, Tensor
from torch.nn import functional as F
from torch.optim import Optimizer
from torchmetrics.functional import accuracy
from torchmetrics import ConfusionMatrix
import wandb


class SSLOnlineEvaluator(Callback):  # pragma: no cover
    """
    Attaches a MLP for fine-tuning using the standard self-supervised protocol.

    Example::

        # your model must have 2 attributes
        model = Model()
        model.z_dim = ... # the representation dim
        model.num_classes = ... # the num of classes in the model

        online_eval = SSLOnlineEvaluator(
            z_dim=model.z_dim,
            num_classes=model.num_classes,
            dataset='imagenet'
        )

    """

    def __init__(
        self,
        dataset: str,
        drop_p: float = 0.2,
        hidden_dim: Optional[int] = None,
        z_dim: int = None,
        num_classes: int = None,
    ):
        """
        Args:
            dataset: if stl10, need to get the labeled batch
            drop_p: Dropout probability
            hidden_dim: Hidden dimension for the fine-tune MLP
            z_dim: Representation dimension
            num_classes: Number of classes
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.drop_p = drop_p
        self.optimizer: Optimizer

        self.z_dim = z_dim
        self.num_classes = num_classes
        self.dataset = dataset

        self.confusion_matrix = None

        self.time_to_sample = True
        self.images = None
        self.mlp_preds = None
        self.labels = None
        self.batch_size = None
        self.epoch = 0

    def on_pretrain_routine_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        from pl_bolts.models.self_supervised.evaluator import SSLEvaluator

        pl_module.non_linear_evaluator = SSLEvaluator(
            n_input=self.z_dim,
            n_classes=self.num_classes,
            p=self.drop_p,
            n_hidden=self.hidden_dim,
        ).to(pl_module.device)

        self.confusion_matrix = ConfusionMatrix(self.num_classes).to(pl_module.device)

        self.optimizer = torch.optim.Adam(pl_module.non_linear_evaluator.parameters(), lr=1e-4)

    def get_representations(self, pl_module: LightningModule, x: Tensor) -> Tensor:
        representations = pl_module(x)
        representations = representations.reshape(representations.size(0), -1)
        return representations

    def to_device(self, batch: Sequence, device: Union[str, device]) -> Tuple[Tensor, Tensor]:
        # get the labeled batch
        if self.dataset == 'stl10':
            labeled_batch = batch[1]
            batch = labeled_batch

        inputs, y = batch

        # last input is for online eval
        x = inputs[-1]
        x = x.to(device)
        y = y.to(device)

        return x, y

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        x, y = self.to_device(batch, pl_module.device)

        with torch.no_grad():
            representations = self.get_representations(pl_module, x)

        representations = representations.detach()

        # forward pass
        mlp_preds = pl_module.non_linear_evaluator(representations)  # type: ignore[operator]
        mlp_loss = F.cross_entropy(mlp_preds, y)

        # update finetune weights
        mlp_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        # log metrics
        train_acc = accuracy(mlp_preds, y)
        pl_module.log('online_train_acc', train_acc, on_step=True, on_epoch=False)
        pl_module.log('online_train_loss', mlp_loss, on_step=True, on_epoch=False)


    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        x, y = self.to_device(batch, pl_module.device)

        with torch.no_grad():
            representations = self.get_representations(pl_module, x)

        representations = representations.detach()

        # forward pass
        mlp_preds = pl_module.non_linear_evaluator(representations)  # type: ignore[operator]
        mlp_loss = F.cross_entropy(mlp_preds, y)

        # log metrics
        val_acc = accuracy(mlp_preds, y)
        pl_module.log('online_val_acc', val_acc, on_step=False, on_epoch=True, sync_dist=True)
        pl_module.log('online_val_loss', mlp_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.confusion_matrix(mlp_preds, y)

        if self.time_to_sample:
            N, C, H, W = batch[0][2].shape
            num = min(N, 16)
            self.images = batch[0][2][0:num]
            self.mlp_preds = torch.argmax(mlp_preds[0:num], dim=1)
            self.labels = y[0:num]
            self.time_to_sample = False

    def on_validation_epoch_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        confusion = self.confusion_matrix.compute()
        confusion = confusion.type(torch.int)
        confusion_table = wandb.Table(data=confusion.tolist(), columns=['no_reward', 'reward'])
        pl_module.logger.experiment.log({'confusion': confusion_table})
        table = []
        for image, pred, label in zip(self.images.unbind(0), self.mlp_preds.unbind(0), self.labels.unbind(0)):
            table.append((self.epoch, label.cpu(), pred.cpu(), wandb.Image(image[0:3].detach().cpu())))
        table = wandb.Table(data=table, columns=['epoch', 'label', 'pred', 'image'])
        pl_module.logger.experiment.log({'validation_sample': table})
        self.epoch += 1
        self.confusion_matrix = ConfusionMatrix(self.num_classes).to(pl_module.device)
        self.time_to_sample = True