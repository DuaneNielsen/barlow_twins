from typing import Optional, Tuple, Any

import torch
from pytorch_lightning import Callback, LightningModule, Trainer
import pytorch_lightning as pl

from pl_bolts.utils import _TORCHVISION_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg

if _TORCHVISION_AVAILABLE:
    import torchvision
else:  # pragma: no cover
    warn_missing_pkg("torchvision")
import cv2


class CV2ModelImageSampler(Callback):
    """
    Generates images and logs to tensorboard.
    Your model must implement the ``forward`` function for generation
    Requirements::
        # model must have img_dim arg
        model.img_dim = (1, 28, 28)
        # model forward must work for sampling
        z = torch.rand(batch_size, latent_dim)
        img_samples = your_model(z)
    Example::
        from pl_bolts.callbacks import TensorboardGenerativeModelImageSampler
        trainer = Trainer(callbacks=[TensorboardGenerativeModelImageSampler()])
    """

    def __init__(
        self,
        num_samples: int = 3,
        nrow: int = 8,
        padding: int = 2,
        normalize: bool = False,
        norm_range: Optional[Tuple[int, int]] = None,
        scale_each: bool = False,
        pad_value: int = 0,
    ) -> None:
        """
        Args:
            num_samples: Number of images displayed in the grid. Default: ``3``.
            nrow: Number of images displayed in each row of the grid.
                The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
            padding: Amount of padding. Default: ``2``.
            normalize: If ``True``, shift the image to the range (0, 1),
                by the min and max values specified by :attr:`range`. Default: ``False``.
            norm_range: Tuple (min, max) where min and max are numbers,
                then these numbers are used to normalize the image. By default, min and max
                are computed from the tensor.
            scale_each: If ``True``, scale each image in the batch of
                images separately rather than the (min, max) over all images. Default: ``False``.
            pad_value: Value for the padded pixels. Default: ``0``.
        """
        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError("You want to use `torchvision` which is not installed yet.")

        super().__init__()
        self.num_samples = num_samples
        self.nrow = nrow
        self.padding = padding
        self.normalize = normalize
        self.norm_range = norm_range
        self.scale_each = scale_each
        self.pad_value = pad_value

    def on_train_batch_start(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:


        grid0 = torchvision.utils.make_grid(
            tensor=batch[0][0],
            nrow=self.nrow,
            padding=self.padding,
            normalize=self.normalize,
            range=self.norm_range,
            scale_each=self.scale_each,
            pad_value=self.pad_value,
        )

        grid1 = torchvision.utils.make_grid(
            tensor=batch[0][1],
            nrow=self.nrow,
            padding=self.padding,
            normalize=self.normalize,
            range=self.norm_range,
            scale_each=self.scale_each,
            pad_value=self.pad_value,
        )

        grid2 = torchvision.utils.make_grid(
            tensor=batch[0][2],
            nrow=self.nrow,
            padding=self.padding,
            normalize=self.normalize,
            range=self.norm_range,
            scale_each=self.scale_each,
            pad_value=self.pad_value,
        )

        grid = torch.cat((grid0, grid1, grid2), dim=1)

        str_title = f"{pl_module.__class__.__name__}_images"
        screen = grid.permute(1, 2, 0).cpu().numpy()
        cv2.imshow(str_title, screen[:, :, 0:3])
        cv2.waitKey(0)
        cv2.imshow(str_title, screen[:, :, 3:6])
        cv2.waitKey(0)