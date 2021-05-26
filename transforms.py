import numpy as np

from pl_bolts.utils import _OPENCV_AVAILABLE, _TORCHVISION_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg

if _TORCHVISION_AVAILABLE:
    from torchvision import transforms as transforms
else:  # pragma: no cover
    warn_missing_pkg('torchvision')

if _OPENCV_AVAILABLE:
    import cv2
else:  # pragma: no cover
    warn_missing_pkg('cv2', pypi_name='opencv-python')

from torchvision.transforms.functional import to_tensor, to_pil_image
import torch


class AtariSimCLRTrainDataTransform(object):
    """
    Transforms for SimCLR

    Transform::

        RandomResizedCrop(size=self.input_height)
        RandomHorizontalFlip()
        RandomApply([color_jitter], p=0.8)
        RandomGrayscale(p=0.2)
        GaussianBlur(kernel_size=int(0.1 * self.input_height))
        transforms.ToTensor()

    Example::

        from pl_bolts.models.self_supervised.simclr.transforms import SimCLRTrainDataTransform

        transform = SimCLRTrainDataTransform(input_height=32)
        x = sample()
        (xi, xj) = transform(x)
    """

    def __init__(
        self, input_height: int = 224, gaussian_blur: bool = True, jitter_strength: float = 1., normalize=None
    ) -> None:

        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError('You want to use `transforms` from `torchvision` which is not installed yet.')

        self.jitter_strength = jitter_strength
        self.input_height = input_height
        self.gaussian_blur = gaussian_blur
        self.normalize = normalize

        self.color_jitter = transforms.ColorJitter(
            0.8 * self.jitter_strength, 0.8 * self.jitter_strength, 0.8 * self.jitter_strength,
            0.2 * self.jitter_strength
        )

        kernel_size = int(0.1 * input_height)
        if kernel_size % 2 == 0:
            kernel_size += 1

        self.space_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=(input_height, input_height)),
            transforms.RandomResizedCrop(size=input_height),
            transforms.RandomHorizontalFlip(p=0.5),
            GaussianBlur(kernel_size=kernel_size, p=0.5)
        ])

        if normalize is None:
            self.final_transform = [transforms.ToTensor()]
        else:
            self.final_transform = [transforms.ToTensor(), normalize]

        self.color_transforms = transforms.Compose([
            transforms.RandomApply([self.color_jitter], p=0.8),
        ])

        # add online train transform of the size of global view
        self.online_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomResizedCrop(self.input_height),
            transforms.RandomHorizontalFlip()
        ])

    def __call__(self, sample):

        xi6 = self.space_transforms(sample)
        xj6 = self.space_transforms(sample)

        xi = self.color_transforms(xi6[0:3])
        xi_g = self.color_transforms(xi6[3:6])

        xj = self.color_transforms(xj6[0:3])
        xj_g = self.color_transforms(xj6[3:6])

        xi = torch.cat((xi, xi_g), dim=0)
        xj = torch.cat((xj, xj_g), dim=0)

        return xi, xj, self.online_transform(sample)


class AtariSimCLREvalDataTransform(AtariSimCLRTrainDataTransform):
    """
    Transforms for SimCLR

    Transform::

        Resize(input_height + 10, interpolation=3)
        transforms.CenterCrop(input_height),
        transforms.ToTensor()

    Example::

        from pl_bolts.models.self_supervised.simclr.transforms import SimCLREvalDataTransform

        transform = SimCLREvalDataTransform(input_height=32)
        x = sample()
        (xi, xj) = transform(x)
    """

    def __init__(
        self, input_height: int = 224, gaussian_blur: bool = True, jitter_strength: float = 1., normalize=None
    ):
        super().__init__(
            normalize=normalize,
            input_height=input_height,
            gaussian_blur=gaussian_blur,
            jitter_strength=jitter_strength
        )

        # replace online transform with eval time transform
        self.online_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(int(self.input_height + 0.1 * self.input_height)),
            transforms.CenterCrop(self.input_height),
        ])


class SimCLRFinetuneTransform(object):

    def __init__(
        self,
        input_height: int = 224,
        jitter_strength: float = 1.,
        normalize=None,
        eval_transform: bool = False
    ) -> None:

        self.jitter_strength = jitter_strength
        self.input_height = input_height
        self.normalize = normalize

        self.color_jitter = transforms.ColorJitter(
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.2 * self.jitter_strength,
        )

        if not eval_transform:
            data_transforms = [
                transforms.RandomResizedCrop(size=self.input_height),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([self.color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2)
            ]
        else:
            data_transforms = [
                transforms.Resize(int(self.input_height + 0.1 * self.input_height)),
                transforms.CenterCrop(self.input_height)
            ]

        if normalize is None:
            final_transform = transforms.ToTensor()
        else:
            final_transform = transforms.Compose([transforms.ToTensor(), normalize])

        data_transforms.append(final_transform)
        self.transform = transforms.Compose(data_transforms)

    def __call__(self, sample):
        return self.transform(sample)


class GaussianBlur(object):
    # Implements Gaussian blur as described in the SimCLR paper
    def __init__(self, kernel_size, p=0.5, min=0.1, max=2.0):
        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError('You want to use `GaussianBlur` from `cv2` which is not installed yet.')

        self.min = min
        self.max = max

        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size
        self.p = p

    def __call__(self, sample):
        _type = type(sample)
        sample = np.array(sample)

        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < self.p:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

        if _type == torch.Tensor:
            return torch.from_numpy(sample)

        return sample
