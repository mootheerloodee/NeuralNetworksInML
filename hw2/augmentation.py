import torch
import torchvision.transforms as transforms
import numpy as np
from torchvision.transforms.functional import rotate, affine


class RandomRotation:
    def __init__(self, min_angle=-15, max_angle=15):
        self.min_angle = min_angle
        self.max_angle = max_angle

    def __call__(self, image):
        random_angle = np.random.uniform(self.min_angle, self.max_angle)
        rotated_image = rotate(image, random_angle)
        return rotated_image

    def __repr__(self):
        return f"RandomRotation(degrees={self.degrees})"


class RandomShift:
    def __init__(self, max_shift_percent=0.1):
        self.max_shift_percent = max_shift_percent

    def __call__(self, image):
        channels, height, width = image.shape
        max_shift_x = int(width * self.max_shift_percent)
        max_shift_y = int(height * self.max_shift_percent)
        shift_x = np.random.randint(-max_shift_x, max_shift_x + 1)
        shift_y = np.random.randint(-max_shift_y, max_shift_y + 1)
        shifted_image = affine(
            img=image,
            angle=0,
            translate=(shift_x, shift_y),
            scale=1.0,
            shear=0
        )
        return shifted_image

    def __repr__(self):
        return f"RandomShift(max_shift={self.max_shift})"


class Noise:
    def __init__(self, mean=0.0, stddev=0.05):
        self.mean = mean
        self.stddev = stddev

    def __call__(self, tensor):
        noise = torch.zeros_like(tensor).normal_(self.mean, self.stddev)
        return tensor.add_(noise)

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean},stddev={self.stddev})"


def get_train_transform(augmentations=None):
    transform_list = [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]
    augmentation_dict = {
        'rotation': RandomRotation(),
        'shift': RandomShift(),
        'noise': Noise()
    }
    if augmentations is None:
        augmentations = list(augmentation_dict.keys())
    for aug_name in augmentations:
        if aug_name in augmentation_dict:
            transform_list.insert(1, augmentation_dict[aug_name])
    return transforms.Compose(transform_list)


def get_test_transform():
    transform_list = [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]
    final_transform = transforms.Compose(transform_list)
    return final_transform