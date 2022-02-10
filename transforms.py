import numpy as np
from torchvision.transforms import transforms


class Crop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image):

        h, w = image.size
        diam = min(h, w)
        a = diam // np.sqrt(2)
        x = (h - a) // 2
        rect = image.crop((x, x, x + a, x + a))  # image[x:x + a, x:x + a]
        return rect


def get_transforms():
    base_transform = [transforms.Resize((224, 224)), transforms.ToTensor(),
                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    crop_size = 150
    train_transforms = [
        transforms.Compose([
                               transforms.RandomHorizontalFlip(),
                               transforms.RandomVerticalFlip(),
                               transforms.RandomRotation(180),
                           ] + base_transform),
        transforms.Compose([
                               transforms.RandomHorizontalFlip(),
                               transforms.RandomVerticalFlip(),
                               transforms.RandomRotation(180),
                               Crop(crop_size),
                           ] + base_transform),
        transforms.Compose([
                               transforms.RandomHorizontalFlip(p=1),
                               Crop(crop_size),
                           ] + base_transform),
        transforms.Compose([
                               transforms.RandomVerticalFlip(p=1),
                               Crop(crop_size),
                           ] + base_transform),
        transforms.Compose([
                               transforms.RandomRotation(60),
                               Crop(crop_size),
                           ] + base_transform),
        transforms.Compose([
                               transforms.RandomRotation(60),
                               transforms.RandomHorizontalFlip(),
                               transforms.RandomVerticalFlip(),
                               Crop(crop_size),
                           ] + base_transform),
        transforms.Compose([
                               transforms.ColorJitter(hue=(-0.5, 0.5)),
                               Crop(crop_size),
                           ] + base_transform),
        transforms.Compose([
                               Crop(crop_size),
                           ] + base_transform),
        transforms.Compose([
                               transforms.ColorJitter(hue=(-0.5, 0.5)),
                               transforms.RandomRotation(60),
                               transforms.RandomVerticalFlip(),
                               transforms.RandomHorizontalFlip(),
                           ] + base_transform),
        transforms.Compose([
                               transforms.ColorJitter(hue=(-0.5, 0.5)),
                               transforms.RandomRotation(180),
                               transforms.CenterCrop(crop_size),
                               transforms.RandomVerticalFlip(),
                               transforms.RandomHorizontalFlip(),
                           ] + base_transform),
        transforms.Compose([
                               transforms.CenterCrop(crop_size),
                           ] + base_transform),
    ]

    val_transforms = [
        transforms.Compose([
                               transforms.ColorJitter(hue=(-0.5, 0.5)),
                               transforms.RandomRotation(180),
                               transforms.CenterCrop(crop_size),
                               transforms.RandomVerticalFlip(),
                               transforms.RandomHorizontalFlip(),
                           ] + base_transform),
        transforms.Compose([
                               transforms.ColorJitter(hue=(-0.5, 0.5)),
                               transforms.RandomRotation(180),
                               transforms.RandomVerticalFlip(),
                               transforms.RandomHorizontalFlip(),
                           ] + base_transform),
        transforms.Compose([
                               transforms.ColorJitter(hue=(-0.5, 0.5)),
                               transforms.RandomRotation(180),
                               transforms.RandomVerticalFlip(),
                               transforms.RandomHorizontalFlip(),
                           ] + base_transform),
    ]
    test_transforms = transforms.Compose([Crop(crop_size), ] + base_transform)

    return train_transforms, val_transforms, test_transforms
