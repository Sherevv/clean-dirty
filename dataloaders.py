import torch
import torchvision


class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


def get_dataloaders(train, val, test, batch_size=16):
    train_dir, train_transforms = train
    val_dir, val_transforms = val
    test_dir, test_transforms = test
    train_dataset = torch.utils.data.ConcatDataset(
        [torchvision.datasets.ImageFolder(train_dir, t) for t in train_transforms])

    val_dataset = torch.utils.data.ConcatDataset(
        [torchvision.datasets.ImageFolder(val_dir, t) for t in val_transforms])

    test_dataset = ImageFolderWithPaths(test_dir, test_transforms)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=batch_size)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=batch_size)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_dataloader, val_dataloader, test_dataloader
