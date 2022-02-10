import os
import random
import warnings

import numpy as np
import torch
from torchvision import models

from dataloaders import get_dataloaders
from helpers import show_loss_acc, make_submission, acc_submission
from model import Model
from prepare import prepare_image, make_train_val_test
from transforms import get_transforms

warnings.filterwarnings("ignore")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def go(prepare=False, show=False):
    data_root = '../plates/'

    dirs = ['train/cleaned', 'train/dirty', 'test']
    if prepare:
        prepare_image(dirs, data_root)
        make_train_val_test(data_root)

    train_dir = os.path.join(data_root, 'train')
    val_dir = os.path.join(data_root, 'val')
    test_dir = os.path.join(data_root, 'test')

    train_transforms, val_transforms, test_transforms = get_transforms()
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders((train_dir, train_transforms),
                                                                        (val_dir, val_transforms),
                                                                        (test_dir, test_transforms),
                                                                        batch_size=16)

    model = models.resnet18(pretrained=True)
    #Disable grad for all conv layers
    #     for param in model.parameters():
    #         param.requires_grad = False

    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    device = torch.device( "cpu")
    model.to(device)

    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)#, amsgrad=True)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    m = Model(model, loss, optimizer, scheduler, num_epochs=7, show=show)

    loss, acc = m.fit(train_dataloader, val_dataloader, test_dataloader)

    if show:
        show_loss_acc(loss, acc)

    test_predictions, test_img_paths = m.predict(test_dataloader)

    make_submission(test_predictions, test_img_paths, test_dir, p=0.6)

    return test_predictions, test_img_paths


if __name__ == "__main__":
    set_seed(21)
    test_predictions, test_img_paths = go()

