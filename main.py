import os

import torch
import numpy as np
from prepare import prepare_image, make_train_val_test
from transforms import get_transforms
from dataloaders import get_dataloaders
from model import Model
from helpers import show_loss_acc, make_submission, acc_submission

from torchvision import models

import random

import warnings

warnings.filterwarnings("ignore")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def go(prepare=False):
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
                                                                        (test_dir, test_transforms))

    model = models.resnet18(pretrained=True)
    # for param in model.parameters():
    #     param.requires_grad = False

    model.fc = torch.nn.Linear(model.fc.in_features, 2)

    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), amsgrad=True, lr=1.0e-3)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    m = Model(model, loss, optimizer, scheduler, num_epochs=20)

    loss, acc = m.fit(train_dataloader, val_dataloader)
    show_loss_acc(loss, acc)

    test_predictions, test_img_paths = m.predict(test_dataloader)

    make_submission(test_predictions, test_img_paths, p=0.5)

    acc_submission(test_predictions, test_img_paths, 'stuff/submission_manual.csv')


if __name__ == "__main__":
    go()

# len(train_dataloader), len(train_dataset), len(val_dataset)
#
# X_batch, y_batch = next(iter(train_dataloader))
# mean = np.array([0.485, 0.456, 0.406])
# std = np.array([0.229, 0.224, 0.225])
# plt.imshow(X_batch[0].permute(1, 2, 0).numpy() * std + mean);
#
#
# X_batch, y_batch = next(iter(train_dataloader))
#
# for x_item, y_item in zip(X_batch, y_batch):
#     show_input(x_item, title=class_names[y_item])


#
# for img, pred in zip(inputs, test_predictions):
#     show_input(img, title=pred)
