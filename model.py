import numpy as np
import torch
from tqdm import tqdm


class Model:
    def __init__(self, model, loss, optimizer, scheduler, num_epochs, ):
        # Disable grad for all conv layers

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.loss_hist = {'train': [], 'val': []}
        self.acc_hist = {'train': [], 'val': []}

    def fit(self, train_dataloader, val_dataloader):
        self.loss_hist = {'train': [], 'val': []}
        self.acc_hist = {'train': [], 'val': []}

        for epoch in range(self.num_epochs):
            print('Epoch {}/{}:'.format(epoch, self.num_epochs - 1), flush=True)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    dataloader = train_dataloader
                    # optimizer.step()
                    self.scheduler.step()

                    self.model.train()  # Set model to training mode
                else:
                    dataloader = val_dataloader
                    self.model.eval()  # Set model to evaluate mode

                running_loss = 0.
                running_acc = 0.

                # Iterate over data.
                for inputs, labels in tqdm(dataloader):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    self.optimizer.zero_grad()

                    # forward and backward
                    with torch.set_grad_enabled(phase == 'train'):
                        preds = self.model(inputs)
                        loss_value = self.loss(preds, labels)
                        preds_class = preds.argmax(dim=1)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss_value.backward()
                            self.optimizer.step()

                    # statistics
                    running_loss += loss_value.item()
                    running_acc += (preds_class == labels.data).float().mean()

                #             if phase == 'train':
                #                 scheduler.step()

                epoch_loss = running_loss / len(dataloader)
                epoch_acc = running_acc / len(dataloader)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc), flush=True)

                self.loss_hist[phase].append(epoch_loss)
                self.acc_hist[phase].append(epoch_acc)

        return self.loss_hist, self.acc_hist

    def predict(self, test_dataloader):
        self.model.eval()

        test_predictions = []
        test_img_paths = []
        for inputs, labels, paths in tqdm(test_dataloader):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            with torch.set_grad_enabled(False):
                preds = self.model(inputs)
            test_predictions.append(
                torch.nn.functional.softmax(preds, dim=1)[:, 1].data.cpu().numpy())
            test_img_paths.extend(paths)

        return np.concatenate(test_predictions)




