import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


def show_input(input_tensor, title=''):
    image = input_tensor.permute(1, 2, 0).numpy()
    image = std * image + mean
    plt.imshow(image.clip(0, 1))
    plt.title(title)
    plt.show()
    plt.pause(0.001)


def show_loss_acc(loss, acc):
    plt.rcParams['figure.figsize'] = (14, 7)
    for experiment_id in acc.keys():
        plt.plot(acc[experiment_id], label=experiment_id)
    plt.legend(loc='upper left')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch num', fontsize=15)
    plt.ylabel('Accuracy value', fontsize=15)
    plt.grid(linestyle='--', linewidth=0.5, color='.7')

    for experiment_id in loss.keys():
        plt.plot(loss[experiment_id], label=experiment_id)
    plt.legend(loc='upper left')
    plt.title('Model Loss')
    plt.xlabel('Epoch num', fontsize=15)
    plt.ylabel('Loss function value', fontsize=15)
    plt.grid(linestyle='--', linewidth=0.5, color='.7')


def make_submission(test_predictions, test_img_paths, test_dir, p=0.5, ):

    submission_df = pd.DataFrame.from_dict({'id': test_img_paths, 'label': test_predictions})
    submission_df['label'] = submission_df['label'].map(lambda pred: 'dirty' if pred > p else 'cleaned')
    submission_df['id'] = submission_df['id'].str.replace(os.path.join(test_dir, 'unknown/'), '')
    submission_df['id'] = submission_df['id'].str.replace('.jpg', '')
    submission_df.set_index('id', inplace=True)

    submission_df.to_csv('submission.csv')
