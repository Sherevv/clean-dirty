# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# if INIT:
#
#     print(os.listdir("../input"))
#
#     # Any results you write to the current directory are saved as output.
#     with zipfile.ZipFile('../input/platesv2/plates.zip', 'r') as zip_obj:
#         # Extract all the contents of zip file in current directory
#         zip_obj.extractall('/kaggle/working/')
#
#     print('After zip extraction:')
#     print(os.listdir("/kaggle/working/"))
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


def make_submission(test_predictions, test_img_paths, p=0.5):

    submission_df = pd.DataFrame.from_dict({'id': test_img_paths, 'label': test_predictions})
    submission_df['label'] = submission_df['label'].map(lambda pred: 'dirty' if pred > p else 'cleaned')
    submission_df['id'] = submission_df['id'].str.replace('plates/test/unknown/', '')
    submission_df['id'] = submission_df['id'].str.replace('.jpg', '')
    submission_df.set_index('id', inplace=True)

    submission_df.to_csv('submission.csv')


def acc_submission(test_predictions, test_img_paths, target_file):
    df = pd.DataFrame.from_dict({'id': test_img_paths, 'label0': test_predictions, 'label': ''})
    submission_pre = pd.read_csv(target_file)
    for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        df['label'] = df['label0'].map(lambda pred: 'dirty' if pred > t else 'cleaned')

        print(t, accuracy_score(df['label'], submission_pre['label']))