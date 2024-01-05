import torch
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, matthews_corrcoef
import sys
sys.path.append("..")

# Calculate performance metric
def calculate_metric(y_true, y_pred, y_pred_proba):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred_proba)
    mcc = matthews_corrcoef(y_true, y_pred)

    TP = sum((y_true == 1) & (y_pred == 1))
    TN = sum((y_true == 0) & (y_pred == 0))
    FP = sum((y_true == 0) & (y_pred == 1))
    FN = sum((y_true == 1) & (y_pred == 0))

    # Calculate Sensitivity (True Positive Rate, TPR) and Specificity (True Negative Rate, TNR)
    sen = TP / (TP + FN)
    spe = TN / (TN + FP)

    return acc, f1, auc, mcc
    #return acc, auc, sen, spe

# Check mean and standard deviation
def check_mean_std_performance(result):
    return_list = []


    for m in ['ACC', 'F1', 'AUC', 'MCC']:
        return_list.append('{:.2f}+-{:.2f}'.format(np.array(result[m]).mean() * 100, np.array(result[m]).std() * 100))

    return return_list

# Setting random seed
def set_seed(random_seed):
    # Seed Setting
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# For early stopping
class EarlyStopping:
    def __init__(self, patience=100, delta=1e-3):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score

        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        elif score > self.best_score:
            self.best_score = score
            self.counter = 0
            self.early_stop = False

        else:
            self.best_score = score
            self.counter = 0

# Toy Dataset Class
"""
- Label: Binary
- Modality1: n(579) x d1 (90)
- Modality2: n(579) x d2 (90)
- Modality3: n(579) x d3 (54)
- Modality4: n(579) x d4 (90)
"""
class Toy_Dataset:
    def __init__(self, random_seed):
        # Make random dataset
        # label = np.random.randint(2, size=(376, 1))
        # data1 = np.random.rand(376, 18164)
        # data2 = np.random.rand(376, 19353)
        # data3 = np.random.rand(376, 309)

        # data1N = pd.read_csv('data/mri.csv')
        # data2N = pd.read_csv('data/vbm.csv')
        # data3N = pd.read_csv('data/snp1.csv')
        # data4N = pd.read_csv('data/fdg.csv')
        # # label = pd.read_csv('data/dia_l.csv')
        # label = pd.read_csv('data/label_hcall.csv')


        data1N = pd.read_csv('data_AH/mri.csv')
        data2N = pd.read_csv('data_AH/vbm.csv')
        data3N = pd.read_csv('data_AH/snp1ah.csv')
        data4N = pd.read_csv('data_AH/fdg.csv')
        label = pd.read_csv('data_AH/dia.csv')

        # data1N = pd.read_csv('data_MH/mri.csv')
        # data2N = pd.read_csv('data_MH/vbm.csv')
        # data3N = pd.read_csv('data_MH/snp1mh.csv')
        # data4N = pd.read_csv('data_MH/fdg.csv')
        # label = pd.read_csv('data_MH/dia.csv')

        # data1N = pd.read_csv('data_AM/mri.csv')
        # data2N = pd.read_csv('data_AM/vbm.csv')
        # data3N = pd.read_csv('data_AM/snp1am.csv')
        # data4N = pd.read_csv('data_AM/fdg.csv')
        # label = pd.read_csv('data_AM/dia.csv')

        label = np.array(label)

        data1 = data1N.apply(lambda x:(x-x.mean())/(x.std()))
        data1 = data1.fillna(0)
        data1 = np.array(data1)

        data2 = data2N.apply(lambda x: (x - x.mean()) / (x.std()))
        data2 = data2.fillna(0)
        data2 = np.array(data2)

        data3 = data3N.apply(lambda x: (x - x.mean()) / (x.std()))
        data3 = data3.fillna(0)
        data3 = np.array(data3)

        data4 = data4N.apply(lambda x: (x - x.mean()) / (x.std()))
        data4 = data4.fillna(0)
        data4 = np.array(data4)

        # 5CV Dataset
        self.dataset = {'cv1': None, 'cv2': None, 'cv3': None, 'cv4': None, 'cv5': None}

        # Split Train,Validation and Test with 5 CV Fold
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
        for i, (train_val_index, test_index) in enumerate(kf.split(data1, label)):
            x_train_val_1, x_test_1, y_train_val, y_test = data1[train_val_index], data1[test_index], label[train_val_index], label[test_index]

            x_train_val_2, x_test_2 = data2[train_val_index], data2[test_index]
            x_train_val_3, x_test_3 = data3[train_val_index], data3[test_index]
            x_train_val_4, x_test_4 = data4[train_val_index], data4[test_index]

            # Split Train and Validation
            x_train_1, x_val_1, y_train, y_val = train_test_split(x_train_val_1, y_train_val, test_size=0.2,
                                                                  random_state=random_seed)
            x_train_2, x_val_2, _, _ = train_test_split(x_train_val_2, y_train_val, test_size=0.2,
                                                        random_state=random_seed)
            x_train_3, x_val_3, _, _ = train_test_split(x_train_val_3, y_train_val, test_size=0.2,
                                                        random_state=random_seed)
            x_train_4, x_val_4, _, _ = train_test_split(x_train_val_4, y_train_val, test_size=0.2,
                                                        random_state=random_seed)

            # CV Dataset
            cv_dataset = [[x_train_1, x_val_1, x_test_1], [x_train_2, x_val_2, x_test_2],
                          [x_train_3, x_val_3, x_test_3], [x_train_4, x_val_4, x_test_4], [y_train, y_val, y_test]]
            self.dataset['cv' + str(i + 1)] = cv_dataset

    def __call__(self, cv, tensor=True, device=None):
        [x_train_1, x_val_1, x_test_1], [x_train_2, x_val_2, x_test_2], [x_train_3, x_val_3, x_test_3], [x_train_4, x_val_4, x_test_4],\
        [y_train, y_val, y_test] = self.dataset['cv' + str(cv + 1)]

        # Numpy to tensor
        # Modality 1
        x_train_1 = torch.tensor(x_train_1).float().to(device)
        x_val_1 = torch.tensor(x_val_1).float().to(device)
        x_test_1 = torch.tensor(x_test_1).float().to(device)

        # Modality 2
        x_train_2 = torch.tensor(x_train_2).float().to(device)
        x_val_2 = torch.tensor(x_val_2).float().to(device)
        x_test_2 = torch.tensor(x_test_2).float().to(device)

        # Modality 3
        x_train_3 = torch.tensor(x_train_3).float().to(device)
        x_val_3 = torch.tensor(x_val_3).float().to(device)
        x_test_3 = torch.tensor(x_test_3).float().to(device)

        # Modality 4
        x_train_4 = torch.tensor(x_train_4).float().to(device)
        x_val_4 = torch.tensor(x_val_4).float().to(device)
        x_test_4 = torch.tensor(x_test_4).float().to(device)

        # Label
        y_train = torch.tensor(y_train).long().to(device)
        y_val = torch.tensor(y_val).long().to(device)
        y_test = torch.tensor(y_test).long().to(device)

        return [x_train_1, x_val_1, x_test_1], [x_train_2, x_val_2, x_test_2], [x_train_3, x_val_3, x_test_3], [x_train_4, x_val_4, x_test_4],\
        [y_train, y_val, y_test]
