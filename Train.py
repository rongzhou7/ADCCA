from tqdm.auto import tqdm
from utils import *
from ADCCA import ADCCA_4_M
import torch.nn as nn

import warnings
warnings.simplefilter("ignore", UserWarning)
import numpy as np
# import matplotlib.pyplot as plt

import os
import pandas as pd

# Seed Setting
random_seed = 100
set_seed(random_seed)
# PATH = "./AttDGCCA.pt"
def train_SDGCCA(hyper_dict):
    # Return List
    ensemble_list = {'ACC': [], 'F1': [], 'AUC': [], 'MCC': []}
    metric_list = ['ACC', 'F1', 'AUC', 'MCC']
    hyper_param_list = []
    best_hyper_param_list = []

    # Prepare Toy Dataset
    dataset = Toy_Dataset(hyper_dict['random_seed'])

    # 5 CV
    for cv in tqdm(range(5), desc='CV...'):
        # Prepare Dataset
        [x_train_1, x_val_1, x_test_1], [x_train_2, x_val_2, x_test_2], [x_train_3, x_val_3, x_test_3], [x_train_4, x_val_4, x_test_4],\
        [y_train, y_val, y_test] = dataset(cv, tensor=True, device=hyper_dict['device'])

        # Define Deep neural network dimension of the each modality
        m1_embedding_list = [x_train_1.shape[1]] + hyper_dict['embedding_size']
        m2_embedding_list = [x_train_2.shape[1]] + hyper_dict['embedding_size']
        m3_embedding_list = [x_train_3.shape[1]] + hyper_dict['embedding_size'][1:]
        m4_embedding_list = [x_train_4.shape[1]] + hyper_dict['embedding_size']

        # Train Label -> One_Hot_Encoding
        y_train_onehot = torch.zeros(y_train.shape[0], 2).float().to(hyper_dict['device'])
        y_train_onehot[range(y_train.shape[0]), y_train.squeeze()] = 1

        # Find Best K by Validation MCC
        val_mcc_result_list = []
        test_ensemble_dict = {'ACC': [], 'F1': [], 'AUC': [], 'MCC': []}

        output_list = []

        if not os.path.exists('resultAMmini'):
            os.makedirs('resultAMmini')

        # Grid search for find best hyperparameter by Validation MCC
        for top_k in tqdm(range(1, hyper_dict['max_top_k'] + 1), desc='Grid seach for find best hyperparameter...'):
            for lr in hyper_dict['lr']:
                for reg in hyper_dict['reg']:
                    for lcor in hyper_dict['lcor']:
                        hyper_param_list.append([top_k, lr, reg, lcor])
                        early_stopping = EarlyStopping(patience=hyper_dict['patience'], delta=hyper_dict['delta'])
                        best_loss = np.Inf

                        # Define ADCCA with 4 modality
                        model = ADCCA_4_M(m1_embedding_list, m2_embedding_list, m3_embedding_list, m4_embedding_list, 3).to(
                            hyper_dict['device'])

                        # Optimizer
                        clf_optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=reg)

                        # Cross Entropy Loss
                        criterion = nn.CrossEntropyLoss()

                        epoch_cor_losses = []
                        epoch_clf_losses = []

                        # Model Train
                        for i in range(hyper_dict['epoch']):
                            model.train()

                            # Calculate correlation loss
                            out1, out2, out3, out4, out1o, out2o, out3o, out4o = model(x_train_1, x_train_2, x_train_3, x_train_4)
                            # cor_loss = model.cal_loss([out1, out2, out3, y_train_onehot])
                            cor_loss_o = model.cal_loss([out1o, out2o, out3o, out4o, y_train_onehot])

                            # record the cor loss
                            epoch_cor_losses.append(cor_loss_o)
                            epoch_cor_losses_int = [item.item() for item in epoch_cor_losses]

                            # Calculate classification loss
                            clf_optimizer.zero_grad()

                            y_hat1, y_hat2, y_hat3, y_hat4, _ = model.predict(x_train_1, x_train_2, x_train_3, x_train_4)
                            clf_loss1 = criterion(y_hat1, y_train.squeeze())
                            clf_loss2 = criterion(y_hat2, y_train.squeeze())
                            clf_loss3 = criterion(y_hat3, y_train.squeeze())
                            clf_loss4 = criterion(y_hat4, y_train.squeeze())
                            clf_mean = clf_loss1 + clf_loss2 + clf_loss3 + clf_loss4

                            # clf_loss = lcor * cor_loss_o
                            clf_loss = clf_mean

                            # record the clf loss
                            epoch_clf_loss = clf_loss1 + clf_loss2 + clf_loss3 + clf_loss4
                            epoch_clf_losses.append(epoch_clf_loss)
                            epoch_clf_losses_int = [item.item() for item in epoch_clf_losses]

                            clf_loss.backward()
                            clf_optimizer.step()

                            # Model Validation
                            with torch.no_grad():
                                model.eval()
                                _, _, _, _, y_ensemble = model.predict(x_val_1, x_val_2, x_val_3, x_val_4)
                                val_loss = criterion(y_ensemble, y_val.squeeze())

                                early_stopping(val_loss)
                                if val_loss < best_loss:
                                    best_loss = val_loss

                                if early_stopping.early_stop:
                                    break

                        # # draw the loss figure
                        # plt.figure()
                        # # plt.plot(epoch_cor_losses_int)
                        # # plt.plot(epoch_clf_losses_int)
                        #
                        # plt.title("cv {}".format(cv))
                        # plt.plot(epoch_cor_losses_int, label='epoch_cor_losses')
                        # plt.plot(epoch_clf_losses_int, label='epoch_clf_losses')
                        # plt.ylabel('loss')
                        # plt.xlabel('epoch')
                        # plt.legend()
                        # #plt.show()
                        # plt.savefig(f'images/iteration_{top_k}_{lr}_{reg}.png')

                        # Load Best Model
                        model.eval()

                        # Model Validation
                        _, _, _, _, ensembel_y_hat = model.predict(x_val_1, x_val_2, x_val_3, x_val_4)
                        y_pred_ensemble = torch.argmax(ensembel_y_hat, 1).cpu().detach().numpy()
                        y_pred_proba_ensemble = ensembel_y_hat[:, 1].cpu().detach().numpy()
                        val_acc, val_f1, val_auc, val_mcc = calculate_metric(y_val.cpu().detach().numpy(), y_pred_ensemble,
                                                            y_pred_proba_ensemble)
                        validation_result = [val_acc, val_f1, val_auc, val_mcc]
                        #val_mcc_result_list.append(val_mcc)
                        val_mcc_result_list.append(val_f1)

                        # Model Tset
                        _, _, _, _, ensembel_y_hat = model.predict(x_test_1, x_test_2, x_test_3, x_test_4)
                        y_pred_ensemble = torch.argmax(ensembel_y_hat, 1).cpu().detach().numpy()
                        y_pred_proba_ensemble = ensembel_y_hat[:, 1].cpu().detach().numpy()
                        test_acc, test_f1, test_auc, test_mcc = calculate_metric(y_test.cpu().detach().numpy(),
                                                                                 y_pred_ensemble, y_pred_proba_ensemble)
                        ensemble_result = [test_acc, test_f1, test_auc, test_mcc]
                        for k, metric in enumerate(metric_list):
                            test_ensemble_dict[metric].append(ensemble_result[k])
                        print(f'top_k: {top_k}, lr: {lr}, reg: {reg}, lcor: {lcor}, epoch: {i}, {ensemble_result}')
                        output_dict = {
                            'top_k': top_k,
                            'lr': lr,
                            'reg': reg,
                            'lcor': lcor,
                            'epoch': i,
                            'val_acc': val_acc,
                            'val_f1': val_f1,
                            'val_auc': val_auc,
                            'val_mcc': val_mcc,
                            'test_acc': test_acc,
                            'test_f1': test_f1,
                            'test_auc': test_auc,
                            'test_mcc': test_mcc
                        }

                        output_list.append(output_dict)

        output_df = pd.DataFrame(output_list)

        # output_df.to_csv(f"resultAMmini/AMminicv{cv}.csv", index=False)
        # Find best K
        best_k = np.argmax(val_mcc_result_list)

        # Find best hyperparameter
        best_hyper_param_list.append(hyper_param_list[best_k])

        # torch.save(model.state_dict(), f"resultAMmini/ADGCCAmini_AM{cv}.pt")

        # Append Best K Test Result
        for metric in metric_list:
            ensemble_list[metric].append(test_ensemble_dict[metric][best_k])
        #torch.save(model.state_dict(), f"AttDGCCA{cv}.pt")

    return ensemble_list, best_hyper_param_list

if __name__ == '__main__':
    hyper_dict = {'epoch': 1000, 'delta': 0, 'random_seed': random_seed,
                  'device': torch.device("cuda:2" if torch.cuda.is_available() else "cpu"),
                  'lr': [0.0001, 0.00001], 'reg': [0, 0.01, 0.001, 0.0001],
                  'patience': 30, 'embedding_size': [256, 64, 16], 'max_top_k': 10,
                  'lcor': [0]}
                 # 1, 0.5, 0.4, 0.3, 0.2, 0.1, 0.01, 0


    ensemble_list, hyper = train_SDGCCA(hyper_dict)

    # Check Performance
    performance_result = check_mean_std_performance(ensemble_list)

    print('Test Performance')
    print('ACC: {} F1: {} AUC: {} MCC: {}'.format(performance_result[0], performance_result[1], performance_result[2],
                                                  performance_result[3]))

    print('\nBest Hyperparameter')
    for i, h in enumerate(hyper):
        print('CV: {} Best k: {} Learning Rate: {} Regularization Term: {} lor: {}'.format(i + 1, h[0], h[1], h[2], h[3]))


