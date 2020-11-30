import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn


class LSTMAD:

    def __init__(self, train_0, validation1_0, validation2_0, validation_1, test_0, test_1,
                 model, optimizer, loss_fn, eval_fn, cfg):
        self.train_0 = train_0
        self.validation1_0 = validation1_0
        self.validation2_0 = validation2_0
        self.validation_1 = validation_1
        self.test_0 = test_0
        self.test_1 = test_1
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.eval_fn = eval_fn
        self.cfg = cfg

        mask = torch.zeros((cfg.l, cfg.length), dtype=torch.bool)
        for i in range(cfg.l):
            mask[i, cfg.l - 1 - i:cfg.length - i] = True
        self.mask = mask.T.reshape(-1)

        self.threshold = None
        self.mu = None
        self.var = None

    def train(self):
        train_losses, valid_losses = [], []

        for i_batch, x in enumerate(self.train_0):

            y = self.model(x)
            y = y.reshape(-1)[self.mask]

            x = x[0]
            x = x[self.cfg.l - 1:].expand(-1, 3).reshape(-1)

            loss = self.loss_fn(x, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_losses.append(loss.item())
            print(f'element: {i_batch + 1}/{len(self.train_0)}, loss: {train_losses[-1]}')

        for i_batch, x in enumerate(self.validation1_0):

            y = self.model(x)
            y = y.reshape(-1)[self.mask]

            x = x[0]
            x = x[self.cfg.l - 1:].expand(-1, 3).reshape(-1)

            loss = self.loss_fn(x, y)

            valid_losses.append(loss.item())
            print(f'element: {i_batch + 1}/{len(self.validation1_0)}, loss: {valid_losses[-1]}')

        train_ind = np.random.randint(len(self.train_0))
        valid_ind = np.random.randint(len(self.validation1_0))
        train_gt = self.train_0.dataset[train_ind]
        valid_gt = self.validation1_0.dataset[valid_ind]
        train_pred = self.model(torch.Tensor(train_gt).unsqueeze(0)).detach().numpy()
        valid_pred = self.model(torch.Tensor(valid_gt).unsqueeze(0)).detach().numpy()

        self.__train_plot(train_losses, valid_losses, train_gt, train_pred, valid_gt, valid_pred)

    def validate(self):
        evals_normal_1, mu, var = self.__eval_mu_var()

        evals_normal = self.__eval(self.validation2_0, mu, var)
        evals_anomaly = self.__eval(self.validation_1, mu, var)

        FS, beta, threshold, PR, CM = self.__find_threshold(evals_normal, evals_anomaly)

        self.__validate_plot(evals_normal_1, evals_normal, evals_anomaly, FS, beta, threshold, PR, CM)

        self.threshold = threshold
        self.mu = mu
        self.var = var

    def test(self):
        evals_normal = self.__eval(self.test_0, self.mu, self.var)
        evals_anomaly = self.__eval(self.test_1, self.mu, self.var)

        CM = self.__find_CM(evals_normal, evals_anomaly)

        self.__test_plot(evals_normal, evals_anomaly, CM)

    def __eval_mu_var(self):
        mu, var = [], []
        evals = []
        for i_batch, x in enumerate(self.validation1_0):
            y = self.model(x)
            y = y.reshape(-1)[self.mask]
            y = y.reshape(-1, self.cfg.l)

            x = x[0]
            x = x[self.cfg.l - 1:].expand(-1, 3)

            e = x - y
            mu.append(e.mean(dim=1))
            var.append(e.var(dim=1))
            evals.append(self.eval_fn(e, mu[-1], var[-1]))
            print(f'eval mu and var, element: {i_batch + 1}/{len(self.validation1_0)}')
        return torch.stack(evals), torch.stack(mu), torch.stack(var)

    def __eval(self, dataset, mu, var):
        evals = []
        for i_batch, x in enumerate(dataset):
            y = self.model(x)
            y = y.reshape(-1)[self.mask]
            y = y.reshape(-1, self.cfg.l)

            x = x[0]
            x = x[self.cfg.l - 1:].expand(-1, 3)

            e = x - y
            evals.append(self.eval_fn(e, mu[i_batch], var[i_batch]))
            print(f'compute evals, element: {i_batch + 1}/{len(dataset)}')
        return torch.stack(evals)

    def __find_threshold(self, evals_normal, evals_anomaly):
        evals_normal = evals_normal.min(dim=1)[0]
        evals_anomaly = evals_anomaly.min(dim=1)[0]

        betas = np.arange(1, 10) / 10.
        thresholds = np.linspace(0, -50000, 1000)

        FS = 0
        threshold = 0
        beta = 0
        CM = 0
        PR = 0
        for i, b in enumerate(betas):
            P, R = [], []
            f_score_changed = False
            for j, t in enumerate(thresholds):
                TP, FP, TN, FN = 0, 0, 0, 0
                for normal, anomaly in zip(evals_normal, evals_anomaly):
                    # normal - Negative
                    # anomaly - Positive
                    if normal >= t:
                        TN += 1
                    else:
                        FP += 1

                    if anomaly >= t:
                        FN += 1
                    else:
                        TP += 1

                Precision = TP / (TP + FP) if TP + FP > 0 else 0
                Recall = TP / (TP + FN) if TP + FN > 0 else 0
                P.append(Precision)
                R.append(Recall)
                if Precision == 0 and Recall == 0:
                    F_score = 0
                else:
                    F_score = (1 + b ** 2) * Precision * Recall / (Precision * b ** 2 + Recall)

                if F_score > FS:
                    f_score_changed = True
                    FS = F_score
                    beta = b
                    CM = np.array([[TP, FP], [FN, TN]])
                    threshold = t

                print(f'beta: {b} ({i + 1}/{len(betas)}), threshold: {t} ({j+1}/{len(thresholds)}), best F score: {FS}')
            if f_score_changed:
                PR = [P, R]

        return FS, beta, threshold, PR, CM

    def __find_CM(self, evals_normal, evals_anomaly):
        evals_normal = evals_normal.min(dim=1)[0]
        evals_anomaly = evals_anomaly.min(dim=1)[0]
        TP, FP, TN, FN = 0, 0, 0, 0
        for normal, anomaly in zip(evals_normal, evals_anomaly):
            if normal >= self.threshold:
                TN += 1
            else:
                FP += 1

            if anomaly >= self.threshold:
                FN += 1
            else:
                TP += 1
        CM = np.array([[TP, FP], [FN, TN]])
        return CM

    def __train_plot(self, train_losses, valid_losses, train_gt, train_pred, valid_gt, valid_pred):
        fig, ax = plt.subplots(1, 2, figsize=(16, 9))
        ax[0].set_title('MSE loss on train')
        ax[0].set_xlabel('iteration')
        ax[0].set_ylabel('loss')
        ax[0].plot(train_losses)
        ax[1].set_title('MSE loss on valid')
        ax[1].set_xlabel('iteration')
        ax[1].set_ylabel('loss')
        ax[1].plot(valid_losses)
        fig.tight_layout()
        plt.savefig('losses.png')

        fig, ax = plt.subplots(2, 2, figsize=(16, 9))
        ax[0][0].set_title('train ground truth')
        ax[0][0].set_xlabel('time')
        ax[0][0].set_ylabel('value')
        ax[0][0].plot(train_gt)
        ax[0][1].set_title('train predictions')
        ax[0][1].set_xlabel('time')
        ax[0][1].set_ylabel('predictions')
        for i, line in enumerate(train_pred):
            ax[0][1].plot(np.arange(i, i + len(line)), line)
        ax[1][0].set_title('valid ground truth')
        ax[1][0].set_xlabel('time')
        ax[1][0].set_ylabel('value')
        ax[1][0].plot(valid_gt)
        ax[1][1].set_title('valid predictions')
        ax[1][1].set_xlabel('time')
        ax[1][1].set_ylabel('predictions')
        for i, line in enumerate(valid_pred):
            ax[1][1].plot(np.arange(i, i + len(line)), line)
        fig.tight_layout()
        plt.savefig('predictions.png')

    def __validate_plot(self, evals_normal_1, evals_normal_2, evals_anomaly_2, FS, beta, threshold, PR, CM):
        ind = np.random.randint(0, len(evals_normal_1), 2)
        evals_normal_1 = evals_normal_1[ind]
        evals_normal_2 = evals_normal_2[ind]
        evals_anomaly_2 = evals_anomaly_2[ind]

        fig, ax = plt.subplots(1, 2, figsize=(16, 9))
        for i in range(2):
            ax[i].set_title(f'Example #{i + 1}')
            ax[i].set_xlabel('time')
            ax[i].set_ylabel('eval')

            ax[i].plot(evals_normal_1[i].detach().numpy(), label='validation1_0')
            ax[i].plot(evals_normal_2[i].detach().numpy(), label='validation2_0')
            ax[i].plot(evals_anomaly_2[i].detach().numpy(), label='validation_1')
            ax[i].plot([0, len(evals_normal_1[i].detach().numpy())], [threshold, threshold], '--', label='threshold')
            ax[i].legend()
        fig.tight_layout()
        plt.savefig('valid_examples.png')

        plt.figure()
        plt.title(f'PRC, beta: {beta}, threshold: {np.around(threshold, 3)}, F score: {np.around(FS, 3)}')
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.plot(PR[1], PR[0])
        plt.tight_layout()
        plt.savefig('PRC.png')

        plt.figure(figsize=(16, 9))
        plt.title(f'Confusion matrix (anomaly is positive)')
        plt.xlabel('Real class')
        plt.ylabel('Predicted class')
        CM = np.around(CM / CM.sum(), 3)
        labels = np.array([[f'{CM[0, 0]} (TP)', f'{CM[0, 1]} (FP)'], [f'{CM[1, 0]} (FN)', f'{CM[1, 1]} (TN)']])
        sn.heatmap(CM, annot=labels, fmt='', cmap='Reds', xticklabels=False, yticklabels=False, cbar=False)
        plt.savefig('CM_valid.png')

    def __test_plot(self, evals_normal, evals_anomaly, CM):
        ind = np.random.randint(0, len(evals_normal), 2)
        evals_normal = evals_normal[ind]
        evals_anomaly = evals_anomaly[ind]

        fig, ax = plt.subplots(1, 2, figsize=(16, 9))
        for i in range(2):
            ax[i].set_title(f'Example #{i + 1}')
            ax[i].set_xlabel('time')
            ax[i].set_ylabel('eval')

            ax[i].plot(evals_normal[i].detach().numpy(), label='test_0')
            ax[i].plot(evals_anomaly[i].detach().numpy(), label='test_1')
            ax[i].plot([0, len(evals_normal[i].detach().numpy())], [self.threshold, self.threshold], '--',
                       label='threshold')
            ax[i].legend()
        fig.tight_layout()
        plt.savefig('test_examples.png')

        plt.figure(figsize=(16, 9))
        plt.title(f'Confusion matrix (anomaly is positive)')
        plt.xlabel('Real class')
        plt.ylabel('Predicted class')
        CM = np.around(CM / CM.sum(), 3)
        labels = np.array([[f'{CM[0, 0]} (TP)', f'{CM[0, 1]} (FP)'], [f'{CM[1, 0]} (FN)', f'{CM[1, 1]} (TN)']])
        sn.heatmap(CM, annot=labels, fmt='', cmap='Reds', xticklabels=False, yticklabels=False, cbar=False)
        plt.savefig('CM_test.png')
