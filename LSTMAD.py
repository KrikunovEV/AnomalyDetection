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
        mu, var = self.__eval_mu_var()
        self.mu = mu
        self.var = var

        evals_normal_1 = self.__eval(self.validation1_0)
        evals_normal = self.__eval(self.validation2_0)
        evals_anomaly = self.__eval(self.validation_1)

        FS, beta, threshold, P, R, CM = self.__find_threshold(evals_normal, evals_anomaly)
        self.threshold = threshold

        self.__validate_plot(evals_normal_1, evals_normal, evals_anomaly, FS, beta, P, R, CM)

    def test(self):
        evals_normal = self.__eval(self.test_0)
        evals_anomaly = self.__eval(self.test_1)

        CM = self.__find_CM(evals_normal, evals_anomaly)

        self.__test_plot(evals_normal, evals_anomaly, CM)

    def __eval_mu_var(self):
        errors = []
        for i_batch, x in enumerate(self.validation1_0):
            y = self.model(x)
            y = y.reshape(-1)[self.mask]
            y = y.reshape(-1, self.cfg.l)

            x = x[0]
            x = x[self.cfg.l - 1:].expand(-1, 3)

            e = x - y
            errors.append(e)
            print(f'eval mu and var, element: {i_batch + 1}/{len(self.validation1_0)}')
        errors = torch.stack(errors)
        return errors.mean(), errors.var()

    def __eval(self, dataset):
        evals = []
        for i_batch, x in enumerate(dataset):
            y = self.model(x)
            y = y.reshape(-1)[self.mask]
            y = y.reshape(-1, self.cfg.l)

            x = x[0]
            x = x[self.cfg.l - 1:].expand(-1, 3)

            e = x - y
            evals.append(self.eval_fn(e, self.mu, self.var))
            print(f'compute evals, element: {i_batch + 1}/{len(dataset)}')
        return torch.stack(evals)

    def __find_threshold(self, evals_normal, evals_anomaly):
        evals = torch.cat((evals_normal.min(dim=1)[0], evals_anomaly.min(dim=1)[0])).detach().numpy()
        labels = np.concatenate((np.zeros(len(evals_normal)), np.ones(len(evals_anomaly))))

        FS = 0
        threshold = 0
        beta = self.cfg.beta
        CM = 0
        P, R = [], []

        thresholds = np.sort(evals)[1:-1]
        for i, t in enumerate(thresholds):
            l_negative = labels[evals <= t]
            l_positive = labels[evals > t]

            TP = np.sum(l_positive == 1)
            FP = len(l_positive) - TP
            FN = np.sum(l_negative == 1)

            Precision = TP / (TP + FP) if TP + FP > 0 else 0
            Recall = TP / (TP + FN) if TP + FN > 0 else 0
            P.append(Precision)
            R.append(Recall)

            if Precision == 0 and Recall == 0:
                Fb_score = 0
            else:
                Fb_score = (1 + beta ** 2) * Precision * Recall / (Precision * beta ** 2 + Recall)

            if Fb_score > FS:
                CM = np.array([[TP, FP], [FN, len(l_negative) - FN]])
                threshold = t
                FS = Fb_score

            print(f'threshold: {np.around(t, 3)} ({i + 1}/{len(thresholds)}), best F score: {np.around(FS, 3)}')

        P = P[::-1]
        p = P[-1]
        for i in reversed(range(len(P) - 1)):
            if P[i] < p:
                P[i] = p
            else:
                p = P[i]
        P = P[::-1]

        return FS, beta, threshold, P, R, CM

    def __find_CM(self, evals_normal, evals_anomaly):
        evals = torch.cat((evals_normal.min(dim=1)[0], evals_anomaly.min(dim=1)[0])).detach().numpy()
        labels = np.concatenate((np.zeros(len(evals_normal)), np.ones(len(evals_anomaly))))

        l_negative = labels[evals <= self.threshold]
        l_positive = labels[evals > self.threshold]

        TP = np.sum(l_positive == 1)
        FP = len(l_positive) - TP
        FN = np.sum(l_negative == 1)
        CM = np.array([[TP, FP], [FN, len(l_negative) - FN]])
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

    def __validate_plot(self, evals_normal_1, evals_normal_2, evals_anomaly_2, FS, beta, P, R, CM):
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
            ax[i].plot([0, len(evals_normal_1[i].detach().numpy())], [self.threshold, self.threshold], '--',
                       label='threshold')
            ax[i].legend()
        fig.tight_layout()
        plt.savefig('valid_examples.png')

        plt.figure()
        mu = self.mu.detach().numpy().astype(np.float_)
        var = self.var.detach().numpy().astype(np.float_)
        threshold = (np.array([self.threshold], dtype=np.float_))[0]
        plt.title(f'PRC, beta: {beta}, threshold: {np.around(threshold, 3)}, F score: {np.around(FS, 3)}, '
                  f'mu: {np.around(mu, 3)}, var: {np.around(var, 3)}')
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.plot(R, P)
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
