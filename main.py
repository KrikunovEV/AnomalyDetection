import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


with open('sat', 'rb') as f:
    dict = pickle.load(f)

data = dict['vectors']
labels = dict['labels'].reshape(-1)

print(f'data shape: {data.shape}')
print(f'labels shape: {labels.shape}')

compute_distances = False
if compute_distances:
    distances = np.zeros((data.shape[0], 1))
    for shift in range(1, data.shape[0]):
        shifted_data = np.roll(data, shift=-shift, axis=0)
        d = np.sqrt(np.sum(np.power(data - shifted_data, 2), axis=1))
        distances = np.hstack((distances, d[:, None]))
        print(f'{shift}')
    distances = distances[:, 1:]
    distances = np.sort(distances, axis=1)
    with open('distances.pickle', 'wb') as f:
        pickle.dump(distances, f)
else:
    with open('distances.pickle', 'rb') as f:
        distances = pickle.load(f)

AveragePrecision = [0]
bestP, bestR, bestF1, bestM, best = [], [], [], [], 0
K = np.arange(5, data.shape[0] // 2, 5)
for k in K:
    print(f'k={k}')
    anomaly_scores = distances[:, k]

    ind = np.argsort(anomaly_scores)
    anomaly_scores = anomaly_scores[ind]
    l = labels[ind]

    # Compute metrics
    Precision, Recall, F1Score, Matrix = [], [], [], []
    for t in range(1, len(anomaly_scores) - 1):
        l_negative = l[:t]
        l_positive = l[t:]

        TP = np.sum(l_positive == 1)
        FP = len(l_positive) - TP
        FN = np.sum(l_negative == 1)
        Matrix.append(np.empty((2, 2)))
        Matrix[-1][0, 0] = TP
        Matrix[-1][0, 1] = FP
        Matrix[-1][1, 0] = len(l_negative) - FN
        Matrix[-1][1, 1] = FN

        Precision.append(TP / (TP + FP))
        Recall.append(TP / (TP + FN))
        score = Precision[-1] + Recall[-1]
        if score != 0:  # To check div by 0
            score = 2 * Precision[-1] * Recall[-1] / score
        F1Score.append(score)

    # Fix false holes on curve
    Precision = Precision[::-1]
    p = Precision[-1]
    for i in reversed(range(len(Precision) - 1)):
        if Precision[i] < p:
            Precision[i] = p
        else:
            p = Precision[i]
    Precision = Precision[::-1]

    # Calculate AP
    AveragePrecision.append(0)
    for i in range(len(Recall) - 1):
        AveragePrecision[-1] += Precision[i+1] * (Recall[i] - Recall[i+1])

    # Store best AP
    if AveragePrecision[-1] > AveragePrecision[best]:
        best = len(AveragePrecision) - 1
        bestP = Precision
        bestR = Recall
        bestF1 = F1Score
        bestM = Matrix

# Visualize Precision-Recall curve
ind_max = np.argmax(bestF1)
plt.title(f'Best are F1={np.around(bestF1[ind_max], 3)},'
          f' R={np.around(bestR[ind_max], 3)},'
          f' P={np.around(bestP[ind_max], 3)},'
          f' AP={np.around(AveragePrecision[best], 3)},'
          f' k={K[best-1]}')
plt.plot(bestR, bestP)
plt.show()

# Print results
print(f'\nConfusion Matrix:\n'
      f'         True  False\n'
      f'Positive {bestM[ind_max][0, 0]} {bestM[ind_max][0, 1]}\n'
      f'Negative {bestM[ind_max][1, 0]} {bestM[ind_max][1, 1]}\n\n'
      f'Accuracy by matrix: {(bestM[ind_max][0, 0] + bestM[ind_max][1, 0]) / data.shape[0]}\n\n'
      f'Best: F1={np.around(bestF1[ind_max], 3)}, P={np.around(bestP[ind_max], 3)}, R={np.around(bestR[ind_max], 3)}\n'
      f'AveragePrecision={np.around(AveragePrecision[best], 3)}\n'
      f'K={K[best-1]}')

# Visualize data
k = K[best-1]
anomaly_scores = distances[:, k]
threshold = ind_max + 1  # best F1Score index (+1 because threshold start from 1)

ind = np.argsort(anomaly_scores)
_labels = labels[ind]
_data = data[ind]

tsne = TSNE(n_components=2).fit_transform(_data)
fig, ax = plt.subplots(1, 2)
ax[0].set_title('Real labels')
ax[0].scatter(tsne[:, 0], tsne[:, 1], c=['r' if l == 1 else 'g' for l in _labels])
ax[1].set_title('Predicted labels')
ax[1].scatter(tsne[:threshold, 0], tsne[:threshold, 1], c='g')
ax[1].scatter(tsne[threshold:, 0], tsne[threshold:, 1], c='r')
plt.show()
