import scipy.io as sio
from scipy.io import loadmat
from sklearn.metrics import accuracy_score, average_precision_score
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn import model_selection
import functools
import itertools
import numpy
import operator

labels = ['0','1','2','3','4','5','6','7','8','9','10']
tick_marks = np.array(range(len(labels))) + 0.5

def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.GnBu):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=90)
    plt.yticks(xlocations, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


ytest = loadmat('./results/result.mat')['truth'].T
ytest_pred = loadmat('./results/result.mat')['prediction'].T

cm = confusion_matrix(ytest, ytest_pred)

np.set_printoptions(precision=2)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(12, 8), dpi=120)

ind_array = np.arange(len(labels))
x, y = np.meshgrid(ind_array, ind_array)

for x_val, y_val in zip(x.flatten(), y.flatten()):

    c = cm_normalized[y_val][x_val]

    if (0.1 > c > 0.01):
        plt.text(x_val, y_val, "%0.2f" % (100 * c,)+'%', color='black', fontsize=10, va='center', ha='center')
    elif c > 0.1:
        plt.text(x_val, y_val, "%0.2f" % (100 * c,) + '%', color='white', fontsize=10, va='center', ha='center')
plt.gca().set_xticks(tick_marks, minor=True)
plt.gca().set_yticks(tick_marks, minor=True)
plt.gca().xaxis.set_ticks_position('none')
plt.gca().yaxis.set_ticks_position('none')
plt.grid(True, which='minor', linestyle='-')
plt.gcf().subplots_adjust(bottom=0.15)

plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
plt.show()
