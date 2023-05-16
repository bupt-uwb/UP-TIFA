import scipy.io as sio
import numpy as np
import collections
from sklearn.model_selection import KFold
from data_loader_KFold import *
from torch.utils.data import Dataset, DataLoader, random_split
from models.model import *
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

history = collections.defaultdict(list)  # 记录每一折的各种指标
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, train_loader, loss, optimizer, epoch):
    model.train()
    correct = 0
    for batch_idx, (Data, target) in enumerate(train_loader):
        # print(len(Data))
        Data = Data.to(device)
        # print(Data.size())
        target = target.to(device).squeeze()
        # print(target.squeeze())
        optimizer.zero_grad()
        output = model(Data)
        # print(output.shape)
        loss_curr = loss(output, target.long())
        loss_curr.backward()
        optimizer.step()
        pred = output.argmax(1)
        correct += pred.eq(target.view_as(pred)).sum().item()
        if batch_idx % 80 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(Data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss_curr.item()))
    print('\nTrain set: Accuracy: {}/{} ({:.2f}%)'.format(
        correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))


def test(model, test_loader, loss):
    model.eval()
    correct = 0
    total_test_loss = 0
    with torch.no_grad():
        for Data, target in test_loader:
            Data = Data.to(device)
            # print(target.shape)
            target = target.to(device).squeeze().cpu()
            # print(target.shape)
            output = model(Data).cpu()
            loss_curr = loss(output, target.long()).item()
            total_test_loss = total_test_loss + loss_curr
            # print(output.shape)
            pred = output.argmax(1)  # get the index of the max log-probability
            # print(pred.shape)
            # print(pred.eq(target.view_as(pred)).sum().item())
            correct += pred.eq(target.view_as(pred)).sum().item()
    accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Accuracy: {}/{} ({:.2f}%)\n'.format(
        correct, len(test_loader.dataset),
        accuracy))
    return accuracy


dataset = sio.loadmat('./data/Data0-10/Amplitude_Data_AGC_BN_Fre.mat')
X = dataset['data']
Y = dataset['label']
skf = KFold(n_splits=10, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, X)):
    print('**' * 10, '第', fold + 1, '折', 'ing....', '**' * 10)
    # print(val_idx)
    # for循环得到每一折的训练索引和验证索引，就可以对数据集进行抽取了。
    # 抽取完之后，我们得到了训练数据和验证数据，那就分别转成torch的Dataset形式
    # 然后再分别加载进torch的Dataloader里即可。

    # 假设原数据集df存在Dataframe里，这里取出训练集和验证集
    train_data = X[train_idx]
    train_label = Y[train_idx]
    # print(df_train.shape)
    val_data = X[val_idx]
    val_label = Y[val_idx]
    train_dataset = LoadDataset(train_data, train_label)
    val_dataset = LoadDataset(val_data, val_label)

    # 假设我们已经得到了训练数据的loader和验证数据的loader
    train_data_loader = DataLoader(train_dataset, batch_size=16, drop_last=True, shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size=16, drop_last=True, shuffle=True)

    model = Net()
    model = model.to(device)
    loss = nn.CrossEntropyLoss()
    loss = loss.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    best_score, accuracy, epochs = 0, 0, 0

    for epoch in range(20):
        train(model, train_data_loader, loss, optimizer, epoch)
        accuracy = test(model, val_data_loader, loss)
        if accuracy > best_score:  # save best model
            best_score = accuracy
            epochs = epoch
        print('Best_Accuracy={:.2f}% Epoch:{}\n'.format(best_score, epochs))
        scheduler.step()
    history['Best_Accuracy'].append(best_score)
Best_Accuracy = np.mean(history['Best_Accuracy'])
print(history['Best_Accuracy'],'\n',Best_Accuracy)
