import torch
import os
from torch.utils.data import Dataset, DataLoader, random_split
from models.model import *
from data_loader import *
import numpy as np
import scipy.io as sio
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

full_dataset = LoadDataset('./data/Data0-10/Amplitude_Data_AGC_BN_Fre.mat')
train_size = int(len(full_dataset) * 0.9)
test_size = len(full_dataset) - train_size
# print(train_size,test_size)
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size], torch.manual_seed(0))
# print(train_dataset)

train_loader = DataLoader(train_dataset, batch_size=16, drop_last=True, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, drop_last=True, shuffle=True)


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
        if batch_idx % 40 == 0:
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
    pred_list,target_list = [],[]
    with torch.no_grad():
        for Data, target in test_loader:
            Data = Data.to(device)
            target = target.to(device).squeeze().cpu()
            # print(target.shape)
            target_list = np.append(target_list,target.numpy())
            # print(target.shape)
            output = model(Data).cpu()
            # loss_curr = loss(output, target.long()).item()
            # total_test_loss = total_test_loss + loss_curr
            # print(output.shape)
            pred = output.argmax(1)  # get the index of the max log-probability
            # print(pred.shape)
            # print(pred.eq(target.view_as(pred)).sum().item())
            pred_list = np.append(pred_list,pred.numpy())
            correct += pred.eq(target.view_as(pred)).sum().item()
    accuracy = 100. * correct / len(test_loader.dataset)
    # print(pred_list.shape)
    print('\nTest set: Accuracy: {}/{} ({:.2f}%)\n'.format(correct, len(test_loader.dataset),accuracy))
    sio.savemat('./results/result.mat', {'prediction': pred_list,'truth': target_list})
    return accuracy


model = Net()
model = model.to(device)
loss = nn.CrossEntropyLoss()
loss = loss.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
best_score, accuracy, epochs = 0, 0, 0

# 训练模型
for epoch in range(20):
    train(model, train_loader, loss, optimizer, epoch)
    accuracy = test(model, test_loader, loss)

    if accuracy > best_score:  # save best model
        best_score = accuracy
        epochs = epoch
        # torch.save(model, 'checkpoints/best.pth')
    print('Best_Accuracy={:.2f}% Epoch:{}\n'.format(best_score, epochs))
    scheduler.step()

# accuracy = test(model, test_loader, loss)
# torch.save(model, 'checkpoints/latest_' + str(round(accuracy, 2)) + '%' + '.pth')
