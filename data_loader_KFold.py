from torch.utils.data import Dataset
import scipy.io as sio
import torch


class LoadDataset(Dataset):
    def __init__(self, dataset, label):
        super(LoadDataset, self).__init__()

        X = dataset
        Y = label

        # print(X.shape[1])
        # self.x_data = torch.from_numpy(X).reshape(X.shape[0], 1, X.shape[1]).to(torch.float32)
        self.x_data = torch.from_numpy(X).to(torch.float32)
        # print(self.x_data.shape)
        self.y_data = torch.from_numpy(Y).reshape(Y.shape[0], Y.shape[1]).to(torch.float32)
        self.len = self.x_data.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len
