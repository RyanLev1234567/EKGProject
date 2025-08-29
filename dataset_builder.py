from torch.utils.data import Dataset
import torch
from preprocess import normalize_signal, resample_signal

class ECGDataset(Dataset):
    def __init__(self, X_list, y_list, transform=None):
        self.X_list = X_list
        self.y_list = y_list
        self.transform = transform

    def __len__(self):
        return len(self.X_list)

    def __getitem__(self, idx):
        X, y = self.X_list[idx], self.y_list[idx]
        if self.transform:
            X = self.transform(X)
        # Convert y to torch tensor
        y = torch.tensor(y, dtype=torch.float)
        return X, y
