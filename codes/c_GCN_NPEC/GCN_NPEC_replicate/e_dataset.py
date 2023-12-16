import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

class MyDataloader():
    def __init__(self, file_name, batch_size):
        self.data = np.load(file_name)
        self.batch_size = batch_size

    def load_data(self, data, shuffle=True):
        dimension, capacity, location, distance, demand, reward, load_time, time_limit =\
            (data[i] for i in data.files)
        dataset = TensorDataset(torch.LongTensor(dimension), torch.LongTensor(capacity), torch.FloatTensor(location), torch.FloatTensor(distance), torch.FloatTensor(demand), torch.FloatTensor(reward), torch.FloatTensor(load_time), torch.LongTensor(time_limit))
        dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=shuffle, drop_last=True)
        return dataloader

    def dataloader(self):
        data_loader = self.load_data(self.data)
        return data_loader
    
class MyDataloader2():
    def __init__(self, file_name, batch_size):
        self.data = np.load(file_name)
        self.batch_size = batch_size

    def load_data(self, data, shuffle=True):
        dimension, capacity, location, distance, demand, or_tools_sol, or_tools_obj =\
            (data[i] for i in data.files)
        dataset = TensorDataset(torch.LongTensor(dimension), torch.LongTensor(capacity), torch.FloatTensor(location), torch.FloatTensor(distance), torch.FloatTensor(demand), torch.LongTensor(or_tools_sol), torch.FloatTensor(or_tools_obj))
        dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=shuffle, drop_last=True)
        return dataloader

    def dataloader(self, shuffle=True):
        data_loader = self.load_data(self.data, shuffle=shuffle)
        return data_loader