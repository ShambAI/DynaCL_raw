import torch
import numpy as np
from torch.utils.data import Dataset
import pickle
import src.utils
import os
from tqdm import tqdm
from scipy.stats import mode
import pandas as pd

def load_pickle_file(filepath):
    with open(filepath, 'rb') as file:
        data = pickle.load(file)
    return data

class SleepDataset(Dataset):
    def __init__(self, data_path, is_train=True):
        """
        Args:
            data_path (str): Path to the dataset folder.
            dataset_name (str): Name of the dataset (e.g., 'ECGFiveDays').
            is_train (bool): Whether to load the training or testing data.
        """
        file_suffix = "train.pt" if is_train else "test.pt"
        file_path = os.path.join(data_path, file_suffix)
        
        # Load data using torch
        train = torch.load(file_path)
        
        # Extract labels and features
        labels = np.array(train['labels'])
        features = np.array(train['samples']).transpose(0, 2, 1)
        
        # Normalize labels to range {0, ..., L-1}
        unique_labels = np.unique(labels)
        label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        self.labels = np.vectorize(label_mapping.get)(labels)
        self.data = features
        
        mean = np.nanmean(features)
        std = np.nanstd(features)
        self.data = (features - mean) / std
        

        # Convert to PyTorch tensors
        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        
    def __len__(self):
        """Return the size of the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieve an item by index.
        Args:
            idx (int): Index of the item to retrieve.
        Returns:
            tuple: (data, label) for the given index.
        """
        return self.data[idx], self.labels[idx]
    
class ECGDataset(Dataset):
    def __init__(self, data_path, is_train=True):
        """
        Args:
            data_path (str): Path to the dataset folder.
            dataset_name (str): Name of the dataset (e.g., 'ECGFiveDays').
            is_train (bool): Whether to load the training or testing data.
        """
        feature_suffix = "x_train.pkl" if is_train else "x_test.pkl"
        feature_path = os.path.join(data_path, feature_suffix)
        
        label_suffix = "y_train.pkl" if is_train else "y_test.pkl"
        label_path = os.path.join(data_path, label_suffix)
        
        # Load data using torch
        features = load_pickle_file(feature_path)
        labels = load_pickle_file(label_path) 
        
        # Normalize labels to range {0, ..., L-1}
        unique_labels = np.unique(labels)
        label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        self.labels = np.vectorize(label_mapping.get)(labels)
        self.data = features
        
        mean = np.nanmean(features)
        std = np.nanstd(features)
        self.data = (features - mean) / std
        

        # Convert to PyTorch tensors
        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        """Return the size of the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieve an item by index.
        Args:
            idx (int): Index of the item to retrieve.
        Returns:
            tuple: (data, label) for the given index.
        """
        return self.data[idx], self.labels[idx]
    

class HarDataset(Dataset):
    def __init__(self, data_path, window_size, is_train=True):
        
        data, label = self.read_files(data_path, is_train)
        
        remaining = len(data) % window_size
        if remaining > 0:
            data = data[:-remaining]
            label = label[:-remaining]
            
          
        mean = np.nanmean(data)
        std = np.nanstd(data)
        
        data = (data - mean) / std
        
        self.final_data = torch.reshape(data, (len(data)//window_size, window_size, -1))
        
        final_label = torch.reshape(label, (len(label)//window_size, window_size, -1))
        
        self.final_label = mode(final_label, axis=1).mode.squeeze()
        
        
    def __len__(self):
        return len(self.final_data)

    def __getitem__(self, index):
        
        return self.final_data[index], self.final_label[index]
    
    def read_files(self, file_path, is_train):
        
        """ Reads all csv files in a given path"""
        
        data = []
        label = []
        filenames = [x for x in os.listdir(file_path)]
        
        filenames = filenames[:12] if is_train else filenames[12:]

        for fn in tqdm(filenames):
            df = pd.read_csv(
                os.path.join(file_path, fn),
            )

            x = torch.tensor(df.iloc[:, 1:7].values,dtype=torch.float32)
            y = torch.tensor(df.iloc[:, 7].values,dtype=torch.int)

            data.append(x)
            label.append(y)

        new_data = torch.cat(data, dim=0).squeeze()
        new_data = new_data.reshape(-1, new_data.shape[0]).T
        
        new_label = torch.cat(label, dim=0).squeeze()
        new_label = new_label.reshape(-1, new_label.shape[0]).T


        return new_data, new_label