"""
Data loading and preprocessing module for multivariate time series anomaly detection.
Supports SMAP, MSL, and SMD datasets.
"""

import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader


class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series data."""
    
    def __init__(self, data, labels=None, window_size=100, stride=1):
        """
        Args:
            data: numpy array of shape (n_samples, n_features)
            labels: numpy array of shape (n_samples,) with anomaly labels
            window_size: size of sliding window
            stride: step size for sliding window
        """
        self.window_size = window_size
        self.stride = stride
        
        # Create sliding windows
        self.windows = []
        self.window_labels = []
        
        for i in range(0, len(data) - window_size + 1, stride):
            window = data[i:i + window_size]
            self.windows.append(window)
            
            if labels is not None:
                # Label is anomalous if any point in window is anomalous
                window_label = 1 if np.any(labels[i:i + window_size] == 1) else 0
                self.window_labels.append(window_label)
            else:
                self.window_labels.append(0)
        
        self.windows = np.array(self.windows)
        self.window_labels = np.array(self.window_labels)
        
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        window = torch.FloatTensor(self.windows[idx])
        label = torch.LongTensor([self.window_labels[idx]])[0]
        return window, label


def load_smap_data(data_dir='data'):
    """Load SMAP dataset."""
    train_path = os.path.join(data_dir, 'SMAP', 'train')
    test_path = os.path.join(data_dir, 'SMAP', 'test')
    label_path = os.path.join(data_dir, 'SMAP', 'test_label')
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"SMAP dataset not found at {data_dir}. Please download from https://github.com/elisejiuqizhang/TS-AD-Datasets")
    
    # Load all channels
    train_data = []
    test_data = []
    test_labels = []
    
    for file in os.listdir(train_path):
        if file.endswith('.txt'):
            channel_name = file.replace('.txt', '')
            
            train_file = os.path.join(train_path, file)
            test_file = os.path.join(test_path, file)
            label_file = os.path.join(label_path, file)
            
            if os.path.exists(test_file) and os.path.exists(label_file):
                train_channel = pd.read_csv(train_file, header=None).values
                test_channel = pd.read_csv(test_file, header=None).values
                label_channel = pd.read_csv(label_file, header=None).values.flatten()
                
                train_data.append(train_channel)
                test_data.append(test_channel)
                test_labels.append(label_channel)
    
    # Concatenate all channels
    train_data = np.concatenate(train_data, axis=1)
    test_data = np.concatenate(test_data, axis=1)
    test_labels = np.concatenate(test_labels)
    
    return train_data, test_data, test_labels


def load_msl_data(data_dir='data'):
    """Load MSL dataset."""
    train_path = os.path.join(data_dir, 'MSL', 'train')
    test_path = os.path.join(data_dir, 'MSL', 'test')
    label_path = os.path.join(data_dir, 'MSL', 'test_label')
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"MSL dataset not found at {data_dir}. Please download from https://github.com/elisejiuqizhang/TS-AD-Datasets")
    
    train_data = []
    test_data = []
    test_labels = []
    
    for file in os.listdir(train_path):
        if file.endswith('.txt'):
            train_file = os.path.join(train_path, file)
            test_file = os.path.join(test_path, file)
            label_file = os.path.join(label_path, file)
            
            if os.path.exists(test_file) and os.path.exists(label_file):
                train_channel = pd.read_csv(train_file, header=None).values
                test_channel = pd.read_csv(test_file, header=None).values
                label_channel = pd.read_csv(label_file, header=None).values.flatten()
                
                train_data.append(train_channel)
                test_data.append(test_channel)
                test_labels.append(label_channel)
    
    train_data = np.concatenate(train_data, axis=1)
    test_data = np.concatenate(test_data, axis=1)
    test_labels = np.concatenate(test_labels)
    
    return train_data, test_data, test_labels


def load_smd_data(data_dir='data', machine_id='machine-1-1'):
    """Load SMD dataset for a specific machine."""
    train_path = os.path.join(data_dir, 'SMD', 'train', f'{machine_id}.txt')
    test_path = os.path.join(data_dir, 'SMD', 'test', f'{machine_id}.txt')
    label_path = os.path.join(data_dir, 'SMD', 'test_label', f'{machine_id}.txt')
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"SMD dataset not found at {data_dir}. Please download from https://github.com/elisejiuqizhang/TS-AD-Datasets")
    
    train_data = pd.read_csv(train_path, header=None).values
    test_data = pd.read_csv(test_path, header=None).values
    test_labels = pd.read_csv(label_path, header=None).values.flatten()
    
    return train_data, test_data, test_labels


def create_synthetic_data(n_samples=10000, n_features=10, anomaly_ratio=0.1):
    """Create synthetic multivariate time series data for testing."""
    np.random.seed(42)
    
    # Generate normal data with some temporal correlation
    normal_data = np.random.randn(n_samples, n_features)
    for i in range(1, n_samples):
        normal_data[i] = 0.7 * normal_data[i-1] + 0.3 * normal_data[i]
    
    # Generate labels
    n_anomalies = int(n_samples * anomaly_ratio)
    labels = np.zeros(n_samples)
    anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
    labels[anomaly_indices] = 1
    
    # Add anomalies
    data = normal_data.copy()
    for idx in anomaly_indices:
        data[idx] += np.random.randn(n_features) * 3  # Large deviation
    
    # Split train/test
    train_size = int(0.7 * n_samples)
    train_data = data[:train_size]
    test_data = data[train_size:]
    train_labels = labels[:train_size]
    test_labels = labels[train_size:]
    
    return train_data, test_data, train_labels, test_labels


def preprocess_data(train_data, test_data, normalize=True):
    """Preprocess time series data."""
    if normalize:
        scaler = StandardScaler()
        train_data = scaler.fit_transform(train_data)
        test_data = scaler.transform(test_data)
        return train_data, test_data, scaler
    return train_data, test_data, None


def create_dataloaders(train_data, test_data, train_labels=None, test_labels=None,
                      window_size=100, batch_size=32, stride=1, normalize=True):
    """Create PyTorch DataLoaders for training and testing."""
    
    # Preprocess data
    train_data, test_data, scaler = preprocess_data(train_data, test_data, normalize)
    
    # Create datasets
    train_dataset = TimeSeriesDataset(train_data, train_labels, window_size, stride)
    test_dataset = TimeSeriesDataset(test_data, test_labels, window_size, stride)
    
    # Create dataloaders with GPU optimizations
    pin_memory = torch.cuda.is_available()  # Faster data transfer to GPU
    num_workers = 4 if torch.cuda.is_available() else 0  # Parallel data loading for GPU
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        pin_memory=pin_memory,
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=num_workers
    )
    
    return train_loader, test_loader, scaler

