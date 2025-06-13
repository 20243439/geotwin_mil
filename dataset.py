import os
import json
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
import pandas as pd

class CustomDataset(Dataset):
    def __init__(self, feature_dir, label_dir):
        self.feature_dir = feature_dir
        self.label_dir = label_dir
        self.file_names = [
            f.split('.')[0].split('_')[0] for f in os.listdir(feature_dir) if f.endswith('.pth')
        ]
    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        
        file_name = self.file_names[idx]
        feature_path = os.path.join(self.feature_dir, f"{file_name}.pth")
        label_path = os.path.join(self.label_dir, f"{file_name}.json")
        feature = torch.load(feature_path, weights_only=False)
        with open(label_path, 'r') as f:
            label_data = json.load(f)

        labels = label_data['normal_label']
        # if labels >1 :
        #     labels = 1
        # label_tensor = torch.tensor(labels, dtype=torch.float32)
        label_tensor = torch.tensor(list(labels.values()), dtype=torch.float32)

        labels = label_data['function_label']
        function_tensor = torch.tensor(list(labels.values()), dtype=torch.float32)
        
        return feature, label_tensor, function_tensor