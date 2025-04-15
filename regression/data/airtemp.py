import torch
import os
import os.path as osp
import xarray as xr
import numpy as np
from torch.utils.data import Dataset

# from utils.paths import datasets_path
ROOT = ''
datasets_path = os.path.join(ROOT, 'datasets')

os.makedirs(osp.join(datasets_path, 'air'), exist_ok=True)

class AirTemperatureDataset(Dataset):
    def __init__(self, train=True,split_ratio=0.8):
        """
        Custom Dataset class for Air Temperature data from xarray.
        
        Args:
        - train (bool): Load train or test data.
        - split_ratio (float): Fraction of data used for training (default: 80% train, 20% test).
        """
        ds = xr.tutorial.open_dataset('air_temperature')
        data = torch.tensor(ds.to_array().values[0])
        data = data.unsqueeze(1)
        split_point = int(len(data) * split_ratio)

        train_data = data[:split_point]
        test_data = data[split_point:]
        
        if train:
            self.data = train_data
        else:
            self.data = test_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index].float(),0
    
if __name__ == '__main__':
    import os
    import os.path as osp
    from PIL import Image
    from tqdm import tqdm
    import numpy as np
    import torch

    ds = xr.tutorial.open_dataset('air_temperature')
    data = ds.to_array()[0]

    train_data = data[:2064]
    test_data = data[2064:]

    train_imgs = np.stack(train_data)
    torch.save(train_imgs, osp.join(datasets_path, 'air', 'train.pt'))

    eval_imgs = np.stack(test_data)
    torch.save(eval_imgs, osp.join(datasets_path, 'air', 'eval.pt'))
