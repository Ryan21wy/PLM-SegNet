import torch.utils.data as data
import torch
import os
import numpy as np

class Mydataset(data.Dataset):
    def __init__(self, filepath):
        '''
        dataset for BRNet training, validating and testing. 

        Parameters
        ----------
        filepath : str
            the filepath of cropped patches.

        Returns
        -------
        None.

        '''
        self.filepath = filepath
        dir_names = os.listdir(self.filepath)
        total_num = int(len(dir_names))
        half_num = total_num // 2
        self.x = dir_names[0: half_num]
        self.y = dir_names[half_num: total_num]
            
    def __getitem__(self, index):
        data_x_path = self.filepath + "\\" + self.x[index]
        data_y_path = self.filepath + "\\" + self.y[index]
        data_x = np.load(data_x_path)
        data_y = np.load(data_y_path)
        data_x = data_x/data_x.max()
        inputs = torch.from_numpy(data_x)
        labels = torch.from_numpy(data_y)
        return inputs, labels

    def __len__(self):
        return int(len(self.x))
