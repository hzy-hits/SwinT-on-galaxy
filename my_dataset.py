from torch.utils.data import Dataset
import pandas as pd
from astropy.io import fits
import numpy as np
import torch


class myDataset(Dataset):
    def __init__(self, csv_file, fits_list, path):
        self.image = fits_list
        self.csv_data = csv_file
        self.path = path

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        with fits.open(self.path + self.image[idx]) as f:
            #img=self.image[idx]
            img = f[0].data
            psf = f[1].data
        img = img.astype(np.float32)
        #img=np.where(img>0,img,1e-6)
        #img=np.log(img)
        
        psf = psf.astype(np.float32)
        #img=(img-img.min())/(img.max()-img.min())
        img = img.reshape(1, 64, 64)
        psf = psf.reshape(1, 25, 25)
        img = torch.from_numpy(img)
        psf = torch.from_numpy(psf)
        label = self.csv_data[idx, 1:]
        label = torch.from_numpy(label)
        label = label.reshape(-1)
        return img, psf, label
