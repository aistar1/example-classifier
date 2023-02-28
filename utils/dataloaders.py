

import cv2
import torch
import os
from pathlib import Path
import torchvision

#from torch.utils.data import DataLoader, Dataset, dataloader, distributed
from torch.utils.data import Dataset, DataLoader
from utils.augmentations import classify_transforms


def create_dataloader(path,
                      img_height,
                      img_width,
                      batch_size,
                      workers):
    dataset = ClassificationDataset(path, img_height=img_height, img_width=img_width)
    batch_size = min(batch_size, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])  # number of workers
 
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=nw)
    return train_loader

class ClassificationDataset(torchvision.datasets.ImageFolder):
    def __init__(self, root,
                img_height=256,
                img_width=128):
        super().__init__(root=root)
        self.torch_transforms = classify_transforms(img_height, img_width)
        self.samples = [list(x) + [Path(x[0]).with_suffix('.npy'), None] for x in self.samples]  # file, index, npy, im
        '''
        self.dataset = ImageFolder(root)
        print(f'your classification: {self.dataset.class_to_idx}')
        print(f'total data: {len(self.dataset.imgs)}')
        print(self.dataset[0][0]) # image data
        print(self.dataset[0][1]) # label 
        # {'female': 0, 'male': 1}
        # [('gender/female/00005_023896.jpg', 0), ('gender/female/00006_027209.jpg', 1)]
        '''

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        f, label, fn, im = self.samples[index]  # filename, index, filename.with_suffix('.npy'), image
        # [['gender/female/00005_023896.jpg', 0, PosixPath('gender/female/00005_023896.npy'), None],[...],...]
        im = cv2.imread(f)  # BGR
        sample = self.torch_transforms(im) # torch_transforms only support BGR
        return sample, label

