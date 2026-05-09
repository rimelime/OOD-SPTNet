from torchvision.datasets import SVHN, DTD, LSUN, Places365, ImageFolder
from torch.utils.data import Subset, Dataset
from copy import deepcopy
import numpy as np
import os
from PIL import Image
from config import svhn_root, dtd_root, lsun_root, isun_root, places365_root, imagenet_crop_root, imagenet_resize_root

class CustomSVHN(SVHN):
    def __init__(self, offset=0, ood_index=None, *args, **kwargs):
        super(CustomSVHN, self).__init__(*args, **kwargs)
        self.ood_label = ood_index
        self.uq_idxs = np.arange(len(self)) + offset
    def __getitem__(self, item):
        img, label = super().__getitem__(item)
        if self.ood_label is not None:
            label = self.ood_label
        uq_idx = self.uq_idxs[item]
        return img, label, uq_idx
    def __len__(self):
        return len(self.labels)
    
class CustomDTD(DTD):
    def __init__(self, offset=0, ood_index=None, *args, **kwargs):
        super(CustomDTD, self).__init__(*args, **kwargs)
        self.ood_label = ood_index
        self.uq_idxs = np.arange(len(self)) + offset
    def __getitem__(self, item):
        img, label = super().__getitem__(item)
        if self.ood_label is not None:
            label = self.ood_label
        uq_idx = self.uq_idxs[item]
        return img, label, uq_idx
    def __len__(self):
        return len(self._labels)

class ImageDataset(Dataset):
    def __init__(self, datadir, transform=None, offset=0, ood_index=0):
        self.ood_label = ood_index
        self.transform = transform
        data = []
        for file_name in os.listdir(datadir):
            img = Image.open(os.path.join(datadir,file_name)).convert('RGB')
            if self.transform:
                img = self.transform(img)
            data.append(img)
        self.data = data
        self.uq_idxs = np.arange(len(self)) + offset
    def __getitem__(self, item):
        label = self.ood_label
        uq_idx = self.uq_idxs[item]
        img = self.data[item]
        return img, label, uq_idx
    def __len__(self):
        return len(self.data)
    
class CustomPlaces365(Places365):
    def __init__(self, offset=0, ood_index=None, *args, **kwargs):
        super(CustomPlaces365, self).__init__(*args, **kwargs)
        self.ood_label = ood_index
        self.uq_idxs = np.arange(len(self)) + offset
    def __getitem__(self, item):
        img, label = super().__getitem__(item)
        if self.ood_label is not None:
            label = self.ood_label
        uq_idx = self.uq_idxs[item]
        return img, label, uq_idx
    def __len__(self):
        return len(self.targets)
    
class CustomPlaces365_v2(Dataset):
    def __init__(self, datadir, transform=None,maxdata=10000, offset=0, ood_index=0):
        self.ood_label = ood_index
        self.transform = transform
        data = []
        for root, _, files in os.walk(os.path.join(datadir, "val_large")):
            for file in files:
                if len(data) >= maxdata: 
                    break
                if file.endswith('.jpg'):
                    img = Image.open(os.path.join(root, file)).convert('RGB')
                    if self.transform:
                        img = self.transform(img)
                    data.append(img)  # Label is set to 0 for all OOD samples
        self.samples = data
        self.uq_idxs = np.arange(len(self)) + offset
    def __getitem__(self, item):
        img = self.samples[item]
        label = self.ood_label
        uq_idx = self.uq_idxs[item]
        return img, label, uq_idx
    def __len__(self):
        return len(self.samples)

def getood_datasets(ood_transform, ood_index, offset, args, sample_num=10000):
    if args.ood_dataset_name == 'svhn':
        ood_dataset = CustomSVHN(root=svhn_root, split='test', download=True, transform=ood_transform, offset=offset, ood_index=ood_index)
        ood_dataset = Subset(ood_dataset, np.random.choice(len(ood_dataset), sample_num, replace=False))
    elif args.ood_dataset_name == 'dtd':
        ood_dataset = CustomDTD(root=dtd_root, split='train', download=True, transform=ood_transform, offset=offset, ood_index=ood_index)
    elif args.ood_dataset_name == 'lsun':
        ood_dataset = ImageDataset(lsun_root, ood_transform, offset, ood_index)
    elif args.ood_dataset_name == 'isun':
        ood_dataset = ImageDataset(isun_root, ood_transform, offset, ood_index)
    elif args.ood_dataset_name == 'imagenet_C':
        # ood_dataset = ImageFolder(imagenet_crop_root, transform=ood_transform)
        ood_dataset = ImageDataset(imagenet_crop_root, ood_transform, offset, ood_index)
    elif args.ood_dataset_name == 'imagenet_r':
        ood_dataset = ImageDataset(imagenet_resize_root, ood_transform, offset, ood_index)
    elif args.ood_dataset_name == 'places365':
        ood_dataset = CustomPlaces365_v2(root=places365_root, transform=ood_transform, maxdata=10000, offset=offset, ood_index=ood_index)
    else:
        raise ValueError(f"Invalid OOD dataset name: {args.ood_dataset_name}")
    return ood_dataset