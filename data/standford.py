from torchvision.datasets import OxfordIIITPet, Flowers102
from config import OxfordIIITPet_root
import numpy as np

class CustomOXFORDPET(OxfordIIITPet):

    def __init__(self, ood_index=None, *args, **kwargs):

        super(CustomOXFORDPET, self).__init__(*args, **kwargs)
        self.uq_idxs = np.array(range(len(self)))
        self.ood_label = ood_index

    def __getitem__(self, item):
        img, label = super().__getitem__(item)
        if self.ood_label is not None:
            label = self.ood_label
        uq_idx = self.uq_idxs[item]
        return img, label, uq_idx

def get_standford_Pet_datasets(train_transform, test_transform, ood_transform, train_classes=(0, 1, 8, 9),
                       prop_train_labels=0.8, split_train_val=False, seed=0):
    
    train_dataset = CustomOXFORDPET(root=OxfordIIITPet_root, split="trainval", transform=train_transform, download=True)
    test_dataset = CustomOXFORDPET(root=OxfordIIITPet_root, split="test", transform=test_transform, download=True)
    all_datasets = {
        'train': train_dataset,
        'ood_train': None,
        'test': test_dataset,
    }

    return all_datasets