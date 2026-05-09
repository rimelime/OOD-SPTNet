from torchvision import transforms

import torch

def get_transform(transform_type='imagenet', image_size=32, args=None):

    if transform_type == 'imagenet':

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        interpolation = args.interpolation
        crop_pct = args.crop_pct

        train_transform = transforms.Compose([
            transforms.Resize(int(image_size / crop_pct), interpolation),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])
        ood_transform = transforms.Compose([
            transforms.Resize(int(image_size / crop_pct), interpolation),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 5.0))], p=0.7),
            transforms.RandomApply([transforms.Resize(image_size // 2),transforms.Resize(image_size)], p=0.5),
            transforms.RandomApply([transforms.RandomAffine(30, translate=(0.2, 0.2), scale=(0.5, 1.5))], p=0.5),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.5),
            transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
        ])

        test_transform = transforms.Compose([
            transforms.Resize(int(image_size / crop_pct), interpolation),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])
    
    else:

        raise NotImplementedError

    return (train_transform, ood_transform, test_transform)