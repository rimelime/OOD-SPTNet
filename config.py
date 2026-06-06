# -----------------
# DATASET ROOTS
# -----------------
cifar_10_root = './dataset/cifar10'
cifar_100_root = './dataset/cifar100'
herbarium_dataroot = '/disk/datasets/herbarium_19'
imagenet_root = './dataset/imagenet100'#ILSVRC12'
svhn_root = './dataset/svhn'#ILSVRC12'
dtd_root = './dataset/dtd'
lsun_root = './dataset/LSUN'
isun_root = './dataset/iSUN'
imagenet_crop_root = './dataset/Imagenet_crop'
imagenet_resize_root = './dataset/Imagenet_resize'
places365_root = './dataset/places365'
Flowers102_root = './dataset/OXFlowers102'
OxfordIIITPet_root = './dataset/OxfordIIITPet'

center_path = './dataset/class_centers.pt'
avg_conf_path = './dataset/avg_conf.pt'

cub_root = '/disk/datasets/ood_zoo/ood_data/CUB'

aircraft_root = './dataset/fgvc-aircraft-2013b'#'/disk/datasets/ood_zoo/ood_data/aircraft/fgvc-aircraft-2013b'

car_root = './dataset/stanford_cars'#'/disk/datasets/ood_zoo/ood_data/stanford_car'
scars_meta_path = "./dataset/stanford_cars/devkit/cars_{}.mat"

# OSR Split dir
osr_split_dir = './data/ssb_splits'#open_set_splits'

# -----------------
# OTHER PATHS
# -----------------
dino_pretrain_path = '/disk/work/hjwang/pretrained_models/dino/dino_vitbase16_pretrain.pth' 
clip_pretrain_path = '/disk/work/hjwang/pretrained_models/clip/ViT-B-16.pt' 
feature_extract_dir = '/disk/work/hjwang/gcd/extracted_features_public_impl'     # Extract features to this directory

cifar_mean = [0.4914, 0.4822, 0.4465]
cifar_std = [0.2470, 0.2435, 0.2616]