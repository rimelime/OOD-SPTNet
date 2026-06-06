import argparse
import os
import sys 
sys.path.append("../..") 

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from IPython.display import display
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

from models import vision_transformer as vits
from prompters import PadPrompter, PatchPrompter

from scoring_method import compute_class_centers, Mahalanobis_score
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, average_precision_score, roc_curve
from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits

from util.cluster_and_log_utils import log_accs_from_preds
from model import DINOHead

from config import clip_pretrain_path, dino_pretrain_path

parser = argparse.ArgumentParser(description='cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset_name', type=str, default='ox_pet')
parser.add_argument('--num_workers', default=0, type=int)
parser.add_argument('--use_bb', type=bool, default=True)
parser.add_argument('--transform', type=str, default='imagenet')
parser.add_argument('--prompt_type', type=str, default='all')
parser.add_argument('--pretrained_model_path', type=str)
parser.add_argument("--image_path", required=True, default='./image_demo')

args = parser.parse_args()
device = torch.device('cuda')
args.ood_dataset_name = None
args = get_class_splits(args)

def cal_thresshold (model, data_loader, args, centers=None, global_mean=None, global_std=None, inv_cov=None):
    score = []
    for images, _, _ in tqdm(data_loader):
        images = images.cuda(non_blocking=True).requires_grad_(True)
        bb_feat, pro_feat, _ = model(images)
        if args.use_bb:
            features = bb_feat
        else:
            features = pro_feat
        with torch.no_grad():
            Mahalanobis = Mahalanobis_score(features, centers, inv_cov, global_mean, global_std, normalize=True, eps=1e-8).min(dim=1)[0].cpu().numpy()
            score.extend(Mahalanobis)
    score = np.array(score)
    threshold_95 = np.percentile(score, 5)
    return threshold_95


if __name__ == "__main__":
    args.interpolation = 3
    args.crop_pct = 0.875
    args.image_size = 224
    args.feat_dim = 384
    args.proj_dim = 256
    args.num_mlp_layers = 3
    args.patch_size = 16
    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)
    args.num_ctgs = args.num_labeled_classes 

    backbone = vits.__dict__['vit_small14_dinov3']().to(device)
    if args.prompt_type == 'patch':
        args.prompt_size = 1
        prompter = PatchPrompter(args)

    elif args.prompt_type == 'all':
        args.prompt_size = 30
        prompter1 = PadPrompter(args)
        args.prompt_size = 1
        prompter2 = PatchPrompter(args)
        prompter = nn.Sequential(prompter1, prompter2)

    projector = DINOHead(in_dim=args.feat_dim, use_bb=args.use_bb, out_dim=args.num_ctgs, nlayers=args.num_mlp_layers)
    classifier = nn.Sequential(backbone, projector).cuda()
    model = nn.Sequential(prompter, classifier).cuda()

    for p in model.parameters():
        p.requires_grad = False

    model.eval()
    model.cuda()

    state_dict = torch.load(args.pretrained_model_path, map_location="cpu",weights_only=False)
    train_transform, ood_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    train_dataset, _, _ = get_datasets(args.dataset_name, train_transform, ood_transform, test_transform, args)
    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=128, shuffle=False,
                            sampler=None, drop_last=True, pin_memory=True)
    class_names = train_dataset.classes

    if not all(k in state_dict for k in ["model_state_dict","centers","avg_conf","global_mean","global_std","inv_cov"]):
        model.load_state_dict(state_dict)
        centers, avg_conf, global_mean, global_std, inv_cov = compute_class_centers(
            model, 
            train_loader, 
            num_classes=args.num_ctgs, 
            temperature=0.1, 
            eps=1e-8,
            normalize=True,
            use_bb=args.use_bb,
            device=device)
    else:
        model.load_state_dict(state_dict["model_state_dict"],strict=True)
        centers = state_dict["centers"].to(device)
        avg_conf = state_dict["avg_conf"]
        global_mean = state_dict["global_mean"].to(device)
        global_std = state_dict["global_std"].to(device)
        inv_cov = state_dict["inv_cov"].to(device)

    if "thresshold95" not in state_dict: 
        thresshold95 = cal_thresshold(model,train_loader, args, centers, global_mean, global_std, inv_cov)
    else:
        thresshold95 = state_dict["thresshold95"]
    if not all(k in state_dict for k in ["model_state_dict","centers","avg_conf","global_mean","global_std","inv_cov","thresshold95"]):
        checkpoint = {
                "model_state_dict": state_dict["model_state_dict"] if "model_state_dict" in state_dict else state_dict,
                "centers": centers,
                "avg_conf": avg_conf,
                "global_mean": global_mean,
                "global_std": global_std,
                "inv_cov": inv_cov,
                "thresshold95": thresshold95
        }
        torch.save(checkpoint, args.pretrained_model_path)
    if not os.path.exists("./output"): os.makedirs("./output")
    if not os.path.isfile(args.image_path):
        for file in os.listdir(args.image_path):
            file_path = os.path.join(args.image_path,file)
            try:
                img = Image.open(file_path).convert("RGB")
                image = test_transform(img).unsqueeze(0).to(device)
                model.eval()
                with torch.no_grad():
                    bb_feat, pro_feat, logits = model(image)
                if args.use_bb:
                    features = bb_feat
                else:
                    features = pro_feat
                _, preds = torch.max(logits,1)
                score = Mahalanobis_score(features, centers, inv_cov, global_mean, global_std, normalize=True, eps=1e-8).min(dim=1)[0].cpu().numpy()
                plt.figure(figsize=(6, 6))
                plt.imshow(img)
                file_name = ext = os.path.splitext(file)[-2]
                plt.title(f"image {file} Prediction: {class_names[preds.item()] if score > thresshold95 else 'OOD'}, {score.item():.4f}, {thresshold95.item():.4f}")
                plt.axis("off")
                plt.savefig(f"output/result_{file_name}", bbox_inches="tight")
                plt.close()
            except Exception as e:
                print(f"Error processing {file}: {e}")
    else:
        try:
            img = Image.open(args.image_path).convert("RGB")
            image = test_transform(img).unsqueeze(0).to(device)
            model.eval()
            with torch.no_grad():
                bb_feat, pro_feat, logits = model(image)
            if args.use_bb:
                features = bb_feat
            else:
                features = pro_feat
            _, preds = torch.max(logits,1)
            score = Mahalanobis_score(features, centers, inv_cov, global_mean, global_std, normalize=True, eps=1e-8).min(dim=1)[0].cpu().numpy()
            plt.figure(figsize=(6, 6))
            plt.imshow(img)
            file_name = ext = os.path.splitext(args.image_path)[-2]
            plt.title(f"image {args.image_path} Prediction: {class_names[preds.item()] if score > thresshold95 else 'OOD'}, {score.item():.4f}, {thresshold95.item():.4f}")
            plt.axis("off")
            plt.savefig(f"output/result_{file_name}", bbox_inches="tight")
            plt.close()
        except:
            raise ValueError(f"file {args.image_path} is invalid image file")