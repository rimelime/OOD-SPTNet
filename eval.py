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
from tqdm import tqdm

from models import vision_transformer as vits
from prompters import PadPrompter, PatchPrompter

from scoring_method import compute_class_centers, confident_score, cadref_score, cadref_score_2, odin_preprocess, energy_score, odin_score, Mahalanobis_score
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, average_precision_score, roc_curve
from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits

from util.cluster_and_log_utils import log_accs_from_preds
from model import DINOHead

from config import clip_pretrain_path, dino_pretrain_path


parser = argparse.ArgumentParser(description='cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--num_workers', default=0, type=int)#default=8
parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v2', 'v2p'])

parser.add_argument('--dataset_name', type=str, default='scars', help='options: cifar10, cifar100, imagenet_100, cub, scars, fgvc_aricraft, herbarium_19')
parser.add_argument('--ood_dataset_name', type=str, default='svhn', help='options: svhn, dtd, lsun, places365')
parser.add_argument('--use_ssb_splits', action='store_true', default=True)
parser.add_argument('--use_bb', type=bool, default=True)

parser.add_argument('--transform', type=str, default='imagenet')
parser.add_argument('--prompt_type', type=str, default='all')
parser.add_argument('--pretrained_model_path', type=str)


# ----------------------
# INIT
# ----------------------
args = parser.parse_args()
device = torch.device('cuda')
args = get_class_splits(args)

def evaluate(method, preds, score, targets, args):
    ood_class = len(args.train_classes)
    y_in = (targets != ood_class).astype(int)
    if method in ['Energy Score', 'CADREF']:
        in_score = -score
    else:
        in_score = score
    auroc = roc_auc_score(y_in, in_score)
    aupr_in = average_precision_score(y_in, in_score)
    aupr_out = average_precision_score(1-y_in, -in_score)
    fpr, tpr, thresholds = roc_curve(y_in, in_score)
    idx = np.argmin(np.abs(tpr - 0.95))
    fpr95 = fpr[idx]

    n_in = np.sum(y_in == 1)
    n_ood = np.sum(y_in == 0)
    pi_in = n_in / len(y_in)
    pi_ood = n_ood / len(y_in)

    detection_error = np.min(pi_in * (1 - tpr) + pi_ood * fpr)
    preds[in_score < thresholds[idx]] = ood_class
    print(f'{method} AUROC: {auroc:.2f}, AUPR_IN: {aupr_in:.2f}, AUPR_OUT: {aupr_out:.2f}, FPR95: {fpr95:.2f}, Detection Error: {detection_error:.2f}')
    print("ID mean:", score[targets != ood_class].mean(), "OOD mean:", score[targets == ood_class].mean(), "accuracy:", accuracy_score(targets, preds))
    # print(classification_report(targets, preds, digits=4))
    return auroc, aupr_in, aupr_out, fpr95, detection_error

def test(model, test_loader, save_name, args, centers=None, avg_conf=None, global_mean=None, global_std=None, inv_cov=None):
    model.eval()
    preds, targets, cad_score, conf_score, odin_conf_score, e_score, m_score = [], [], [], [], [], [], []
    ood_class = len(args.train_classes)
    for images, label, _ in tqdm(test_loader):
        images = images.cuda(non_blocking=True).requires_grad_(True)
        odin_conf = odin_score(images, model).max(dim=1)[0].cpu().numpy()
        bb_feat, pro_feat, logits = model(images)
        if args.use_bb:
            features = bb_feat
        else:
            features = pro_feat
        with torch.no_grad():
            batch_preds = logits.argmax(dim=1).cpu().numpy()
            targets.append(label.cpu().numpy())
            cad = cadref_score_2(features, logits, temperature=10, 
                                    centers=centers, 
                                    classifier_weight= model[1][1].last_layer.weight, 
                                    avg_conf=avg_conf,
                                    eps=0.001).cpu().numpy()
            energy = energy_score(logits, temperature=1).cpu().numpy()
            conf = confident_score(logits, temperature=10).cpu().numpy()
            Mahalanobis = Mahalanobis_score(features, centers, inv_cov, global_mean, global_std, normalize=True, eps=1e-8).min(dim=1)[0].cpu().numpy()
            cad_score.extend(cad)
            preds.extend(batch_preds)
            conf_score.extend(conf)
            odin_conf_score.extend(odin_conf)
            e_score.extend(energy)
            m_score.extend(Mahalanobis)
    targets = np.concatenate(targets)
    cad_score = np.array(cad_score)
    conf_score = np.array(conf_score)
    odin_conf_score = np.array(odin_conf_score)
    e_score = np.array(e_score)
    m_score = np.array(m_score)
    preds = np.array(preds)

    evaluate('CADREF', preds.copy(), cad_score, targets, args)
    evaluate('Confident Score', preds.copy(), conf_score, targets, args)
    evaluate('Energy Score', preds.copy(), e_score, targets, args)
    evaluate('ODIN Score', preds.copy(), odin_conf_score, targets, args)
    evaluate('Mahalanobis Score', preds.copy(), m_score, targets, args)

if __name__ == "__main__":

    # print(f'Using evaluation function {args.eval_funcs[0]} to print results')
    
    torch.backends.cudnn.benchmark = True

    # ----------------------
    # Hyper-paramters
    # ----------------------
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

    # ----------------------
    # BASE MODEL
    # ----------------------
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

    # print(args)

    # ----------------------
    # CLS HEAD
    # ----------------------
    projector = DINOHead(in_dim=args.feat_dim, use_bb=args.use_bb, out_dim=args.num_ctgs, nlayers=args.num_mlp_layers)
    classifier = nn.Sequential(backbone, projector).cuda()
    model = nn.Sequential(prompter, classifier).cuda()

    # projector = DINOHead(in_dim=args.feat_dim, out_dim=args.num_ctgs, nlayers=args.num_mlp_layers)
    # state_dict = torch.load(args.pretrained_model_path, map_location='cpu')
    # backbone.load_state_dict(state_dict, strict=False)
    # classifier = nn.Sequential(backbone, projector).cuda()
    # # state_dict = torch.load(args.pretrained_model_path, map_location='cpu')
    # # classifier.load_state_dict(state_dict)
    # model = nn.Sequential(prompter, classifier).cuda()

    for p in model.parameters():
        p.requires_grad = False

    model.eval()
    model.cuda()

    state_dict = torch.load(args.pretrained_model_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)

    # DATASETS
    train_transform, ood_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    train_dataset, _, test_dataset = get_datasets(args.dataset_name, train_transform, ood_transform, test_transform, args)

    # ------------------
    # DATALOADERS
    # --------------------
    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False,
                              sampler=None, drop_last=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, num_workers=args.num_workers,
                                      batch_size=128, shuffle=False, pin_memory=False)
    
    # ----------------------
    # EVAL
    # ----------------------
    centers, avg_conf, global_mean, global_std, inv_cov = compute_class_centers(
        model, 
        train_loader, 
        num_classes=args.num_ctgs, 
        temperature=0.1, 
        eps=1e-8,
        normalize=True,
        use_bb=args.use_bb,
        device=device)
    
    test(model, test_loader,  
        centers=centers, 
        avg_conf=avg_conf, 
        global_mean=global_mean, 
        global_std=global_std, 
        inv_cov=inv_cov, 
        save_name='Eval ACC Unlabelled', 
        args=args)
