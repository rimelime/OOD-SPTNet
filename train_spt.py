import argparse
import os

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader, ConcatDataset

from tqdm import tqdm

from models import vision_transformer as vits
from prompters import PadPrompter, PatchPrompter
from scoring_method import compute_class_centers, confident_score, cadref_score, odin_preprocess, energy_score
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, average_precision_score, roc_curve

from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits

from util.general_utils import str2bool, get_params_groups, finetune_params, freeze, unfreeze, cosine_lr
from util.cluster_and_log_utils import log_accs_from_preds
from model import DINOHead, info_nce_logits, SupConLoss, DistillLoss, ContrastiveLearningViewGenerator

from config import clip_pretrain_path, dino_pretrain_path

parser = argparse.ArgumentParser(description='SPTNet', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v2', 'v2p'])

parser.add_argument('--warmup_model_dir', type=str, default=None)
parser.add_argument('--dataset_name', type=str, default='scars', help='options: cifar10, cifar100, imagenet_100, cub, scars, fgvc_aricraft, herbarium_19')
parser.add_argument('--ood_dataset_name', type=str, default='svhn', help='options: svhn, dtd, lsun, places365')
parser.add_argument('--prop_train_labels', type=float, default=0.5)
parser.add_argument('--use_ssb_splits', action='store_true', default=True)

parser.add_argument('--grad_from_block', type=int, default=11)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--lr2', type=float, default=0.1)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--transform', type=str, default='imagenet')
parser.add_argument('--sup_weight', type=float, default=0.35)
parser.add_argument('--n_views', default=2, type=int)
parser.add_argument('--lamb', type=float, default=0.1, help='The balance factor.')
parser.add_argument('--model', type=str, default='dino')
parser.add_argument('--model_path', type=str, default='./pretrained_models/dino/dino_deitsmall16_pretrain.pth')
parser.add_argument('--model_name', type=str, default='dinoB16_best_trainul.pt')

parser.add_argument('--freq_rep_learn', type=int)
parser.add_argument('--prompt_size', type=int)
parser.add_argument('--prompt_type', type=str, default='patch')
parser.add_argument('--pretrained_model_path', type=str)
parser.add_argument('--use_checkpoint', type=bool, default=False)

parser.add_argument('--memax_weight', type=float, default=2)
parser.add_argument('--ood_threshold', type=float, default=-1)
parser.add_argument('--warmup_teacher_temp', default=0.07, type=float, help='Initial value for the teacher temperature.')
parser.add_argument('--teacher_temp', default=0.04, type=float, help='Final value (after linear warmup)of the teacher temperature.')
parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int, help='Number of warmup epochs for the teacher temperature.')

parser.add_argument('--use_margin', type=bool, default=False)
parser.add_argument('--use_supconloss', type=bool, default=False)
parser.add_argument('--use_bb', type=bool, default=True)

parser.add_argument('--fp16', action='store_true', default=True)
parser.add_argument('--eval_freq', default=10, type=int)

# ----------------------
# INIT
# ----------------------
args = parser.parse_args()
device = torch.device('cuda')
args = get_class_splits(args)


def construct_gcd_loss(prompter, backbone, projector, images, class_labels, ood_images, epoch, args):
    if prompter is None:
        feats = backbone(images)
        _, student_proj, student_out = projector(feats)
        # ood_feats = backbone(ood_images)
        # _, ood_proj, ood_out = projector(ood_feats)
    else:
        feats = backbone(prompter(images))
        _, student_proj, student_out = projector(feats)
        # ood_feats = backbone(prompter(ood_images))
        # _, ood_proj, ood_out = projector(ood_feats)
    teacher_out = student_out.detach()

    cls_loss = nn.CrossEntropyLoss()(student_out, class_labels)

    if args.use_margin:
        logits = (student_out / 0.1)#.chunk(2)[0]
        correct_logits = logits.gather(1, class_labels.unsqueeze(1)).squeeze(1)
        mask = torch.ones_like(logits).scatter_(1, class_labels.unsqueeze(1), 0)
        wrong_logits = logits.masked_select(mask.bool()).view(logits.size(0), -1)
        max_wrong_logits, _ = wrong_logits.max(dim=1)
        loss = F.relu(1 - (correct_logits - max_wrong_logits))
        margin_loss = loss.mean()

    # energy = energy_score(student_out, temperature=1)
    # energy_ood = energy_score(ood_out, temperature=1)
    # energy = torch.relu(energy + 25).mean()
    # energy_ood = torch.relu(-5 - energy_ood).mean()
    # energy_loss = energy + energy_ood

    if args.use_supconloss:
        student_proj1 = torch.cat([f.unsqueeze(1) for f in student_proj.chunk(2)], dim=1)
        student_proj1 = F.normalize(student_proj1, dim=-1)
        sup_con_labels = class_labels.chunk(2)[0]
        sup_con_loss = SupConLoss()(student_proj1, labels=sup_con_labels)

    if args.use_margin and args.use_supconloss:
        loss = 0.4 * cls_loss + 0.3 * margin_loss + 0.3 * sup_con_loss
    elif args.use_margin:
        loss = 0.5 * cls_loss + 0.5 * margin_loss 
    elif args.use_supconloss:
        loss = 0.5 * cls_loss + 0.5 * sup_con_loss 
    else:
        loss = cls_loss

    return loss, feats, student_out


def train(prompter, backbone, projector, train_loader, ood_loader, optimizer, optimizer_cls, exp_lr_scheduler, exp_lr_scheduler_cls, cluster_criterion, epoch, args):

    prompter.train()
    backbone.train()
    projector.train()

    num_batches_per_epoch = len(train_loader)
    switch_to_cls = False

    for batch_idx, batch in enumerate(tqdm(train_loader)):

        if (batch_idx + 1) % args.freq_rep_learn == 0: # train classifier
            switch_to_cls = not switch_to_cls

        step = num_batches_per_epoch * epoch + batch_idx
        exp_lr_scheduler(step)

        images, class_labels, uq_idxs = batch
        ood_images = None
        # ood_images, _, _ = ood_batch

        class_labels = class_labels.cuda(non_blocking=True)

        if switch_to_cls: # train classifier
            freeze(prompter)
            args.grad_from_block = 11
            finetune_params(backbone, args)

            with torch.amp.autocast("cuda", enabled=args.fp16_scaler is not None):
                images = torch.cat([images[0].cuda(non_blocking=True), prompter(images[0].cuda(non_blocking=True)).detach()], dim=0)
                class_labels = torch.cat([class_labels, class_labels], dim=0)
                # ood_images = ood_images.cuda(non_blocking=True)
                loss, feats, outs = construct_gcd_loss(None, backbone, projector, images, class_labels, ood_images, epoch, args)

            optimizer_cls.zero_grad()

            if args.fp16_scaler is None:
                loss.backward()
                optimizer_cls.step()

            else:
                args.fp16_scaler.scale(loss).backward()
                args.fp16_scaler.step(optimizer_cls)
                args.fp16_scaler.update()

        else: # train prompter
            unfreeze(prompter)
            args.grad_from_block = 20 # large enough
            finetune_params(backbone, args)

            with torch.amp.autocast("cuda", enabled=args.fp16_scaler is not None):
                images = torch.cat(images, dim=0).cuda(non_blocking=True)
                class_labels_p = torch.cat([class_labels] * args.n_views, dim=0)
                # ood_images = ood_images.cuda(non_blocking=True)
                loss, feats, outs = construct_gcd_loss(prompter, backbone, projector, images, class_labels_p, ood_images, epoch, args)

            optimizer.zero_grad()

            if args.fp16_scaler is None:
                loss.backward()
                optimizer.step()
            else:
                args.fp16_scaler.scale(loss).backward()
                args.fp16_scaler.step(optimizer)
                args.fp16_scaler.update()

    exp_lr_scheduler_cls.step()

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
    return auroc, aupr_in, aupr_out, fpr95, detection_error

def test(model, test_loader, epoch, save_name, args, centers=None, avg_conf=None):

    model.eval()
    preds, targets, conf_score, e_score = [], [], [], []
    for images, label, _ in tqdm(test_loader):
        images = images.cuda(non_blocking=True)
        with torch.no_grad():
            bb_feat, pro_feat, logits = model(images)
            if args.use_bb:
                features = bb_feat
            else:
                features = pro_feat
            batch_preds = logits.argmax(dim=1).cpu().numpy()
            targets.append(label.cpu().numpy())
            conf = confident_score(logits, temperature=10).cpu().numpy()
            energy = energy_score(logits, temperature=1).cpu().numpy()
            preds.extend(batch_preds)
            conf_score.extend(conf)
            e_score.extend(energy)
    targets = np.concatenate(targets)
    conf_score = np.array(conf_score)
    e_score = np.array(e_score)
    preds = np.array(preds)
    evaluate('Confident Score', preds.copy(), conf_score, targets, args)


if __name__ == "__main__":

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    print(f'Using evaluation function {args.eval_funcs[0]} to print results')
    
    torch.backends.cudnn.benchmark = True

    # ----------------------
    # Hyper-paramters
    # ----------------------
    args.interpolation = 3
    args.crop_pct = 0.875
    args.image_size = 224
    args.feat_dim = 384 #768
    args.proj_dim = 256
    args.num_mlp_layers = 3
    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)
    args.num_ctgs = args.num_labeled_classes #+ args.num_unlabeled_classes

    # ----------------------
    # BASE MODEL
    # ----------------------
    # backbone = vits.__dict__['vit_base']().to(device)
    backbone = vits.__dict__['vit_small14_dinov3']().to(device)
    args.patch_size = 16
        
    if args.prompt_type == 'patch':
        args.prompt_size = 1
        prompter = PatchPrompter(args)

    elif args.prompt_type == 'all':
        args.prompt_size = 30
        prompter1 = PadPrompter(args)
        args.prompt_size = 1
        prompter2 = PatchPrompter(args)
        prompter = nn.Sequential(prompter1, prompter2)

    print(args)

    finetune_params(backbone, args) # HOW MUCH OF BASE MODEL TO FINETUNE

    # ----------------------
    # CLS HEAD
    # ----------------------
    projector = DINOHead(in_dim=args.feat_dim, use_bb=args.use_bb, out_dim=args.num_ctgs, nlayers=args.num_mlp_layers)
    if args.use_checkpoint:
        checkpoint = torch.load(args.pretrained_model_path, map_location='cpu')
        classifier = nn.Sequential(backbone, projector).cuda()
        model = nn.Sequential(prompter, classifier).cuda()
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else: model.load_state_dict(checkpoint)
    else:
        state_dict = torch.load(args.pretrained_model_path, map_location='cpu')
        backbone.load_state_dict(state_dict)
        classifier = nn.Sequential(backbone, projector).cuda()
        model = nn.Sequential(prompter, classifier).cuda()

    # ----------------------
    # OPTIMIZATION
    # ----------------------
    optimizer = SGD(get_params_groups(prompter), lr=args.lr, momentum=args.momentum, weight_decay=0)
    optimizer_cls = SGD(get_params_groups(classifier), lr=args.lr2, momentum=args.momentum, weight_decay=args.weight_decay)

    args.fp16_scaler = None
    if args.fp16:
        args.fp16_scaler = torch.amp.GradScaler("cuda")

    cluster_criterion = DistillLoss(
                        args.warmup_teacher_temp_epochs,
                        args.epochs,
                        args.n_views,
                        args.warmup_teacher_temp,
                        args.teacher_temp,
                    )

    # CONTRASTIVE TRANSFORM
    train_transform, ood_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)
    
    # DATASETS
    train_dataset, _, test_dataset = get_datasets(args.dataset_name, train_transform, ood_transform, test_transform, args)

    # --------------------
    # SAMPLER
    # Sampler which balances labelled and unlabelled examples in each batch
    # --------------------
    # label_len = len(train_dataset.labelled_dataset)
    # unlabelled_len = len(train_dataset.unlabelled_dataset)
    # sample_weights = [1 if i < label_len else label_len / unlabelled_len for i in range(len(train_dataset))]
    # sample_weights = torch.DoubleTensor(sample_weights)
    # sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(train_dataset))

    # ------------------
    # DATALOADERS
    # --------------------
    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False,
                              sampler=None, drop_last=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, num_workers=args.num_workers,
                                      batch_size=256, shuffle=False, pin_memory=False)
    # ood_loader = DataLoader(ood_train_dataset, num_workers=args.num_workers,
    #                                  batch_size=args.batch_size, shuffle=False, pin_memory=False)

    total_steps = len(train_loader) * args.epochs
    exp_lr_scheduler = cosine_lr(optimizer, args.lr, 1000, total_steps)
    exp_lr_scheduler_cls = lr_scheduler.CosineAnnealingLR(
            optimizer_cls,
            T_max=args.epochs,
            eta_min=args.lr2 * 0.1,
        )
    
    # ----------------------
    # TRAIN
    # ----------------------

    for epoch in range(args.epochs):
        print("Epoch: " + str(epoch+1))

        train(prompter, backbone, projector, train_loader, None, optimizer, optimizer_cls, exp_lr_scheduler, exp_lr_scheduler_cls, cluster_criterion, epoch, args)
    
        if (epoch+1) % args.eval_freq == 0:
            with torch.no_grad():
                test(model, test_loader, centers=None, avg_conf=None, epoch=epoch, save_name='Train ACC Unlabelled', args=args)
                if not os.path.exists(args.model_path):
                    os.makedirs(args.model_path)
                torch.save(model.state_dict(), os.path.join(args.model_path, args.model_name))
