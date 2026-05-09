from matplotlib.pylab import diff

import torch
import torch.nn.functional as F
from tqdm import tqdm
from config import cifar_mean, cifar_std
import numpy as np

@torch.no_grad()
def compute_class_centers(model, data_loader, num_classes, temperature=1.0, eps = 1e-8, device = "cuda", normalize=True, use_bb=True, save_path=None):
    model.eval()
    features = []
    labels = []
    total_conf = 0.0
    total_samples = 0
    with torch.no_grad():
        for image, label, _ in tqdm(data_loader, desc="Computing Class Centers"):
            image = image.cuda(non_blocking=True)
            bb_feat, pro_feat, logits = model(image)
            if use_bb:
                feat = bb_feat
            else:
                feat = pro_feat
            features.extend(feat.cpu())
            labels.extend(label.cpu())

            probs = torch.softmax(logits / temperature, dim=1)
            conf = probs.max(dim=1)[0]
            total_conf += conf.sum().item()
            total_samples += conf.size(0)

    avg_conf = total_conf / total_samples
    features = torch.stack(features)
    labels = torch.tensor(labels)
    D = features.size(1)
    global_mean = features.mean(dim=0)
    global_std = features.std(dim=0) + eps
    if normalize:
        features = (features - global_mean) / global_std

    centers = torch.zeros(num_classes, D, device=features.device)
    # counts = torch.zeros(num_classes, device=features.device)
    # inv_cov = torch.zeros(num_classes, D, D, device=features.device)
    all_diff = []
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() > 0:
            class_feat = features[mask]
            centers[c] = class_feat.mean(dim=0)
            # counts[c] = mask.sum()
            diff = class_feat - centers[c]
            # cov = (diff.t() @ diff) / (mask.sum() - 1)
            # inv_cov[c] = torch.linalg.inv(cov + eps * torch.eye(D, device=cov.device))
            all_diff.append(diff)
    all_diff = torch.cat(all_diff, dim=0)
    cov = (all_diff.t() @ all_diff) / (len(all_diff) - 1)
    inv_cov = torch.linalg.inv(cov + eps * torch.eye(D, device=cov.device))
    if save_path:
        torch.save(centers, save_path)
    return centers.to(device),avg_conf, global_mean.to(device), global_std.to(device), inv_cov.to(device)

@torch.no_grad()
def compute_average_conf(model, data_loader, temperature=1.0, eps=1e-8, device = "cuda", save_path=None):
    model.eval()
    total_conf = 0.0
    total_samples = 0
    with torch.no_grad():
        for image, _, _ in tqdm(data_loader, desc="Computing Average Confidence"):
            image = image.cuda(non_blocking=True)
            _, _, logits = model(image)

            probs = torch.softmax(logits / temperature, dim=1)
            conf = probs.max(dim=1)[0]
            total_conf += conf.sum().item()
            total_samples += conf.size(0)

    avg_conf = total_conf / total_samples
    if save_path:
        torch.save(avg_conf, save_path)
    return avg_conf

def confident_score(logits, temperature):
    probs = F.softmax(logits / temperature, dim=1)
    conf = probs.max(dim=1)[0] 
    return conf

def energy_score(logits, temperature=1.0):
    return -temperature * torch.logsumexp(logits / temperature, dim=1)

def cadref_score(features, logits, centers, classifier_weight, avg_conf, temperature, eps=1e-8):
    preds = torch.argmax(logits, dim=1)
    Fy = centers[preds]
    Wy = classifier_weight[preds]
    R = features - Fy
    sign_align = Wy * R
    pos_mask = sign_align > 0
    neg_mask = sign_align < 0
    abs_R = torch.abs(R)
    norm = torch.norm(features, p=1, dim=1, keepdim=True) + eps
    Ep = (abs_R * pos_mask).sum(dim=1) / norm.squeeze(1)
    En = (abs_R * neg_mask).sum(dim=1) / norm.squeeze(1)
    probs = F.softmax(logits / temperature, dim=1)
    conf = probs.max(dim=1)[0] 
    score = -(Ep/conf + En/avg_conf) 
    return score

def cadref_score_2(features, logits, centers, classifier_weight, avg_conf, temperature, eps=1e-8):
    preds = torch.argmax(logits, dim=1)
    Fy = centers[preds]
    R = features - Fy
    sg = classifier_weight[preds].sign()
    ep_dist = (R * sg)
    ep_dist[ep_dist < 0] = 0
    ep_error = ep_dist.norm(p=1, dim=1)/features.norm(p=1, dim=1)

    en_dist = R * (-sg)
    en_dist[en_dist < 0] = 0
    en_error = en_dist.norm(p=1, dim=1)/features.norm(p=1, dim=1)

    logit_score = confident_score(logits, 1)
    score = -(ep_error/logit_score + en_error/avg_conf)

    return score

def odin_preprocess(model, x, temperature=0.5, epsilon=0.001):
    """
    x: input tensor (requires grad)
    model: pretrained classifier
    """
    model.eval()
    with torch.enable_grad():
        x = x.clone().detach().requires_grad_(True)
        _, _, logits = model(x)
        logits = logits / temperature
        pred = logits.argmax(dim=1)
        log_probs = F.log_softmax(logits, dim=1)
        selected = log_probs[torch.arange(x.size(0)), pred]
        selected.sum().backward()
        grad_sign = x.grad.data.sign()
        x_tilde = x - epsilon * grad_sign
    return x_tilde.detach()

def odin_score(inputs, model, temperature=1000.0, noise=0.001, mean=cifar_mean, std=cifar_std):
    device = inputs.device
    inputs = inputs.clone().detach().requires_grad_(True)
    _, _, outputs = model(inputs)
    preds = outputs.argmax(dim=1)
    scaled_outputs = outputs / temperature
    loss = F.cross_entropy(scaled_outputs, preds)
    loss.backward()
    grad = inputs.grad.detach()
    grad_sign = grad.sign()
    if std is not None:
        std = torch.tensor(std, device=device).view(1, -1, 1, 1)
        grad_sign = grad_sign / std
    perturbed_inputs = inputs - noise * grad_sign
    with torch.no_grad():
        _, _, outputs = model(perturbed_inputs)
        outputs = outputs / temperature
        probs = F.softmax(outputs, dim=1)
    return probs

def energy_score(logits, temperature=1.0):
    return -temperature * torch.logsumexp(logits / temperature, dim=1)

def Mahalanobis_score(features, centers, inv_cov, global_mean= None, global_std= None, normalize=False, eps=1e-8):
    if normalize:
        features = (features - global_mean) / global_std
    diff = features.unsqueeze(1) - centers.unsqueeze(0)
    # m_dist = torch.einsum('bcd,cde,bce->bc', diff, inv_cov, diff)
    tmp = torch.einsum('bcd,de->bce', diff, inv_cov)
    m_dist = (tmp * diff).sum(dim=-1)
    return -m_dist
