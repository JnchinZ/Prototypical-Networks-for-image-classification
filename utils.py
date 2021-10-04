import os
import re
import glob
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_seed(seed=1024):
    if seed < 0:
        seed = random.randint(0, 100000)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return seed


def cosine_similarity(x, y):
    # x: N x D
    # y: M x D
    cos = nn.CosineSimilarity(dim=0)
    cos_sim = []
    for xi in x:
        cos_sim_i = []
        for yj in y:
            cos_sim_i.append(cos(xi, yj))
        cos_sim_i = torch.stack(cos_sim_i)
        cos_sim.append(cos_sim_i)
    cos_sim = torch.stack(cos_sim)
    return cos_sim  # (N, M)


def euclidean_dist_similarity(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return -torch.pow(x - y, 2).sum(2)  # N*M


def metrics(pred, true):
    pred = np.array(pred).reshape(-1)
    true = np.array(true).reshape(-1)
    # acc
    acc = np.mean((pred == true))
    # f_score
    f_score = f1_score(true, pred, average='macro')
    return acc, f_score


def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        suffix = path.suffix
        path = path.with_suffix('')
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # update path
    dir = path if path.suffix == '' else path.parent  # directory
    if not dir.exists() and mkdir:
        dir.mkdir(parents=True, exist_ok=True)  # make directory
    return path