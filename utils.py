import math
import os
import json
import logging
import random
import subprocess
import time
import numpy as np
from math import sqrt
from scipy import stats
import torch

def set_seed(seed=0):
    """Seed python, numpy, torch, and CUDA deterministically."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def seed_worker(worker_id):
    """Worker init function for DataLoader reproducibility."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_git_commit_hash(repo_dir=None):
    """Get current git commit hash, or 'unknown' if not in a git repo."""
    try:
        cmd = ['git', 'rev-parse', 'HEAD']
        cwd = repo_dir or os.path.dirname(os.path.abspath(__file__))
        commit = subprocess.check_output(cmd, cwd=cwd, stderr=subprocess.DEVNULL).decode('ascii').strip()
        return commit
    except Exception:
        return 'unknown'


def setup_run_dir(output_dir='runs', dataset='davis', split='random', variant='full', timestamp=None):
    """Create and return a timestamped run directory."""
    if timestamp is None:
        timestamp = time.strftime('%Y%m%d_%H%M%S')
    run_name = f"{dataset}_{split}_{variant}_{timestamp}"
    run_dir = os.path.join(output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_run_config(run_dir, args):
    """Save all effective args and git commit hash to config.json and git_commit.txt."""
    os.makedirs(run_dir, exist_ok=True)
    commit_hash = get_git_commit_hash()
    config_dict = vars(args).copy() if hasattr(args, '__dict__') else dict(args)
    config_dict['git_commit'] = commit_hash
    config_dict['run_dir'] = run_dir

    config_path = os.path.join(run_dir, 'config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, default=str)

    commit_path = os.path.join(run_dir, 'git_commit.txt')
    with open(commit_path, 'w', encoding='utf-8') as f:
        f.write(commit_hash + '\n')

    return config_dict


def setup_logger(run_dir, name='BiSSA'):
    """Setup a dual logger writing to stdout and run_dir/train.log."""
    os.makedirs(run_dir, exist_ok=True)
    log_file = os.path.join(run_dir, 'train.log')
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


class IsolatedLoader:
    """Wrapper that enforces test-set isolation by throwing on access while locked."""
    def __init__(self, loader, name="test_loader"):
        self._loader = loader
        self._name = name
        self.is_locked = True

    def unlock(self):
        self.is_locked = False

    def __iter__(self):
        if self.is_locked:
            raise RuntimeError(f"CRITICAL PROTOCOL VIOLATION: {self._name} accessed while locked during training! Test-set isolation violated.")
        return iter(self._loader)

    def __len__(self):
        return len(self._loader)

    @property
    def dataset(self):
        return self._loader.dataset


def rmse(y,f):
    rmse = sqrt(((y - f)**2).mean(axis=0))
    return rmse
def mse(y,f):
    mse = ((y - f)**2).mean(axis=0)
    return mse
def pearson(y,f):
    rp = np.corrcoef(y, f)[0,1]
    return rp
def spearman(y,f):
    rs = stats.spearmanr(y, f)[0]
    return rs
def ci(y,f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y)-1
    j = i-1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z+1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i-1
    ci = S/z
    return ci

def r_squared_error(y_obs,y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    y_pred_mean = [np.mean(y_pred) for y in y_pred]

    mult = sum((y_pred-y_pred_mean) * (y_obs-y_obs_mean))
    mult = mult * mult

    y_obs_sq = sum((y_obs-y_obs_mean) * (y_obs-y_obs_mean))
    y_pred_sq = sum((y_pred-y_pred_mean) * (y_pred-y_pred_mean))

    return mult / float(y_obs_sq*y_pred_sq)

def get_k(y_obs,y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    return sum(y_obs*y_pred) / float(sum(y_pred*y_pred))


def squared_error_zero(y_obs,y_pred):
    k = get_k(y_obs,y_pred)

    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    upp = sum((y_obs-(k*y_pred)) * (y_obs-(k*y_pred)))
    down= sum((y_obs-y_obs_mean) * (y_obs-y_obs_mean))

    return 1 - (upp/float(down))

def rm2(ys_orig,ys_line):
    r2 = r_squared_error(ys_orig, ys_line)
    r02 = squared_error_zero(ys_orig, ys_line)

    return r2 * (1-np.sqrt(np.absolute((r2*r2)-(r02*r02))))


def mae(predictions, targets):
    if len(predictions) != len(targets):
        raise ValueError("error")

    total_error = np.sum(np.abs(predictions - targets))
    mae = total_error / len(predictions)
    return mae
