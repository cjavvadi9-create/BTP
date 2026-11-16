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


from metrics import (
    rmse,
    mse,
    mae,
    pearson,
    spearman,
    ci,
    rm2,
    compute_classification_metrics,
)
