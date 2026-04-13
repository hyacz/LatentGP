import subprocess
import sys
import time
from pathlib import Path
import warnings
import re

import gpytorch
import numpy as np
import torch
import torch.nn as nn
from gpytorch.utils.warnings import GPInputWarning
import functools
import inspect

from config import *
from simple_parsing import ArgumentParser

from model import GenotypeGPModel
from data import load_data

torch._dynamo.config.cache_size_limit = 64

def get_git_hash() -> str:
    """Get current Git commit hash."""
    try:
        cmd = ['git', 'rev-parse', '--short', 'HEAD']
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode('utf-8').strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"

def check_git_status(allow_dirty: bool = False):
    """Check if Git working directory has uncommitted changes."""
    try:
        cmd = ['git', 'status', '--porcelain']
        status = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode('utf-8').strip()

        if status:
            if not allow_dirty:
                print("\n" + "!" * 50)
                print("CRITICAL ERROR: Working directory is dirty!")
                print("!" * 50)
                print("Uncommitted changes detected:\n")
                print(status)
                print("\nUse '--allow_dirty' argument to bypass this error.")
                sys.exit(1)
            else:
                print("\nWARNING: Running with uncommitted changes (Dirty State)!")
                return "dirty"
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("WARNING: Not a git repository.")
    return "clean"

class Logger(object):
    """
    Dual-output logger: writes to both console and file.
    """
    def __init__(self, filename):
        self.terminal = sys.stdout  # keep reference to original stdout
        self.log = open(filename, "a", encoding='utf-8')  # open file in append mode

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()    # flush immediately to preserve logs on crash

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return getattr(self.terminal, 'isatty', lambda: False)()

    def __getattr__(self, attr):
        if attr == 'terminal':  # prevent infinite recursion
            raise AttributeError()
        return getattr(self.terminal, attr)


class TorchStandardScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mean", None)
        self.register_buffer("std", None)

    def fit(self, y):
        """
        Compute mean and std (fitted on training y only).
        """
        self.mean = y.mean(dim=0, keepdim=True)
        self.std = y.std(dim=0, keepdim=True) + 1e-6  # prevent division by zero
        return self

    def transform(self, y):
        """
        Apply standardization: (y - mean) / std
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler has not been fitted yet!")
        # broadcasting handled automatically
        return (y - self.mean) / self.std

    def inverse_transform(self, x_norm):
        """
        Reverse standardization: x_norm * std + mean
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler has not been fitted yet!")
        return x_norm * self.std + self.mean

    def fit_transform(self, y):
        return self.fit(y).transform(y)


def compute_auc(labels, scores):
    """Compute AUC via Wilcoxon-Mann-Whitney statistic (pure torch)."""
    n_pos = labels.sum()
    n_neg = labels.numel() - n_pos
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    order = scores.argsort()
    ranks = torch.zeros_like(scores)
    ranks[order] = torch.arange(1, scores.numel() + 1, dtype=scores.dtype, device=scores.device)
    return ((ranks * labels).sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _eval_metric(y_true, y_pred, metric):
    """Compute evaluation metric: 'corr' or 'auc'."""
    if metric == "auc":
        return compute_auc(y_true, y_pred)
    return torch.corrcoef(torch.stack([y_true, y_pred]))[0, 1].item()


def _get_device_from_bound(bound):
    cfg = bound.arguments.get('cfg', None)
    if isinstance(cfg, ExpConfig):
        dev = cfg.sys.device
        try:
            return torch.device(dev)
        except Exception:
            return None
    return dev


def gpu_peak_monitor(func):
    """Decorator: reset CUDA peak stats before call and report peak after call.

    It inspects the wrapped function signature to find a `device` argument.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        sig = inspect.signature(func)
        bound = sig.bind_partial(*args, **kwargs)
        device = _get_device_from_bound(bound)

        cuda_dev_idx = None
        if device is not None and torch.cuda.is_available() and getattr(device, 'type', None) == 'cuda':
            cuda_dev_idx = device.index if device.index is not None else torch.cuda.current_device()
            try:
                torch.cuda.reset_peak_memory_stats(cuda_dev_idx)
            except Exception:
                try:
                    torch.cuda.reset_peak_memory_stats()
                except Exception:
                    pass

        result = func(*args, **kwargs)

        if cuda_dev_idx is not None:
            try:
                peak_alloc = torch.cuda.max_memory_allocated(cuda_dev_idx)
                peak_reserved = torch.cuda.max_memory_reserved(cuda_dev_idx)
            except Exception:
                peak_alloc = torch.cuda.max_memory_allocated()
                peak_reserved = torch.cuda.max_memory_reserved()
            print(f"Peak GPU memory allocated during training: {peak_alloc/1024**2:.2f} MB (reserved: {peak_reserved/1024**2:.2f} MB)")

        return result

    return wrapper


@gpu_peak_monitor
def model_run(train_x, train_y, test_x, test_y, cfg: ExpConfig):
    # best-effort reset of Dynamo (may not be present in all torch builds)
    try:
        torch._dynamo.reset()
    except Exception:
        pass

    # ensure device type
    device = cfg.sys.device
    if isinstance(device, str):
        device = torch.device(device)

    # Move data
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    test_x = test_x.to(device)
    test_y = test_y.to(device)

    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = GenotypeGPModel(train_x, train_y, likelihood,
                            kernel_configs=cfg.model.kernels,
                            h2_init=cfg.train.h2_init,
                            snp_batch_size=cfg.sys.snp_batch_size,
                            sample_block_size=cfg.sys.sample_block_size
                            ).to(device)

    likelihood.train()
    model.train()
    prev_loss = float('inf')
    count = 0

    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=cfg.train.lr,
        max_iter=20,
        history_size=10,
        line_search_fn='strong_wolfe'
    )

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # training loop
    trait_name = 'Simulated'
    fold_name = ''
    if cfg.data.mode == 'real':
        trait_name = cfg.data.phe_col
        fold_name = f' _ {cfg.data.fold_col}'
    print(f"Trait: {trait_name}{fold_name} - Method: {cfg.model.method}")
    print("Start Training...")
    start_time = time.time()

    with gpytorch.settings.max_cholesky_size(10000):
        for i in range(cfg.train.epochs):
            model.train()
            likelihood.train()

            def closure():
                optimizer.zero_grad()
                output = model(train_x)
                loss = -mll(output, train_y)
                loss.backward()
                return loss

            loss = optimizer.step(closure)

            # convergence check
            is_converged = False
            loss_delta = abs(prev_loss - loss.item())
            if loss_delta < cfg.train.tolerance:
                count += 1
                if count >= cfg.train.patience:
                    is_converged = True
            else:
                count = 0
            prev_loss = loss.item()

            # logging / evaluation
            logs = f"Iter {i+1}/{cfg.train.epochs} - Loss: {loss.item():.6f}"
            if (i + 1) % cfg.train.eval_step == 0 or i == cfg.train.epochs - 1 or is_converged:
                model.eval()
                likelihood.eval()
                metric_name = cfg.data.metric
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", GPInputWarning)
                        train_metric = _eval_metric(train_y, model(train_x).mean, metric_name)
                    test_metric = _eval_metric(test_y, model(test_x).mean, metric_name)

                    stats = model.get_stats()
                    logs += "".join([f" - {k}: {v:.2f}" for k, v in stats.items() if not k.startswith('Emb')])
                    logs += f" - {metric_name}: [{train_metric:.4f} {test_metric:.4f}]"
                    for k in stats.keys():
                        if k.startswith('Emb'):
                            logs += f" - {k}: {re.sub(r'\s+', ' ', str(np.round(stats[k].T, 4)).replace('\n', ','))}"

                print(logs)
                if is_converged:
                    print(f"Converged at epoch {i+1}. Loss delta {loss_delta:.2e} < {cfg.train.tolerance:.2e}")
                    break
            else:
                print(f"Iter {i+1}/{cfg.train.epochs} - Loss: {loss.item():.6f}")

    print(f"Training Done. Total time: {time.time() - start_time:.2f}s")
    return float(test_metric)


def main():
    # parse arguments
    parser = ArgumentParser()
    parser.add_arguments(ExpConfig, dest="cfg")
    args = parser.parse_args()
    cfg: ExpConfig = args.cfg

    if "cuda" in cfg.sys.device:
        torch.cuda.set_device(cfg.sys.device)

    # set random seed
    torch.manual_seed(cfg.sys.seed)
    np.random.seed(cfg.sys.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.sys.seed)

    # set torch dtype and threads
    if cfg.sys.enable_float64:
        torch.set_default_dtype(torch.float64)
    else:
        torch.set_float32_matmul_precision('high')
    torch.set_num_threads(cfg.sys.num_threads)

    # create output dir and save config
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(out_dir / cfg.output_conf)

    # configure logger
    log_path = out_dir / cfg.output_log
    sys.stdout = Logger(log_path)
    sys.stderr = sys.stdout

    # check git status
    git_status = check_git_status(allow_dirty=cfg.allow_dirty)
    cfg.git_hash = f"{get_git_hash()} ({git_status})"

    # print configuration
    print("=" * 40)
    print("Experiment Configuration:")
    print(cfg.dumps_yaml(allow_unicode=True).strip())
    print("=" * 40)

    # load data
    train_x, train_y, test_x, test_y = load_data(cfg)

    # normalize Y (skip for AUC -- binary 0/1 labels must stay intact)
    if cfg.data.metric == "auc":
        train_y_norm, test_y_norm = train_y, test_y
    else:
        y_scaler = TorchStandardScaler().to(cfg.sys.device)
        train_y_norm = y_scaler.fit_transform(train_y)
        test_y_norm = y_scaler.transform(test_y)

    print(f"VRAM estimate: {sum(map(lambda x:x.numel(), [train_x, train_y, test_x, test_y]))/1024**2:.2f} MB")

    # run model
    result = model_run(train_x, train_y_norm, test_x, test_y_norm, cfg=cfg)

    print(f"Final test {cfg.data.metric}: {result:.4f}")


if __name__ == "__main__":
    main()
