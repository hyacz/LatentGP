import numpy as np
import pandas as pd
import torch
from bed_reader import open_bed
from pathlib import Path

from simulator import PhenotypeSimulator
from config import *

# --- Utility Functions ---

def _read_genotypes(bed_path, valid_indices=None):
    """
    Read genotypes from a PLINK BED file.
    If valid_indices is provided, only those rows are read; otherwise all rows are read.
    Returns: torch.int8 tensor
    """
    with open_bed(bed_path) as f:
        if valid_indices is not None:
            # bed-reader supports numpy array as index
            x_raw = f.read(index=np.s_[valid_indices, :], dtype='int8')
        else:
            x_raw = f.read(dtype='int8')

    # handle PLINK missing values (-127 -> 3)
    x_raw[x_raw < 0] = 3
    return x_raw

def _split_indices(n_samples, test_ratio, predefined_train_mask=None):
    """
    Train/test split logic.
    Args:
        predefined_train_mask: boolean array, True indicates training sample.
    Returns:
        (train_mask, test_mask)
    """
    if predefined_train_mask is not None:
        # [Branch A] Use predefined mask (real data with fold_col)
        # externally defined train/test assignment
        if len(predefined_train_mask) != n_samples:
            raise ValueError(f"Predefined mask length ({len(predefined_train_mask)}) does not match samples ({n_samples})")

        train_mask = predefined_train_mask
        test_mask = ~train_mask
        print(f"  [Split] Using pre-defined train/test. Train: {train_mask.sum()}, Test: {test_mask.sum()}")

    else:
        # [Branch B] Random split (sim data or real data without fold_col)
        perm = np.random.permutation(n_samples)
        split_idx = int(n_samples * (1 - test_ratio))

        train_indices = perm[:split_idx]

        train_mask = np.zeros(n_samples, dtype=bool)
        train_mask[train_indices] = True
        test_mask = ~train_mask
        print(f"  [Split] Random split ({1-test_ratio:.1f}/{test_ratio:.1f}). Train: {train_mask.sum()}, Test: {test_mask.sum()}")

    return train_mask, test_mask

# --- Data Loading Pipelines ---
def _load_real_mode(data_cfg: DataConfig):
    print(f"Loading phenotype from {data_cfg.phe_path}...")

    phe_df = pd.read_csv(data_cfg.phe_path, sep=r'\s+', na_values='NA', header='infer')

    use_index = str(data_cfg.phe_col).isdigit() and data_cfg.phe_col not in phe_df.columns

    # 1. Filter samples with missing phenotypes
    if use_index:
        y_raw = phe_df.iloc[:, int(data_cfg.phe_col)].values
    else:
        y_raw = phe_df[data_cfg.phe_col].values
    valid_mask = ~np.isnan(y_raw)
    valid_indices = np.where(valid_mask)[0]

    print(f"  Samples: Total {len(phe_df)}, Valid {len(valid_indices)} (Removed {len(phe_df)-len(valid_indices)} NaN)")

    # 2. Read valid genotypes
    x_raw = _read_genotypes(data_cfg.bed_path, valid_indices)

    # 3. Extract valid Y
    y = y_raw[valid_mask]

    # 4. Extract predefined train mask (if available)
    predefined_train_mask = None
    if data_cfg.fold_col:
        if use_index:
            fold_values = phe_df.iloc[valid_indices, int(data_cfg.fold_col)]
        elif data_cfg.fold_col in phe_df.columns:
            fold_values = phe_df.loc[valid_mask, data_cfg.fold_col]
        predefined_train_mask = fold_values.notna().values

    return x_raw, y, predefined_train_mask


def _load_sim_mode(data_cfg: DataConfig, output_dir):
    print(f"Simulating phenotype based on {data_cfg.bed_path}...")

    # 1. Read genotypes
    x_raw = _read_genotypes(data_cfg.bed_path, valid_indices=None)

    # 2. Run simulator
    simulator = PhenotypeSimulator(x_raw, h2=data_cfg.sim.h2, num_qtls=data_cfg.sim.num_qtls)
    y = simulator.simulate(dominance_degree=data_cfg.sim.dominance)

    # 3. Save simulated phenotype
    save_path = Path(output_dir) / "sim_phenotype.txt"
    np.savetxt(save_path, y, fmt='%.6f', header='Simulated_Phenotype')
    print(f"  Simulated phenotype saved to: {save_path}")

    return x_raw, y

# --- Main Entry Point ---
def load_data(cfg: ExpConfig):
    """
    Unified data loading entry point.
    cfg: top-level config object (contains cfg.data, cfg.sim, cfg.output_dir)
    Returns: train_x, train_y, test_x, test_y (all on CPU)
    """
    predefined_train_mask = None
    if cfg.data.mode == 'real':
        x_raw, y, predefined_train_mask = _load_real_mode(cfg.data)
    else:
        x_raw, y = _load_sim_mode(cfg.data, cfg.output_dir)

    # perform split
    train_mask, test_mask = _split_indices(
        n_samples=len(y),
        test_ratio=cfg.data.test_ratio,
        predefined_train_mask=predefined_train_mask
    )

    # save split result
    split_save_path = Path(cfg.output_dir) / "data_split.csv"
    pd.DataFrame({
        'train_mask': train_mask,
        'test_mask': test_mask
    }).to_csv(split_save_path, index=False)
    print(f"  [Split] Data split saved to: {split_save_path}")

    # slice tensors
    train_x = torch.from_numpy(x_raw[train_mask]).to(dtype=torch.int8)
    train_y = torch.from_numpy(y[train_mask]).float()
    test_x = torch.from_numpy(x_raw[test_mask]).to(dtype=torch.int8)
    test_y = torch.from_numpy(y[test_mask]).float()

    return train_x, train_y, test_x, test_y
