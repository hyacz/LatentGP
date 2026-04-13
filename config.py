from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import torch
from simple_parsing import Serializable


# --- Data Configuration ---
@dataclass
class SimConfig:
    """Phenotype simulation parameters (only used when data.mode='sim')."""
    h2: float = 0.5             # heritability
    num_qtls: int = 1000        # number of QTL loci
    dominance: float = 0.0      # dominance degree (0=additive, 1=full dominance)

@dataclass
class DataConfig:
    """Real data path configuration."""
    bed_path: str = "path/to/genotypes.bed"
    mode: str = "real"   # "real" or "sim"
    test_ratio: float = 0.2  # only used for simulated data or real data without fold_col

    # --- Real Data Specific ---
    phe_path: str = "path/to/phenotypes.txt"
    phe_col: str = "trait_name"
    fold_col: str = "trait_name_1"  # column name for train/test split
    metric: str = "corr"  # evaluation metric: "corr" or "auc"

    # --- Simulation Data Specific ---
    sim: SimConfig = field(default_factory=SimConfig)

# --- Model Architecture Configuration ---
@dataclass
class KernelConfig:
    """Parameters for a single kernel."""
    name: str = "GenotypeKernel"                        # kernel name
    fixed_weights: Optional[List[List[float]]] = None   # fixed weights (if any)
    learnable_dims: int = 1                             # dimension of learnable embeddings
    init_learnable_weights: Optional[List[float]] = None  # initial weights for learnable embeddings
    init_share: float = 1                               # initial variance component share (after excluding noise)
    lr_scale: float = 0.1                               # reparameterization scaling factor for learnable embeddings
    mask_missing: bool = True                           # whether to mask missing values (Index 3)

@dataclass
class ModelConfig:
    """Kernel and embedding architecture parameters."""
    method: str = "lrn"
    lrn_init: Optional[List[float]] = None  # CLI override for lrn kernel's init_learnable_weights

    kernels: List[KernelConfig] = field(init=False, metadata={"cmd": False})

    def __post_init__(self):
        presets = {
            "add": [
                KernelConfig("add", fixed_weights=[[0., 1., 2., 0.]], learnable_dims=0)
            ],
            "dom": [
                KernelConfig("dom", fixed_weights=[[0., 1., 0., 0.]], learnable_dims=0)
            ],
            "lrn": [
                KernelConfig("lrn", init_learnable_weights=[[0., 1., 2., 0.]], learnable_dims=1)
            ],
            "lrn2": [
                KernelConfig("lrn", init_learnable_weights=[[0., 1., 2., 0.]], learnable_dims=2)
            ],
            "add_lrn": [
                KernelConfig("add_lrn", fixed_weights=[[0., 1., 2., 0.]], learnable_dims=1)
            ],
            "add_dom": [
                KernelConfig("add_lrn", fixed_weights=[[0., 1., 2., 0.], [0., 1., 0., 0.]], learnable_dims=0)
            ],
            "add+dom": [
                KernelConfig("add", fixed_weights=[[0., 1., 2., 0.]], learnable_dims=0, init_share=0.8),
                KernelConfig("dom", fixed_weights=[[0., 1., 0., 0.]], learnable_dims=0, init_share=0.2),
            ],
            "add+lrn": [
                KernelConfig("add", fixed_weights=[[0., 1., 2., 0.]], learnable_dims=0, init_share=0.8),
                KernelConfig("lrn", init_learnable_weights=[[0., 1., 0., 0.]], learnable_dims=1, init_share=0.2)
            ],
            "dom+lrn": [
                KernelConfig("lrn", init_learnable_weights=[[0., 1., 2., 0.]], learnable_dims=1, init_share=0.8),
                KernelConfig("dom", fixed_weights=[[0., 1., 0., 0.]], learnable_dims=0, init_share=0.2),
            ],
            "lrn+lrn": [
                KernelConfig("lrn1", init_learnable_weights=[[0., 1., 2., 0.]], learnable_dims=1, init_share=0.5),
                KernelConfig("lrn2", init_learnable_weights=[[0., 1., 0., 0.]], learnable_dims=1, init_share=0.5),
            ],
            "add+dom+lrn": [
                KernelConfig("add", fixed_weights=[[0., 1., 2., 0.]], learnable_dims=0, init_share=0.3),
                KernelConfig("dom", fixed_weights=[[0., 1., 0., 0.]], learnable_dims=0, init_share=0.3),
                KernelConfig("lrn", init_learnable_weights=[[0., 1., 4., 0.]], learnable_dims=1, init_share=0.3)
            ],
        }
        if self.method not in presets:
            raise ValueError(f"Unknown method '{self.method}'. Choose from: {list(presets.keys())}")
        self.kernels = presets[self.method]

        # CLI override for lrn kernel encoding initial values
        if self.lrn_init is not None:
            for k in self.kernels:
                if k.name == "lrn":
                    k.init_learnable_weights = [self.lrn_init]
            self.lrn_init = None

# --- Training Hyperparameters ---
@dataclass
class TrainConfig:
    """Optimizer and training loop parameters."""
    h2_init: float = 0.5             # initial heritability estimate (Vg / Vy)
    lr: float = 0.1                  # LBFGS global learning rate
    epochs: int = 50
    eval_step: int = 1               # evaluation interval (in epochs)
    patience: int = 0                # convergence patience
    tolerance: float = 1e-5          # loss convergence threshold

# --- Hardware and Computation ---
@dataclass
class SystemConfig:
    """Device and performance optimization parameters."""
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    snp_batch_size: int = 100000      # operator computation batch size (reduces VRAM)
    sample_block_size: int = 10000    # sample loop batch size
    enable_float64: bool = True
    num_threads: int = 16            # only for CPU training

# --- Top-level Experiment Configuration ---
@dataclass
class ExpConfig(Serializable):
    """Top-level experiment configuration for GenotypeGPModel."""

    # sub-configurations (initialized with default_factory)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    sys: SystemConfig = field(default_factory=SystemConfig)

    # --- Experiment Logging (kept at top level for convenience) ---
    output_dir: str = field(
        default_factory=lambda: f"./logs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    output_conf: str = "config.yaml"
    output_log: str = "output.log"
    allow_dirty: bool = False        # whether to allow running with uncommitted changes

    # auto-recorded field
    git_hash: str = field(default="unknown", metadata={"cmd": False})
