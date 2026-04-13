# LatentGP

**A Latent Gaussian Process Framework for Genomic Prediction**

LatentGP is a flexible genomic prediction framework that introduces **learnable genotype encodings** within a Gaussian Process (GP) kernel. Instead of relying on fixed genotype-to-numeric mappings (e.g., additive `0-1-2` coding in GBLUP), LatentGP parameterizes the encoding as a learnable embedding matrix optimized via maximum likelihood, enabling data-driven discovery of non-canonical allele interactions.

## Features

- **Learnable Genotype Encoding** — Optimizes per-genotype effect values from data rather than using fixed codings
- **Multiple Genetic Architectures** — Supports additive (`add`), dominance (`dom`), learnable (`lrn`), and combined kernels (`add+dom+lrn`, etc.)
- **Memory-Efficient Computation** — Batched kernel operations with `torch.compile` acceleration, scales to hundreds of thousands of SNPs
- **Variance Component Decomposition** — Multi-kernel model with automatic heritability estimation
- **PLINK Compatible** — Reads standard `.bed` genotype files via `bed-reader`
- **Built-in Phenotype Simulator** — Simulate quantitative traits with configurable heritability and dominance degree

## Installation

```bash
# Clone the repository
git clone https://github.com/hyacz/LatentGP.git
cd LatentGP

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python >= 3.9
- PyTorch >= 2.0 (with CUDA for GPU support)
- GPyTorch >= 1.11

## Quick Start

### Real Data Prediction

```bash
python main.py \
    --data.bed_path path/to/genotypes.bed \
    --data.phe_path path/to/phenotypes.txt \
    --data.phe_col trait_name \
    --data.fold_col trait_name_1 \
    --model.method lrn \
    --device 'cuda:1'
```

### Simulated Data

```bash
python main.py \
    --data.bed_path path/to/genotypes.bed \
    --data.mode sim \
    --data.sim.h2 0.5 \
    --data.sim.num_qtls 1000 \
    --model.method lrn \
    --device 'cuda:1'
```

### Using Config Files

```bash
python main.py --config_path configs/real_lrn.yaml
```

Override specific fields from a config:

```bash
python main.py --config_path configs/real_lrn.yaml --model.method add+lrn --train.lr 0.05
```

## Supported Methods

| Method | Description |
|--------|-------------|
| `add` | Standard additive GBLUP kernel (`0-1-2` coding) |
| `lrn` | Learnable encoding (single dimension) |
| `add+dom` | Additive + dominance as separate kernels |
| `add+lrn` | Additive + learnable as separate kernels |
| `add+dom+lrn` | Additive + dominance + learnable (three kernels) |

## Project Structure

```
LatentGP/
├── main.py          # Training entry point and experiment orchestration
├── config.py        # Configuration dataclasses (CLI-serializable)
├── data.py          # Genotype/phenotype loading and train-test splitting
├── kernel.py        # GenotypeKernel with memory-efficient batched computation
├── model.py         # Multi-kernel ExactGP model
├── simulator.py     # Phenotype simulator with configurable QTL effects
├── configs/         # Example configuration files
│   ├── real_lrn.yaml
│   ├── real_add_lrn.yaml
│   └── sim_additive.yaml
├── requirements.txt
├── pyproject.toml
└── LICENSE
```

## Configuration

All hyperparameters are configurable via CLI flags or YAML files. Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model.method` | `lrn` | Kernel architecture preset |
| `--train.lr` | `0.1` | LBFGS learning rate |
| `--train.h2_init` | `0.5` | Initial heritability estimate |
| `--sys.device` | `cuda:0` | Compute device |
| `--sys.snp_batch_size` | `100000` | SNP batching size (memory control) |
| `--sys.sample_block_size` | `10000` | Sample batching size |

Run `python main.py --help` for the full list of configurable parameters.

## Citation

If you use LatentGP in your research, please cite:

```bibtex
@article{latentgp2026,
  title={LatentGP: Uncovering Global Gene Action Patterns via Learnable Genotype Encoding},
  author={},
  journal={},
  year={2026},
  volume={},
  pages={},
  doi={}
}
```

## License

This project is licensed under the [MIT License](LICENSE).
