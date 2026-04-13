import torch
import torch.nn as nn
import gpytorch
from kernel import GenotypeKernel
from config import KernelConfig
from typing import List

class GenotypeGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel_configs: List[KernelConfig], h2_init=0.5, snp_batch_size=1e5, sample_block_size=1e4):
        super().__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ConstantMean()
        self.kernels = nn.ModuleList()
        self.configs = kernel_configs

        # init var
        y_var = train_y.var().item()
        total_vg = h2_init * y_var
        total_share = sum(cfg.init_share for cfg in kernel_configs)

        likelihood.noise = (1 - h2_init) * y_var

        for cfg in kernel_configs:
            k = gpytorch.kernels.ScaleKernel(
                GenotypeKernel(
                    train_x_raw=train_x,
                    fixed_weights=cfg.fixed_weights,
                    init_learnable_weights=cfg.init_learnable_weights,
                    learnable_dims=cfg.learnable_dims,
                    lr_scale=cfg.lr_scale,
                    snp_batch_size=snp_batch_size,
                    sample_block_size=sample_block_size
                )
            )
            k.outputscale = cfg.init_share * total_vg / total_share
            self.kernels.append(k)

    def forward(self, x):
        dummy = torch.zeros(x.size(0), 1, device=x.device)  # pass dummy input to avoid OOM
        mean_x = self.mean_module(dummy)
        covar_x = self.kernels[0](x)
        for i in range(1, len(self.kernels)):
            covar_x += self.kernels[i](x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def get_stats(self):
        stats = {}

        total_vg = 0
        for i, (k, cfg) in enumerate(zip(self.kernels, self.configs)):
            name = cfg.name if hasattr(cfg, 'name') else f'K{i}'

            vg = k.outputscale.item()
            total_vg += vg
            stats[f'V_{name}'] = vg

            if cfg.learnable_dims > 0:
                stats[f'Emb_{name}'] = k.base_kernel.raw_learnable.squeeze().cpu().numpy() * cfg.lr_scale

        ve = self.likelihood.noise.item()
        total_v = total_vg + ve
        stats['Ve'] = ve
        stats['h2'] = total_vg / total_v

        return stats
