import torch
import numpy as np

class PhenotypeSimulator:
    def __init__(self, x_raw, h2=0.5, num_qtls=1000):
        """
        Pure NumPy phenotype simulator.
        Args:
            x_raw: [N, M] int8 array (CPU array).
            h2: target heritability.
            num_qtls: number of QTL loci (randomly sampled from M).
        """
        # convert int8 to float32 for numerical precision
        self.X = x_raw.astype(np.float32)
        self.N, self.M = self.X.shape
        self.h2 = h2
        self.num_qtls = num_qtls

        # 1. Randomly select QTL loci (without replacement)
        all_snps = np.arange(self.M)
        self.qtl_indices = np.random.choice(all_snps, size=num_qtls, replace=False)

        # 2. Generate additive effects (beta), drawn from standard normal
        self.beta = np.random.normal(loc=0.0, scale=1.0, size=num_qtls)

    def simulate(self, dominance_degree=0.0):
        """
        Generate phenotype y, supporting different dominance degrees.
        Args:
            dominance_degree (d): dominance degree
                d = 0   : pure additive -> 0, 1, 2
                d = 0.5 : full dominance -> 0, 1, 1
                d = 1.0 : over-dominance -> 0, 1, 0
        Returns:
            y: standardized phenotype array [N,]
            map_vec: genotype-to-effect mapping array
        """
        # extract QTL genotype matrix [N, num_qtls]
        X_qtl = self.X[:, self.qtl_indices]

        # --- Build genotype-to-effect lookup table ---
        add_vec = np.array([0.0, 1.0, 2.0, 0.0], dtype=np.float32)
        dom_vec = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        map_vec = add_vec * (1 - dominance_degree) + dom_vec * dominance_degree

        # map genotypes to effect values (NumPy indexing-based lookup)
        # convert X_qtl to integer index first (avoid float indexing issues)
        x_mapped = map_vec[X_qtl.astype(np.int64)]

        # --- Compute genetic values G (G = x_mapped @ beta) ---
        G = x_mapped @ self.beta  # matrix multiplication, result shape [N,]

        # --- Add environmental noise to control heritability h2 ---
        var_g = np.var(G)
        # prevent zero variance (avoid division by zero)
        if var_g == 0:
            var_g = 1.0

        # environmental noise variance: var_e = var_g * (1 - h2) / h2
        var_e = var_g * (1 - self.h2) / self.h2
        # generate normally distributed environmental noise
        noise = np.random.normal(loc=0.0, scale=np.sqrt(var_e), size=self.N)

        # phenotype = genetic value + environmental noise
        y = G + noise

        # standardize phenotype (mean=0, std=1) for model training
        y = (y - np.mean(y)) / np.std(y)

        return y
