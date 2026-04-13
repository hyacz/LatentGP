import gpytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from gpytorch.kernels import Kernel


# Operations and Kernels ==========================================
def _get_norm_chunk(indices, weights, freqs_chunk, mask_missing):
    """
    Atomic operation: Embedding -> Global Normalize -> (Optional) Mask Missing.
    If mask_missing=True, weights at missing positions must be zero for correct results.
    """
    # 1. Embedding [N, B] -> [N, B, D]
    idx_long = indices.contiguous().long()
    x = F.embedding(idx_long, weights, padding_idx=0)

    # 2. Compute global statistics
    # Global Mean = Freqs @ Weights
    global_mean = freqs_chunk @ weights

    # Global Var = E[X^2] - (E[X])^2
    global_sq_exp = freqs_chunk @ (weights ** 2)
    global_var = global_sq_exp - global_mean ** 2
    global_std = torch.sqrt(global_var + 1e-6)

    # 3. Normalize (Broadcasting)
    # [N, B, D]
    z = (x - global_mean.unsqueeze(0)) / global_std.unsqueeze(0)

    # 4. Optionally mask missing values (Index 3)
    # If mask_missing=True, force missing positions to 0
    if mask_missing:
        mask = (idx_long == 3).unsqueeze(-1) # [N, B, 1]
        z = z.masked_fill(mask, 0.0)

    return z.flatten(1)


@torch.compile(mode="reduce-overhead", fullgraph=True)
def fused_forward_step(x1_c, x2_c, weights, freqs_c, mask_missing, is_same_input):
    f1 = _get_norm_chunk(x1_c, weights, freqs_c, mask_missing)
    if is_same_input:   # inner .data_ptr() call would break torch.compile
        f2 = f1
    else:
        f2 = _get_norm_chunk(x2_c, weights, freqs_c, mask_missing)

    return f1 @ f2.t()


@torch.compile(mode="reduce-overhead")
def fused_backward_step(grad_output, x1_c, x2_c, weights, freqs_c, mask_missing, is_same_input):
    f1 = _get_norm_chunk(x1_c, weights, freqs_c, mask_missing)
    if is_same_input:
        f2 = f1
    else:
        f2 = _get_norm_chunk(x2_c, weights, freqs_c, mask_missing)

    local_loss = (grad_output * (f1 @ f2.t())).sum()
    gw = torch.autograd.grad(local_loss, weights, retain_graph=False)[0]
    return gw


class MemoryEfficientSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weights, x1, x2, freqs, batch_size, mask_missing):
        # save tensor arguments
        ctx.save_for_backward(weights, x1, x2, freqs)
        # save non-tensor arguments (int, bool)
        ctx.bs = batch_size
        ctx.mask = mask_missing

        N1, M = x1.shape
        N2, _ = x2.shape
        output = torch.zeros((N1, N2), device=weights.device)

        with torch.no_grad():
            for start in range(0, M, batch_size):
                end = min(start + batch_size, M)

                x1_c = x1[:, start:end]
                x2_c = x2[:, start:end]
                freqs_c = freqs[start:end]
                is_same_input = (x1_c.data_ptr() == x2_c.data_ptr())
                output += fused_forward_step(x1_c, x2_c, weights, freqs_c, mask_missing, is_same_input)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # unpack saved arguments
        weights, x1, x2, freqs = ctx.saved_tensors
        batch_size = ctx.bs
        mask_missing = ctx.mask

        M = x1.shape[1]
        grad_weights = torch.zeros_like(weights)

        # skip gradient computation if weights don't require grad (fixed GBLUP mode)
        if not weights.requires_grad:
             return None, None, None, None, None, None

        with torch.enable_grad():
            for start in range(0, M, batch_size):
                end = min(start + batch_size, M)

                x1_c = x1[:, start:end]
                x2_c = x2[:, start:end]
                freqs_c = freqs[start:end]
                is_same_input = (x1_c.data_ptr() == x2_c.data_ptr())
                # accumulate gradients
                grad_weights += fused_backward_step(grad_output, x1_c, x2_c, weights, freqs_c, mask_missing, is_same_input)

        # lock padding row (Index 0)
        grad_weights[0] = 0
        if mask_missing:
            grad_weights[-1] = 0

        # return values correspond to the 6 inputs of forward
        return grad_weights, None, None, None, None, None


class GenotypeKernel(Kernel):
    def __init__(self, train_x_raw,
                 fixed_weights=None,
                 init_learnable_weights=None,
                 learnable_dims=1,
                 lr_scale=0.1,
                 mask_missing=True,
                 snp_batch_size=100000, sample_block_size=10000, **kwargs):
        """
        Genotype kernel with optional learnable embeddings.

        Args:
            train_x_raw: raw genotype matrix [N, M] (int8, values in {0,1,2,3}).
            fixed_weights: list of fixed encoding vectors (e.g., [[0,1,2,0]] for additive).
            init_learnable_weights: initial values for learnable embeddings.
            learnable_dims: number of learnable embedding dimensions.
            lr_scale: reparameterization scaling factor for learnable embeddings.
            mask_missing: whether to mask missing genotypes (Index 3).
            snp_batch_size: batch size for SNP dimension (controls VRAM usage).
            sample_block_size: batch size for sample dimension.
        """
        super().__init__(**kwargs)
        self.num_snps = train_x_raw.shape[1]
        self.snp_batch_size = snp_batch_size
        self.sample_block_size = sample_block_size
        self.learnable_dims = learnable_dims
        self.lr_scale = lr_scale
        self.mask_missing = mask_missing

        # 2. Pre-compute allele frequencies
        freqs = self._compute_freqs(train_x_raw, sample_block_size=sample_block_size)
        self.register_buffer("global_freqs", freqs)

        # 3. Initialize weights
        if fixed_weights is not None:
            self.register_buffer("fixed_base", torch.as_tensor(fixed_weights, device=train_x_raw.device).t())

        if self.learnable_dims > 0:
            self.raw_learnable = nn.Parameter(torch.empty(4, self.learnable_dims, device=train_x_raw.device))

            with torch.no_grad():
                nn.init.xavier_normal_(self.raw_learnable, gain=self.lr_scale)

                if init_learnable_weights is not None:
                    iw = torch.as_tensor(init_learnable_weights, device=train_x_raw.device).t()
                    input_rows, input_cols = iw.shape
                    if input_rows != 4:
                        raise ValueError(f"Init weights rows must be 4, got {input_rows}")
                    if input_cols > self.learnable_dims:
                        raise ValueError(f"Init cols ({input_cols}) > learnable_dims ({self.learnable_dims})")
                    self.raw_learnable.data[:, :input_cols] = iw / self.lr_scale

                self.raw_learnable.data[0] = 0
                if self.mask_missing:
                    self.raw_learnable.data[-1] = 0 / self.lr_scale

        # caching for fixed kernels
        self.use_cache = False
        if fixed_weights is not None and learnable_dims == 0:
            self.use_cache = True
            self._train_data_ref = train_x_raw
            self._cached_k = None

        self._cached_trace_scale = None


    @property
    def weights(self):
        """
        Dynamically compute the active weight matrix.
        Autograd handles the gradient: grad_raw = grad_used * lr_scale.
        """
        parts = []

        # 1. Fixed part
        if hasattr(self, "fixed_base") and self.fixed_base is not None:
            parts.append(self.fixed_base)

        # 2. Learnable part (apply lr_scale)
        if hasattr(self, "raw_learnable") and self.raw_learnable is not None:
            parts.append(self.raw_learnable * self.lr_scale)

        # 3. Concatenate
        if len(parts) == 1:
            return parts[0]
        else:
            return torch.cat(parts, dim=1) # [4, F+D]


    def _compute_freqs(self, x_raw, sample_block_size):
        # Chunked allele frequency computation
        N, M = x_raw.shape
        freqs = torch.zeros(M, 4, device=x_raw.device)
        with torch.no_grad():
            for start in range(0, M, sample_block_size):
                end = min(start + sample_block_size, M)
                chunk = x_raw[:, start:end].long()
                counts = F.one_hot(chunk, num_classes=4).sum(dim=0).float()
                freqs[start:end] = counts / N
        return freqs


    def forward(self, x1, x2, diag=False, **params):
        # check cache
        if self.use_cache and self._cached_k is not None:
            is_train = (x1.data_ptr() == self._train_data_ref.data_ptr()) and \
                       (x2.data_ptr() == self._train_data_ref.data_ptr())
            if is_train:
                return self._cached_k.diag() if diag else self._cached_k

        # regular computation
        N1, M = x1.shape
        N2, _ = x2.shape
        device = x1.device
        res = torch.zeros((N1, N2), device=device)

        is_symmetric = (x1 is x2) or (N1 == N2 and x1.data_ptr() == x2.data_ptr())

        # double-blocking over samples
        for i_start in range(0, N1, self.sample_block_size):
            i_end = min(i_start + self.sample_block_size, N1)
            if self.training: torch.cuda.empty_cache()

            j_start_init = i_start if is_symmetric else 0
            for j_start in range(j_start_init, N2, self.sample_block_size):
                j_end = min(j_start + self.sample_block_size, N2)

                # invoke operator
                block_res = MemoryEfficientSum.apply(
                    self.weights,
                    x1[i_start:i_end],
                    x2[j_start:j_end],
                    self.global_freqs,
                    self.snp_batch_size,  # int
                    self.mask_missing     # bool
                )

                res[i_start:i_end, j_start:j_end] = block_res
                if is_symmetric and i_start != j_start:
                    res[j_start:j_end, i_start:i_end] = block_res.T

        if is_symmetric:
            trace_scale = res.trace() / N1
            self._cached_trace_scale = trace_scale
        elif self._cached_trace_scale is not None:
            trace_scale = self._cached_trace_scale
        else:
            # fallback: compute trace from diagonal blocks of K(x1, x1)
            trace_val = 0.0
            for i_start in range(0, N1, self.sample_block_size):
                i_end = min(i_start + self.sample_block_size, N1)
                block_diag = MemoryEfficientSum.apply(
                    self.weights,
                    x1[i_start:i_end],
                    x1[i_start:i_end],
                    self.global_freqs,
                    self.snp_batch_size,
                    self.mask_missing
                )
                trace_val += block_diag.diag().sum()
            trace_scale = trace_val / N1

        res /= trace_scale

        # write to cache
        if self.use_cache and self._cached_k is None:
            is_train = (x1.data_ptr() == self._train_data_ref.data_ptr()) and \
                       (x2.data_ptr() == self._train_data_ref.data_ptr())
            if is_train: self._cached_k = res.detach()

        if diag:
            return res.diag()
        return res
