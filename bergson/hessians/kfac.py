import os
from dataclasses import dataclass, field

import torch
import torch.distributed as dist
import torch.nn as nn
from safetensors.torch import save_file
from torch import Tensor

from bergson.collector.collector import HookCollectorBase
from bergson.hessians.sharded_computation import (
    ShardedMul,
    assign_factor_devices,
    move_to_factor_device,
)
from bergson.utils.utils import assert_type


@dataclass(kw_only=True)
class CovarianceCollector(HookCollectorBase):
    """
    Collects activation and gradient covariances for EKFAC.

    Computes:
        A_cov = sum over batches of (X^T @ X)  for activations
        S_cov = sum over batches of (G^T @ G)  for gradients

    where X is input activations [N*S, I] and G is output gradients [N*S, O].
    """

    dtype: torch.dtype
    path: str
    factor_devices: list[str] = field(default_factory=list)
    """See ``HessianConfig.factor_devices``."""

    def setup(self) -> None:
        """Initialize covariance storage dictionaries."""
        self.A_cov_dict = {}
        self.S_cov_dict = {}
        self.shard_computer = ShardedMul()
        self.placement = (
            assign_factor_devices(self.target_info, self.factor_devices)
            if self.factor_devices
            else None
        )
        # Initialize sharded covariance matrices for ALL modules in target_info
        self.shard_computer._init_covariance_dict(
            activation_covariance_dict=self.A_cov_dict,
            gradient_covariance_dict=self.S_cov_dict,
            dtype=self.dtype,
            target_info=self.target_info,
            factor_devices=self.placement,
        )

    def forward_hook(self, module: nn.Module, a: Tensor) -> None:
        """Compute activation covariance: A^T @ A."""
        name = assert_type(str, module._name)
        A_cov_ki = self.A_cov_dict[name]
        mask = self._current_collection_mask
        assert mask is not None, "Collection mask not set for forward hook."

        # a: [N, S, I], collection mask: [N, S] -> select gradient-carrying positions
        a_bi = move_to_factor_device(a[mask], A_cov_ki, self.dtype)  # [num_valid, I]

        # Augment with a ones column so A matches the [O, I+1] gradient layout
        # produced when the bias gradient is collected.
        if module._collect_bias:
            a_bi = torch.cat(
                [a_bi, a_bi.new_ones(a_bi.shape[0], 1)], dim=1
            )  # [num_valid, I+1]

        self._accumulate(A_cov_ki, a_bi)

    def backward_hook(self, module: nn.Module, g: Tensor) -> None:
        """Compute gradient covariance: G^T @ G."""
        name = assert_type(str, module._name)
        S_cov_po = self.S_cov_dict[name]
        mask = self._current_collection_mask

        # g: [N, S, O], mask: [N, S] -> select gradient-carrying positions
        g_bo = move_to_factor_device(g[mask], S_cov_po, self.dtype)  # [num_valid, O]

        self._accumulate(S_cov_po, g_bo)

    def _accumulate(self, cov_shard: Tensor, x: Tensor) -> None:
        """Add this rank's rows of ``x^T @ x``, summed over ranks, to ``cov_shard``."""
        if not dist.is_initialized():
            cov_shard.addmm_(x.mT, x)
            return

        local_update = x.mT @ x
        dist.all_reduce(local_update, op=dist.ReduceOp.SUM)
        start_row, end_row = self.shard_computer.shard_bounds(local_update.shape[0])
        cov_shard.add_(local_update[start_row:end_row, :])

    def process_batch(self, indices: list[int], **kwargs) -> None:
        """No per-batch processing needed for covariance collection."""
        pass

    def teardown(self) -> None:
        """Save covariance matrices to disk."""
        activation_path = os.path.join(self.path, "activation_sharded")
        gradient_path = os.path.join(self.path, "gradient_sharded")

        os.makedirs(activation_path, exist_ok=True)
        os.makedirs(gradient_path, exist_ok=True)
        self.logger.info(
            f"Saving sharded covariance matrices to {activation_path} "
            f"and {gradient_path}"
        )
        # Save sharded covariance matrices
        save_file(
            self.A_cov_dict,
            os.path.join(activation_path, f"shard_{self.rank}.safetensors"),
        )
        save_file(
            self.S_cov_dict,
            os.path.join(gradient_path, f"shard_{self.rank}.safetensors"),
        )
        self.A_cov_dict.clear()
        self.S_cov_dict.clear()
