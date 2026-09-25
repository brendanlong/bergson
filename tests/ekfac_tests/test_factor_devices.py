"""Fitting with the factors on other devices must match fitting them next to the
model: the factor stores are the same."""

import os
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from bergson.config import DataConfig, HessianConfig, IndexConfig
from bergson.hessians.hessian_approximations import (
    FACTOR_SUBDIRS,
    approximate_hessians,
)
from bergson.hessians.sharded_computation import assign_factor_devices


def test_assign_factor_devices_balances_elements():
    device = torch.device("cpu")
    target_info = {
        "big": (device, torch.Size([8, 8]), False),
        "mid": (device, torch.Size([6, 6]), False),
        "small_a": (device, torch.Size([4, 4]), False),
        "small_b": (device, torch.Size([4, 4]), False),
    }
    placement = assign_factor_devices(target_info, ["cuda:0", "cuda:1"])
    assert placement["big"] == torch.device("cuda:0")
    assert placement["mid"] == torch.device("cuda:1")
    assert placement["small_a"] == torch.device("cuda:1")
    assert placement["small_b"] == torch.device("cuda:1")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_factor_devices_match_default(tmp_path: Path):
    def fit(name: str, factor_devices: list[str]) -> str:
        cfg = IndexConfig(
            run_path=str(tmp_path / name),
            model="EleutherAI/pythia-14m",
            data=DataConfig(
                dataset="NeelNanda/pile-10k", split="train[:8]", truncation=True
            ),
            token_batch_size=512,
            precision="fp32",
            filter_modules="embed_out",
        )
        cfg.distributed.nproc_per_node = 1
        # Dataset labels: sampled labels would differ between the two fits.
        hessian_cfg = HessianConfig(
            method="kfac",
            ev_correction=True,
            use_dataset_labels=True,
            factor_devices=factor_devices,
        )
        return approximate_hessians(cfg, hessian_cfg)

    # The CPU stands in for a second GPU, so some modules' factors are on another
    # device than the model and some share its device.
    default, placed = fit("default", []), fit("placed", ["cpu", "cuda:0"])

    for sub in FACTOR_SUBDIRS:
        a = load_file(os.path.join(default, sub, "shard_0.safetensors"))
        b = load_file(os.path.join(placed, sub, "shard_0.safetensors"))
        assert a.keys() == b.keys(), sub
        for key in a:
            expected, actual = a[key], b[key]
            # Eigenvectors are sign-ambiguous per column, and the CPU's rounding
            # moves them more than the matrices they come from.
            if sub.startswith("eigen_"):
                expected, actual, tolerance = expected.abs(), actual.abs(), 2e-3
            else:
                tolerance = 1e-4
            torch.testing.assert_close(
                actual, expected, rtol=0, atol=tolerance * expected.abs().max()
            )
