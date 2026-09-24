"""Prototype: K-FAC / EK-FAC influence scores against a stored, projected index.

Once: build a projected training index and fit the Hessian factors.
Per query: build full-size query gradients, apply H^-1, project with the
index's matrices, and search the stored index with Attributor.

The reference is an unprojected stored index scored with the same Hessian fit
and the same transformed queries, so only the projection differs.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from scipy.stats import pearsonr, spearmanr

from bergson import GradientProcessor
from bergson.build import build
from bergson.collector.collector import create_module_projection_matrix
from bergson.config import (
    DataConfig,
    HessianConfig,
    IndexConfig,
    InversionConfig,
    PreprocessConfig,
)
from bergson.data import load_module_gradients
from bergson.hessians.apply_hessian import EkfacApplicator, EkfacConfig
from bergson.hessians.hessian_approximations import approximate_hessians
from bergson.query.attributor import Attributor

ROOT = Path("runs/ekfac_stored_index")
MODEL = "EleutherAI/pythia-14m"
DATASET = "NeelNanda/pile-10k"
TRAIN = "train[:64]"
QUERY = "train[:8]"  # the first 8 training docs, for the self-match check
NUM_QUERIES = 8
SEED = 3


def index_cfg(run: str, split: str, projection_dim: int) -> IndexConfig:
    return IndexConfig(
        run_path=str(ROOT / run),
        model=MODEL,
        data=DataConfig(dataset=DATASET, split=split, truncation=True),
        token_batch_size=2048,
        projection_dim=projection_dim,
        projection_seed=SEED,
        overwrite=True,
    )


def build_once(run: str, split: str, projection_dim: int = 0):
    if not (ROOT / run).exists():
        build(index_cfg(run, split, projection_dim), PreprocessConfig())


def fit_hessian_once():
    if not (ROOT / "hessian").exists():
        approximate_hessians(
            index_cfg("hessian", TRAIN, 0),
            HessianConfig(method="kfac", ev_correction=True),
        )


def transformed_queries(ev_correction: bool) -> dict[str, torch.Tensor]:
    """Full-size H^-1 q for each query."""
    build_once("query", QUERY)
    out = ROOT / f"query_ihvp_ev{int(ev_correction)}"
    if not out.exists():
        EkfacApplicator(
            EkfacConfig(
                hessian_method_path=str(ROOT / "hessian"),
                gradient_path=str(ROOT / "query"),
                run_path=str(out),
                ev_correction=ev_correction,
            ),
            inversion_cfg=InversionConfig(),
        ).compute_ivhp_sharded()
    grads = load_module_gradients(out)
    return {name: torch.from_numpy(np.asarray(grads[name][:])) for name in grads}


def project_like_index(
    queries: dict[str, torch.Tensor], index: str
) -> dict[str, torch.Tensor]:
    """Project full-size queries with the stored index's projection matrices."""
    processor = GradientProcessor.load(ROOT / index)
    p = processor.projection_dim
    eigen_a = load_file(ROOT / "hessian/eigen_activation_sharded/shard_0.safetensors")
    eigen_g = load_file(ROOT / "hessian/eigen_gradient_sharded/shard_0.safetensors")

    projected = {}
    for name, flat in queries.items():
        o, i = eigen_g[name].shape[1], eigen_a[name].shape[1]
        left, right = (
            create_module_projection_matrix(
                name,
                role,
                p,
                n,
                flat.dtype,
                flat.device,
                processor.projection_type,
                processor.projection_scale,
                processor.projection_seed,
            )
            for role, n in [("left", o), ("right", i)]
        )
        g = flat.view(-1, o, i)
        projected[name] = torch.einsum("ps,nsa,ra->npr", left, g, right).flatten(1)
    return projected


def search(index: str, queries: dict[str, torch.Tensor]) -> np.ndarray:
    """[num_train, num_queries] scores from a stored index."""
    attr = Attributor(ROOT / index, device="cpu", dtype=torch.float32)
    values, indices = attr.search(queries, k=None)
    return torch.empty_like(values).scatter_(1, indices, values).T.numpy()


def report(label: str, s: np.ndarray, ref: np.ndarray):
    diag = s[np.arange(NUM_QUERIES), np.arange(NUM_QUERIES)]
    top = (s.argmax(0) == np.arange(NUM_QUERIES)).sum()
    # Rank each query's training docs, excluding the query itself.
    off = ~np.eye(len(s), NUM_QUERIES, dtype=bool)
    per_query = [
        spearmanr(s[off[:, q], q], ref[off[:, q], q]).statistic
        for q in range(NUM_QUERIES)
    ]
    print(
        f"{label:<18} self>0 {(diag > 0).sum()}/8  top {top}/8  "
        f"pearson {pearsonr(s.ravel(), ref.ravel()).statistic:.3f}  "
        f"per-query spearman (excl. self) {np.mean(per_query):.3f}",
        flush=True,
    )


if __name__ == "__main__":
    dims = [int(a) for a in sys.argv[1:]] or [16, 32, 64, 128]
    fit_hessian_once()
    build_once("index_full", TRAIN)
    for ev_correction, method in [(False, "K-FAC"), (True, "EK-FAC")]:
        queries = transformed_queries(ev_correction)
        ref = search("index_full", queries)
        report(f"{method} unprojected", ref, ref)
        for p in dims:
            build_once(f"index_p{p}", TRAIN, p)
            s = search(f"index_p{p}", project_like_index(queries, f"index_p{p}"))
            report(f"{method} p={p}", s, ref)
