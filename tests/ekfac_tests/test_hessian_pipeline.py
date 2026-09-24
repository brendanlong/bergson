import pytest

from bergson.config import (
    HessianConfig,
    HessianPipelineConfig,
    IndexConfig,
    PreprocessConfig,
    ScoreConfig,
)
from bergson.hessians.pipeline import hessian_pipeline


def test_hessian_pipeline_rejects_unit_normalize(tmp_path):
    """Cosine similarity (unit_normalize) is not supported with the
    Kronecker-factored Hessians hessian_pipeline fits and applies."""
    with pytest.raises(ValueError, match="unit_normalize"):
        hessian_pipeline(
            IndexConfig(run_path=str(tmp_path / "run")),
            HessianConfig(method="kfac"),
            ScoreConfig(),
            PreprocessConfig(unit_normalize=True),
            HessianPipelineConfig(),
        )


def test_hessian_pipeline_passes_projection_config_to_apply(tmp_path, monkeypatch):
    """The apply step must compress the query with the same projection settings
    as the index it is scored against."""
    captured = []
    for name in [
        "build_query",
        "approximate_hessians",
        "score_dataset",
        "save_run_config",
    ]:
        monkeypatch.setattr(f"bergson.hessians.pipeline.{name}", lambda *a, **kw: None)
    monkeypatch.setattr(
        "bergson.hessians.pipeline.launch_distributed_run",
        lambda name, fn, args, dist_cfg: captured.append(args[0]),
    )

    index_cfg = IndexConfig(
        run_path=str(tmp_path / "run"),
        projection_dim=16,
        projection_type="normal",
        projection_scale="row_norm",
        projection_seed=3,
    )
    hessian_pipeline(
        index_cfg,
        HessianConfig(method="kfac"),
        ScoreConfig(),
        PreprocessConfig(),
        HessianPipelineConfig(),
    )

    (ekfac_cfg,) = captured
    assert ekfac_cfg.projection_dim == 16
    assert ekfac_cfg.projection_type == "normal"
    assert ekfac_cfg.projection_scale == "row_norm"
    assert ekfac_cfg.projection_seed == 3
