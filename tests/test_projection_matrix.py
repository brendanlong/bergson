"""Random projection matrices must be the same wherever they're generated, since
an index and the queries scored against it are often projected on different
machines."""

import hashlib
import json
import math

import pytest
import torch
import yaml

import bergson.collector.projection_matrix as projection_matrix
from bergson import GradientProcessor
from bergson.collector.gradient_collectors import GradientCollector
from bergson.collector.projection_matrix import random_matrix
from bergson.config import IndexConfig, ScoreConfig
from bergson.gradients import PROJECTION_VERSION
from bergson.process_grads import mix_autocorrelation_matrices
from bergson.score.score import get_query_grads

SHAPES = [(2, 4), (7, 1001), (64, 13824), (64, 1 << 18)]


def test_matrices_are_pinned():
    """Changing these values breaks every saved index, so it needs a new
    ``PROJECTION_VERSION``."""
    normal = random_matrix("layer/left", 2, 4, "cpu", "normal")
    torch.testing.assert_close(
        normal,
        torch.tensor(
            [
                [-1.0932976, -1.0874549, -0.3873174, -0.4707755],
                [-1.9625934, 1.6714863, 0.1380329, -0.4476005],
            ]
        ),
        rtol=0,
        atol=1e-6,
    )
    rademacher = random_matrix("layer/left", 16, 64, "cpu", "rademacher")
    digest = hashlib.sha256(rademacher.numpy().tobytes()).hexdigest()
    assert digest == "2dc4725decc3ec5fe0b9a16ae64c4dabc2abeb35123e3783e40edfbf79d74e5a"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("shape", SHAPES)
def test_cuda_matches_cpu(shape):
    m, n = shape
    cpu = random_matrix("layer/left", m, n, "cpu", "rademacher")
    cuda = random_matrix("layer/left", m, n, "cuda", "rademacher")
    assert not projection_matrix._kernels_failed, "the CUDA kernel didn't run"
    assert cuda.device.type == "cuda"
    assert torch.equal(cuda.cpu(), cpu)

    cpu = random_matrix("layer/left", m, n, "cpu", "normal")
    cuda = random_matrix("layer/left", m, n, "cuda", "normal")
    # log, sin and cos may round differently on the GPU.
    torch.testing.assert_close(cuda.cpu(), cpu, rtol=0, atol=2e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("projection_type", ["normal", "rademacher"])
@pytest.mark.parametrize("n", [13824, 13825])
def test_cuda_bf16_matches_rounded_fp32(projection_type, n):
    """bf16 matrices come from their own kernel, which writes two entries at a
    time, so odd widths are cut down from the next even one."""
    fp32 = random_matrix("layer/left", 64, n, "cuda", projection_type)
    bf16 = random_matrix("layer/left", 64, n, "cuda", projection_type, torch.bfloat16)
    assert bf16.dtype == torch.bfloat16 and bf16.shape == (64, n)
    assert bf16.is_contiguous()
    cpu = random_matrix("layer/left", 64, n, "cpu", projection_type, torch.bfloat16)
    if projection_type == "rademacher":
        assert torch.equal(bf16, fp32.to(torch.bfloat16))
        assert torch.equal(bf16.cpu(), cpu)
    else:
        torch.testing.assert_close(bf16, fp32.to(torch.bfloat16))
        torch.testing.assert_close(bf16.cpu(), cpu)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("projection_type", ["normal", "rademacher"])
def test_cuda_generates_other_dtypes_in_slabs(monkeypatch, projection_type):
    """Narrow dtypes without their own kernel are generated in fp32 slabs."""
    monkeypatch.setattr(projection_matrix, "_GPU_SLAB", 64 * 1000)
    fp32 = random_matrix("layer/left", 64, 13824, "cuda", projection_type)
    fp16 = random_matrix("layer/left", 64, 13824, "cuda", projection_type, torch.half)
    assert fp16.dtype == torch.half
    assert torch.equal(fp16, fp32.to(torch.half))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("projection_type", ["normal", "rademacher"])
def test_cuda_falls_back_when_the_kernel_fails(monkeypatch, projection_type):
    fused = random_matrix("layer/left", 64, 13824, "cuda", projection_type)
    assert not projection_matrix._kernels_failed
    # Restored after the test, which the fallback sets to True.
    monkeypatch.setattr(projection_matrix, "_kernels_failed", False)

    def broken(_):
        raise RuntimeError("no compiler")

    monkeypatch.setattr(projection_matrix, "_kernel", broken)
    fallback = random_matrix("layer/left", 64, 13824, "cuda", projection_type)
    assert projection_matrix._kernels_failed
    assert fallback.device.type == "cuda"
    torch.testing.assert_close(fallback, fused, rtol=0, atol=2e-6)


@pytest.mark.parametrize("projection_type", ["normal", "rademacher"])
def test_entries_do_not_depend_on_the_shape(projection_type):
    """Each entry depends only on its row and column, so matrices of any size
    agree where they overlap."""
    big = random_matrix("layer/left", 70, 70001, "cpu", projection_type)
    small = random_matrix("layer/left", 3, 1001, "cpu", projection_type)
    assert torch.equal(big[:3, :1001], small)


def test_normal_entries_are_standard_normal():
    z = random_matrix("layer/left", 64, 1 << 16, "cpu", "normal").double().flatten()
    assert abs(z.mean().item()) < 0.005
    assert z.var().item() == pytest.approx(1.0, abs=0.005)
    kurtosis = ((z - z.mean()) ** 4).mean() / z.var() ** 2
    assert kurtosis.item() == pytest.approx(3.0, abs=0.02)
    for s in (2, 3, 4):
        expected = math.erfc(s / math.sqrt(2))
        assert (z.abs() > s).double().mean().item() == pytest.approx(expected, rel=0.1)


@pytest.mark.parametrize("projection_type", ["normal", "rademacher"])
def test_entries_are_uncorrelated(projection_type):
    a = random_matrix("layer/left", 64, 1 << 16, "cpu", projection_type).flatten()
    b = random_matrix("layer/right", 64, 1 << 16, "cpu", projection_type).flatten()
    # Four standard errors for 4M entries.
    bound = 4 / math.sqrt(a.numel())
    for lag in (1, 2, 31, 32, 33):
        assert abs(torch.corrcoef(torch.stack([a[:-lag], a[lag:]]))[0, 1]) < bound
    assert abs(torch.corrcoef(torch.stack([a, b]))[0, 1]) < bound


def test_rejects_unknown_type_and_oversized_matrices():
    with pytest.raises(ValueError, match="Unknown projection type"):
        random_matrix("layer/left", 2, 2, "cpu", "uniform")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="2\\^31"):
        random_matrix("layer/left", 1, 1 << 31, "cpu", "normal")


def _save_old_processor(path, projection_dim):
    GradientProcessor(projection_dim=projection_dim).save(path)
    cfg_path = path / "processor_config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    assert cfg["projection_version"] == PROJECTION_VERSION
    del cfg["projection_version"]
    cfg_path.write_text(yaml.safe_dump(cfg))


def test_processor_without_projection_version_loads_as_version_1(tmp_path):
    _save_old_processor(tmp_path, projection_dim=8)
    processor = GradientProcessor.load(tmp_path)
    assert processor.projection_version == 1
    with pytest.raises(ValueError, match="Rebuild the index"):
        processor.check_projection_version(tmp_path)


def test_unprojected_old_processor_is_accepted(tmp_path):
    _save_old_processor(tmp_path, projection_dim=None)
    GradientProcessor.load(tmp_path).check_projection_version(tmp_path)


def test_collector_rejects_old_projected_processor(tmp_path, model, dataset):
    _save_old_processor(tmp_path, projection_dim=8)
    with pytest.raises(ValueError, match="Rebuild the index"):
        GradientCollector(
            model=model,
            cfg=IndexConfig(run_path=str(tmp_path / "run")),
            data=dataset,
            processor=GradientProcessor.load(tmp_path),
            skip_index=True,
        )


def test_mixing_rejects_old_projected_hessians(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "bergson.process_grads.assert_autocorrelation_hessian", lambda path: None
    )
    old, new = tmp_path / "old", tmp_path / "new"
    _save_old_processor(old, projection_dim=8)
    GradientProcessor(projection_dim=8).save(new)
    for query, index in [(new, old), (old, new)]:
        with pytest.raises(ValueError, match="Rebuild the index"):
            mix_autocorrelation_matrices(query, index, tmp_path / "mixed")


def test_scoring_rejects_old_projected_query(tmp_path):
    _save_old_processor(tmp_path, projection_dim=8)
    (tmp_path / "info.json").write_text(json.dumps({"grad_sizes": {"layer": 64}}))
    with pytest.raises(ValueError, match="Rebuild the index"):
        get_query_grads(ScoreConfig(query_path=str(tmp_path)))
