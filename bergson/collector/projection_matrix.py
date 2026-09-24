"""Random projection matrices that are the same on every device.

Each entry is computed from a keyed hash of its row and column, so the values
don't depend on the device, on how the work is split across threads, or on the
matrix's shape: ``A[:k, :l]`` is the ``[k, l]`` matrix with the same identifier.
Rademacher matrices are bit-identical everywhere; normal matrices can differ in
the last bits because ``log``, ``sin`` and ``cos`` round differently across
devices.
"""

import hashlib
import math
from typing import Callable, Literal

import numpy as np
import torch
from torch import Tensor

from bergson.utils.logger import get_logger

logger = get_logger("projection_matrix", level="INFO")

# NumPy is single-threaded, so the CPU path isn't slowed down by torch's thread
# pool when the machine has fewer free cores than it reports.
_CPU_CHUNK = 1 << 16
"""Entries per CPU chunk, small enough to keep its temporaries in cache."""

_GPU_SLAB = 1 << 25
"""Entries generated in fp32 at a time when the output dtype is narrower."""

# Written as one function because the jiterator only accepts a single template
# function. lowbias32 is Chris Wellons' 32-bit integer hash. The row key is
# added after the first round so that rows aren't shuffled copies of each other.
_HASH = """
    auto lowbias32 = [](unsigned int v) {
        v ^= v >> 16; v *= 0x7feb352du; v ^= v >> 15; v *= 0x846ca68bu; v ^= v >> 16;
        return v;
    };
    unsigned int row_key = lowbias32(static_cast<unsigned int>(row)
                                     ^ static_cast<unsigned int>(k2));
    auto keyed = [&](unsigned int c) {
        return lowbias32(lowbias32(c ^ static_cast<unsigned int>(k1)) + row_key);
    };
"""
# Box-Muller on the hashes of columns 2p and 2p + 1 gives r and t, and columns
# 2p and 2p + 1 are r * cos(t) and r * sin(t).
_BOX_MULLER = """
    float u1 = (__uint2float_rn(keyed(2 * p)) + 0.5f) * 2.3283064365386963e-10f;
    float u2 = __uint2float_rn(keyed(2 * p + 1)) * 2.3283064365386963e-10f;
    float r = sqrtf(-2.0f * logf(u1));
    float t = 6.2831853071795864769f * u2;
"""


def _header(name: str, index: str) -> str:
    return f"template <typename T> T {name}(T row, T {index}, T k1, T k2) {{"


# Rounds to nearest even, like torch's float -> bfloat16 conversion.
_BF16 = """
    auto bf16 = [](float z) {
        unsigned int b = __float_as_uint(z);
        return (b + 0x7fffu + ((b >> 16) & 1u)) >> 16;
    };
"""
_KERNELS = {
    "normal": _header("hashed_normal", "col")
    + _HASH
    + """
    unsigned int c = static_cast<unsigned int>(col);
    unsigned int p = c >> 1;"""
    + _BOX_MULLER
    + """
    float z = (c & 1u) ? r * sinf(t) : r * cosf(t);
    return static_cast<T>(__float_as_int(z));
}
""",
    "rademacher": _header("hashed_rademacher", "col")
    + _HASH
    + """
    unsigned int c = static_cast<unsigned int>(col);
    float z = ((keyed(c >> 5) >> (c & 31u)) & 1u) ? 1.0f : -1.0f;
    return static_cast<T>(__float_as_int(z));
}
""",
    # Columns 2p and 2p + 1 as bfloat16, packed into one 32-bit output.
    "normal_bf16": _header("hashed_normal_bf16", "pair")
    + _HASH
    + _BF16
    + """
    unsigned int p = static_cast<unsigned int>(pair);"""
    + _BOX_MULLER
    + """
    return static_cast<T>(bf16(r * cosf(t)) | (bf16(r * sinf(t)) << 16));
}
""",
    "rademacher_bf16": _header("hashed_rademacher_bf16", "pair")
    + _HASH
    + """
    unsigned int c = 2 * static_cast<unsigned int>(pair);
    unsigned int bits = keyed(c >> 5) >> (c & 31u);
    unsigned int lo = (bits & 1u) ? 0x3f80u : 0xbf80u;
    unsigned int hi = (bits & 2u) ? 0x3f80u : 0xbf80u;
    return static_cast<T>(lo | (hi << 16));
}
""",
}
_jit_fns: dict[str, Callable] = {}
_kernels_failed = False


def random_matrix(
    identifier: str,
    m: int,
    n: int,
    device: torch.device | str,
    projection_type: Literal["normal", "rademacher"],
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """An ``[m, n]`` matrix of standard normal or ±1 entries determined by
    ``identifier``."""
    if projection_type not in ("normal", "rademacher"):
        raise ValueError(f"Unknown projection type: {projection_type}")
    if max(m, n) >= 1 << 31:
        raise ValueError(
            f"Projection matrices are limited to 2^31 rows and columns, got [{m}, {n}]."
        )

    device = torch.device(device)
    k1, k2 = _keys(identifier)
    if device.type == "cuda" and not _kernels_failed:
        try:
            return _random_matrix_cuda(k1, k2, m, n, device, projection_type, dtype)
        except torch.OutOfMemoryError:
            raise
        except Exception as e:
            _disable_kernels(e)

    # One torch op at the end: each torch op on the CPU may wait for torch's
    # whole thread pool.
    out = np.empty((m, n), dtype=np.float32)
    generate = _normal_cpu if projection_type == "normal" else _rademacher_cpu
    rows_per = max(1, _CPU_CHUNK // n)
    # Chunks start at a multiple of 32 so each Rademacher hash lands in one chunk.
    cols_per = min(n, _CPU_CHUNK)
    for r0 in range(0, m, rows_per):
        r1 = min(m, r0 + rows_per)
        rows = np.arange(r0, r1, dtype=np.uint32)[:, None]
        for c0 in range(0, n, cols_per):
            c1 = min(n, c0 + cols_per)
            out[r0:r1, c0:c1] = generate(k1, k2, rows, c0, c1)
    return torch.from_numpy(out).to(device, dtype)


def _keys(identifier: str) -> tuple[int, int]:
    digest = hashlib.md5(identifier.encode()).digest()
    return (
        int.from_bytes(digest[:4], "little", signed=True),
        int.from_bytes(digest[4:8], "little", signed=True),
    )


def _random_matrix_cuda(
    k1: int,
    k2: int,
    m: int,
    n: int,
    device: torch.device,
    projection_type: str,
    dtype: torch.dtype,
) -> Tensor:
    rows = torch.arange(m, dtype=torch.int32, device=device)[:, None]
    if dtype == torch.bfloat16:
        kernel = _kernel(f"{projection_type}_bf16")
        pairs = torch.arange((n + 1) // 2, dtype=torch.int32, device=device)[None, :]
        A = kernel(rows, pairs, k1=k1, k2=k2).view(torch.bfloat16)
        return A if n % 2 == 0 else A[:, :n].contiguous()

    kernel = _kernel(projection_type)

    def columns(c0: int, c1: int) -> Tensor:
        cols = torch.arange(c0, c1, dtype=torch.int32, device=device)[None, :]
        return kernel(rows, cols, k1=k1, k2=k2).view(torch.float32)

    if dtype == torch.float32 or m * n <= _GPU_SLAB:
        return columns(0, n).to(dtype)

    # Generate fp32 in slabs so the fp32 copy of the whole matrix never exists.
    out = torch.empty(m, n, dtype=dtype, device=device)
    cols_per = max(1, _GPU_SLAB // m)
    for c0 in range(0, n, cols_per):
        c1 = min(n, c0 + cols_per)
        out[:, c0:c1] = columns(c0, c1)
    return out


def _kernel(name: str) -> Callable:
    """A fused CUDA/ROCm kernel, compiled for each device on first use."""
    if name not in _jit_fns:
        from torch.cuda.jiterator import _create_jit_fn

        _jit_fns[name] = _create_jit_fn(_KERNELS[name], k1=0, k2=0)
    return _jit_fns[name]


def _disable_kernels(e: Exception) -> None:
    # The CPU path computes the same values, so falling back only costs speed.
    global _kernels_failed
    _kernels_failed = True
    logger.warning(
        f"Couldn't run the projection matrix kernel ({e}); generating projection "
        "matrices on the CPU instead."
    )


_M1 = np.uint32(0x7FEB352D)
_M2 = np.uint32(0x846CA68B)


def _lowbias32(v: np.ndarray) -> np.ndarray:
    v ^= v >> np.uint32(16)
    v *= _M1
    v ^= v >> np.uint32(15)
    v *= _M2
    v ^= v >> np.uint32(16)
    return v


def _hash(k1: int, k2: int, rows: np.ndarray, counters: np.ndarray) -> np.ndarray:
    """The kernel's ``keyed(c)`` for each row in ``rows`` (a column vector) and
    counter ``c`` in ``counters``."""
    row_key = _lowbias32(rows ^ np.uint32(k2 & 0xFFFFFFFF))
    return _lowbias32(_lowbias32(counters ^ np.uint32(k1 & 0xFFFFFFFF)) + row_key)


def _normal_cpu(k1: int, k2: int, rows: np.ndarray, c0: int, c1: int) -> np.ndarray:
    """Columns ``2p`` and ``2p + 1`` share a Box-Muller pair built from the
    hashes of ``2p`` and ``2p + 1``."""
    pairs = np.arange(c0 // 2, (c1 + 1) // 2, dtype=np.uint32)
    a = _hash(k1, k2, rows, pairs * np.uint32(2))
    b = _hash(k1, k2, rows, pairs * np.uint32(2) + np.uint32(1))
    # Rounded to fp32 like __uint2float_rn.
    u1 = (a.astype(np.float32) + np.float32(0.5)) * np.float32(2.0**-32)
    u2 = b.astype(np.float32) * np.float32(2.0**-32)
    r = np.sqrt(np.float32(-2.0) * np.log(u1))
    t = np.float32(2 * math.pi) * u2
    z = np.stack([r * np.cos(t), r * np.sin(t)], axis=-1).reshape(len(rows), -1)
    start = c0 - 2 * (c0 // 2)
    return z[:, start : start + c1 - c0]


_SIGNS = np.array([-1.0, 1.0], dtype=np.float32)


def _rademacher_cpu(k1: int, k2: int, rows: np.ndarray, c0: int, c1: int) -> np.ndarray:
    """Column ``c`` is bit ``c % 32`` of the hash of ``c // 32``."""
    words = _hash(k1, k2, rows, np.arange(c0 // 32, (c1 + 31) // 32, dtype=np.uint32))
    bits = np.unpackbits(words.view(np.uint8), axis=-1, bitorder="little")
    start = c0 - 32 * (c0 // 32)
    return _SIGNS[bits[:, start : start + c1 - c0]]
