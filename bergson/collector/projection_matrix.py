"""Random projection matrices that are the same on every device.

Each entry is computed from a keyed hash of its position, so the values don't
depend on the device or on how the work is split across threads. Rademacher
matrices are bit-identical everywhere; normal matrices can differ in the last
bits because ``log``, ``sin`` and ``cos`` round differently across devices.
"""

import hashlib
import math
from typing import Callable, Literal

import torch
from torch import Tensor

from bergson.utils.logger import get_logger

logger = get_logger("projection_matrix", level="INFO")

_M1 = 0x7FEB352D
_M2 = 0x846CA68B - (1 << 32)
_TWO_PI = 2 * math.pi
_INV_2_32 = 1.0 / (1 << 32)

# Sizes that keep each CPU chunk's temporaries in cache.
_CPU_PAIRS = 1 << 18
_CPU_WORDS = 1 << 16

# Written as one function because the jiterator only accepts a single template
# function. lowbias32 is Chris Wellons' 32-bit integer hash; the second round
# is keyed so that different keys don't give shuffled copies of the same values.
_HASH = """
    auto lowbias32 = [](unsigned int v) {
        v ^= v >> 16; v *= 0x7feb352du; v ^= v >> 15; v *= 0x846ca68bu; v ^= v >> 16;
        return v;
    };
    auto keyed = [&](unsigned int c) {
        return lowbias32(lowbias32(c ^ static_cast<unsigned int>(k1))
                         + static_cast<unsigned int>(k2));
    };
    unsigned int i = static_cast<unsigned int>(row) * static_cast<unsigned int>(cols)
                     + static_cast<unsigned int>(col);
"""
_NORMAL_KERNEL = (
    "template <typename T> T hashed_normal(T row, T col, T cols, T k1, T k2) {"
    + _HASH
    + """
    unsigned int a = keyed(i & ~1u);
    unsigned int b = keyed(i | 1u);
    float u1 = (__uint2float_rn(a) + 0.5f) * 2.3283064365386963e-10f;
    float u2 = __uint2float_rn(b) * 2.3283064365386963e-10f;
    float r = sqrtf(-2.0f * logf(u1));
    float t = 6.2831853071795864769f * u2;
    float z = (i & 1u) ? r * sinf(t) : r * cosf(t);
    return static_cast<T>(__float_as_int(z));
}
"""
)
_RADEMACHER_KERNEL = (
    "template <typename T> T hashed_rademacher(T row, T col, T cols, T k1, T k2) {"
    + _HASH
    + """
    float z = ((keyed(i >> 5) >> (i & 31u)) & 1u) ? 1.0f : -1.0f;
    return static_cast<T>(__float_as_int(z));
}
"""
)
_kernels: dict[str, Callable] = {}
_kernels_failed = False


def random_matrix(
    identifier: str,
    m: int,
    n: int,
    device: torch.device | str,
    projection_type: Literal["normal", "rademacher"],
) -> Tensor:
    """An fp32 ``[m, n]`` matrix of standard normal or ±1 entries determined by
    ``identifier``."""
    if projection_type not in ("normal", "rademacher"):
        raise ValueError(f"Unknown projection type: {projection_type}")
    if m * n >= 1 << 31:
        raise ValueError(
            f"Projection matrices are limited to 2^31 entries, got [{m}, {n}]."
        )

    device = torch.device(device)
    k1, k2 = _keys(identifier)
    if device.type == "cuda":
        kernel = _kernel(projection_type)
        if kernel is not None:
            rows = torch.arange(m, dtype=torch.int32, device=device)[:, None]
            cols = torch.arange(n, dtype=torch.int32, device=device)[None, :]
            return kernel(rows, cols, cols=n, k1=k1, k2=k2).view(torch.float32)

    if projection_type == "normal":
        A = _normal_cpu(k1, k2, m * n)
    else:
        A = _rademacher_cpu(k1, k2, m * n)
    return A.view(m, n).to(device)


def _keys(identifier: str) -> tuple[int, int]:
    digest = hashlib.md5(identifier.encode()).digest()
    return (
        int.from_bytes(digest[:4], "little", signed=True),
        int.from_bytes(digest[4:8], "little", signed=True),
    )


def _kernel(projection_type: str) -> Callable | None:
    """The fused CUDA/ROCm kernel, or ``None`` if it can't be compiled here."""
    global _kernels_failed
    if _kernels_failed:
        return None
    if not _kernels:
        try:
            from torch.cuda.jiterator import _create_jit_fn

            _kernels["normal"] = _create_jit_fn(_NORMAL_KERNEL, cols=0, k1=0, k2=0)
            _kernels["rademacher"] = _create_jit_fn(
                _RADEMACHER_KERNEL, cols=0, k1=0, k2=0
            )
            # Compile now so a failure falls back instead of surfacing later.
            one = torch.zeros(1, 1, dtype=torch.int32, device="cuda")
            for kernel in _kernels.values():
                kernel(one, one, cols=1, k1=0, k2=0)
        except Exception as e:
            _kernels.clear()
            _kernels_failed = True
            logger.warning(
                f"Couldn't compile the projection matrix kernel ({e}); generating "
                "projection matrices on the CPU instead."
            )
            return None
    return _kernels[projection_type]


def _hash_(x: Tensor, k1: int, k2: int) -> Tensor:
    """The kernel's keyed hash, in place on int32 as if unsigned."""
    tmp = torch.empty_like(x)
    x ^= k1
    _lowbias32_(x, tmp)
    x += k2
    _lowbias32_(x, tmp)
    return x


def _lowbias32_(x: Tensor, tmp: Tensor) -> None:
    # Arithmetic right shifts plus a mask give the logical shifts of uint32.
    torch.bitwise_right_shift(x, 16, out=tmp)
    tmp &= 0xFFFF
    x ^= tmp
    x.mul_(_M1)
    torch.bitwise_right_shift(x, 15, out=tmp)
    tmp &= 0x1FFFF
    x ^= tmp
    x.mul_(_M2)
    torch.bitwise_right_shift(x, 16, out=tmp)
    tmp &= 0xFFFF
    x ^= tmp


def _normal_cpu(k1: int, k2: int, numel: int) -> Tensor:
    """Entries ``2p`` and ``2p + 1`` share a Box-Muller pair built from the
    hashes of ``2p`` and ``2p + 1``."""
    out = torch.empty(numel)
    step = 2 * _CPU_PAIRS
    for lo in range(0, numel, step):
        hi = min(numel, lo + step)
        pairs = torch.arange(lo // 2, (hi + 1) // 2, dtype=torch.int32)
        a = _hash_(pairs * 2, k1, k2)
        b = _hash_(pairs * 2 + 1, k1, k2)
        # Round each hash, read as uint32, to fp32 like __uint2float_rn.
        u1 = ((a.long() & 0xFFFFFFFF).float() + 0.5) * _INV_2_32
        u2 = (b.long() & 0xFFFFFFFF).float() * _INV_2_32
        r = torch.sqrt(-2.0 * torch.log(u1))
        t = _TWO_PI * u2
        z = torch.stack([r * torch.cos(t), r * torch.sin(t)], dim=1).view(-1)
        out[lo:hi] = z[: hi - lo]
    return out


def _rademacher_cpu(k1: int, k2: int, numel: int) -> Tensor:
    """Entry ``i`` is bit ``i % 32`` of the hash of ``i // 32``."""
    out = torch.empty(numel)
    shifts = torch.arange(32, dtype=torch.int32)
    step = 32 * _CPU_WORDS
    for lo in range(0, numel, step):
        hi = min(numel, lo + step)
        words = _hash_(
            torch.arange(lo // 32, (hi + 31) // 32, dtype=torch.int32), k1, k2
        )
        bits = torch.bitwise_right_shift(words[:, None], shifts) & 1
        out[lo:hi] = bits.view(-1)[: hi - lo] * 2 - 1
    return out
