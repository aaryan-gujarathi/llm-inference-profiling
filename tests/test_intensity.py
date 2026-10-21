"""Unit tests for analysis.intensity."""

import pytest

from analysis.intensity import (
    T4,
    TINYLLAMA_LAYERS,
    LinearLayer,
    arithmetic_intensity,
    attainable_flops,
    is_memory_bound,
    layer_bytes,
    layer_flops,
)

GATE = LinearLayer("gate_proj", 2048, 5632)


def test_t4_ridge_point():
    assert T4.ridge_point == pytest.approx(203.125)


def test_flops_formula():
    assert layer_flops(GATE, batch=1) == 2 * 2048 * 5632


def test_batch1_4bit_intensity_is_about_4():
    """Core finding: at B=1 the K*N terms cancel, I -> 2 / w_bytes."""
    ai = arithmetic_intensity(GATE, batch=1, w_bytes=0.5)
    assert 3.9 < ai < 4.0


def test_intensity_independent_of_shape_at_batch1():
    vals = [arithmetic_intensity(l, batch=1) for l in TINYLLAMA_LAYERS]
    assert max(vals) - min(vals) < 0.15


def test_every_layer_memory_bound_at_batch1():
    for layer in TINYLLAMA_LAYERS:
        assert is_memory_bound(T4, arithmetic_intensity(layer, batch=1))


def test_large_batch_crosses_ridge():
    assert not is_memory_bound(T4, arithmetic_intensity(GATE, batch=256))


def test_fp16_weights_halve_batch1_intensity():
    ai_fp16 = arithmetic_intensity(GATE, batch=1, w_bytes=2.0)
    assert ai_fp16 == pytest.approx(1.0, abs=0.01)


def test_attainable_caps_at_peak():
    assert attainable_flops(T4, 1e6) == T4.peak_flops
    assert attainable_flops(T4, 4.0) == pytest.approx(4.0 * T4.mem_bw)


@pytest.mark.parametrize("batch", [0, -1])
def test_rejects_bad_batch(batch):
    with pytest.raises(ValueError):
        layer_bytes(GATE, batch=batch)


def test_rejects_nonpositive_intensity():
    with pytest.raises(ValueError):
        attainable_flops(T4, 0)
