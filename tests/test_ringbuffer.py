import threading
import numpy as np
import pytest

from stream_voice_focus.ringbuffer import RingBuffer


def test_push_pop_roundtrip_and_wrap():
    rb = RingBuffer(1000, 1)
    x = np.arange(700, dtype=np.float32)
    assert rb.push(x) == 0
    assert rb.available == 700
    y = rb.pop(500)[:, 0]
    assert np.array_equal(y, x[:500])
    x2 = np.arange(700, 1400, dtype=np.float32)
    assert rb.push(x2) == 0            # wraps around the end
    assert rb.available == 900
    y = rb.pop(900)[:, 0]
    assert np.array_equal(y, np.arange(500, 1400, dtype=np.float32))


def test_bounded_never_grows():
    rb = RingBuffer(480, 2)
    dropped = rb.push(np.ones((1000, 2), np.float32))
    assert dropped == 520
    assert rb.available == 480
    assert rb.free == 0
    assert rb.push(np.ones((10, 2), np.float32)) == 10


def test_pop_too_many_raises_and_skip():
    rb = RingBuffer(100)
    rb.push(np.zeros(50, np.float32))
    with pytest.raises(ValueError):
        rb.pop(60)
    assert rb.skip(30) == 30
    assert rb.available == 20
    rb.clear()
    assert rb.available == 0


def test_spsc_threads_preserve_order():
    rb = RingBuffer(4096)
    n = 200000
    src = np.arange(n, dtype=np.float32)
    got = []

    def producer():
        i = 0
        while i < n:
            k = min(333, n - i)
            d = rb.push(src[i:i + k])
            i += k - d

    def consumer():
        total = 0
        while total < n:
            a = rb.available
            if a:
                k = min(a, 257)
                got.append(rb.pop(k)[:, 0].copy())
                total += k

    t1, t2 = threading.Thread(target=producer), threading.Thread(target=consumer)
    t1.start(); t2.start(); t1.join(); t2.join()
    out = np.concatenate(got)
    assert np.array_equal(out, src)
