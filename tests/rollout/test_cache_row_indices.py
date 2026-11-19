"""Row compaction preserves cache values and uses indices on the selected tensors' devices."""

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils._python_dispatch import TorchDispatchMode
from transformers import DynamicCache
from transformers.cache_utils import DynamicLayer, DynamicSlidingWindowLayer

from monorl.rollout.hf_engine import _select_cache_rows


def _cache(devices: tuple[str, str], dtype: torch.dtype = torch.float32) -> DynamicCache:
    cache = DynamicCache()
    cache.layers = [DynamicLayer(), DynamicSlidingWindowLayer(4), DynamicLayer()]
    base = torch.arange(4 * 2 * 7 * 3, dtype=torch.float32).reshape(4, 2, 7, 3)
    for i, device in enumerate(devices):
        keys = (base + 256 * i).to(dtype=dtype, device=device)
        values = (base + 512 * i + 11).to(dtype=dtype, device=device)
        cache.update(keys, values, i)
    return cache


def _assert_cache_equal(actual: DynamicCache, expected: DynamicCache) -> None:
    for left, right in zip(actual.layers, expected.layers, strict=True):
        assert left.is_initialized == right.is_initialized
        assert left.get_seq_length() == right.get_seq_length()
        assert left.get_mask_sizes(1) == right.get_mask_sizes(1)
        if right.is_initialized:
            torch.testing.assert_close(left.keys, right.keys, rtol=0, atol=0)
            torch.testing.assert_close(left.values, right.values, rtol=0, atol=0)
        else:
            assert left.keys is None and left.values is None


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "devices",
    [
        ("cpu", "cpu"),
        pytest.param(("cuda:0", "cuda:0"), marks=pytest.mark.gpu),
        pytest.param(("cuda:0", "cpu"), marks=pytest.mark.gpu),
        pytest.param(("cuda:0", "cuda:1"), marks=pytest.mark.gpu),
    ],
)
def test_compaction_preserves_rows_and_continued_cache_updates(devices, dtype):
    required_devices = max((torch.device(d).index + 1 for d in devices if d.startswith("cuda")), default=0)
    if torch.cuda.device_count() < required_devices:
        pytest.skip("requires the requested CUDA/HIP devices")
    actual = _cache(devices, dtype)
    expected = _cache(devices, dtype)
    sequences = torch.arange(32, device=devices[0]).reshape(4, 8)
    mask = torch.tensor(
        [[0, 0, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1]],
        device=devices[0],
    )
    for rows in ([3, 1], [1]):
        expected.batch_select_indices(torch.tensor(rows, dtype=torch.long))
        expected_sequences, expected_mask = sequences[rows], mask[rows]
        sequences, mask = _select_cache_rows(actual, rows, sequences, mask)
        torch.testing.assert_close(sequences, expected_sequences, rtol=0, atol=0)
        torch.testing.assert_close(mask, expected_mask, rtol=0, atol=0)
        _assert_cache_equal(actual, expected)

        # A sliding layer's logical length and retained window must still advance together.
        for i, device in enumerate(devices):
            keys = torch.full((len(rows), 2, 1, 3), 17 + i, dtype=dtype, device=device)
            values = keys + 3
            actual.update(keys, values, i)
            expected.update(keys, values, i)
        _assert_cache_equal(actual, expected)


class _IndexTrace(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.calls = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func == torch.ops.aten.index.Tensor:
            tensor, indices = args[:2]
            self.calls.extend((tensor, index) for index in indices if index is not None)
        elif func == torch.ops.aten.index_select.default:
            tensor, _, index = args[:3]
            self.calls.append((tensor, index))
        return func(*args, **(kwargs or {}))


@pytest.mark.parametrize("offload", [False, True])
def test_indices_follow_current_storage_and_are_reused_per_device(offload):
    # Meta tensors exercise heterogeneous residency without representing GPU execution.
    with FakeTensorMode():
        cache = _cache(("cpu", "meta"))
        if offload:
            cache.layers[1].offload()
            assert cache.layers[1].device == torch.device("meta")
            assert cache.layers[1].keys.device == torch.device("cpu")
        sequences = torch.empty((4, 8), dtype=torch.long, device="meta")
        mask = torch.empty((4, 6), dtype=torch.long, device="cpu")
        with _IndexTrace() as trace:
            selected_sequences, selected_mask = _select_cache_rows(cache, [3, 1], sequences, mask)
        assert selected_sequences.shape == (2, 8)
        assert selected_mask.shape == (2, 6)
        indices_by_device = {}
        for tensor, index in trace.calls:
            assert index.device == tensor.device
            assert index.dtype == torch.long
            previous = indices_by_device.setdefault(tensor.device, index)
            assert index is previous
        assert set(indices_by_device) == {torch.device("cpu"), torch.device("meta")}
        assert cache.layers[0].keys.shape[0] == cache.layers[1].keys.shape[0] == 2
        assert cache.layers[2].keys is None
        if offload:
            cache.layers[1].prefetch()
            assert cache.layers[1].keys.device == torch.device("meta")
            assert cache.layers[1].values.device == torch.device("meta")
            assert cache.layers[1].keys.shape == (2, 2, 3, 3)
