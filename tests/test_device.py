"""device 模块测试 — 通过 mock torch.cuda 验证设备选择逻辑。"""

from unittest.mock import patch

import pytest

from maskperson.device import resolve_device


class TestResolveDeviceAuto:
    """auto 模式：CUDA 可用时优先，否则降级 CPU。"""

    def test_auto_with_cuda_available_returns_cuda(self):
        with patch("torch.cuda.is_available", return_value=True):
            assert resolve_device("auto") == "cuda"

    def test_auto_without_cuda_falls_back_cpu(self):
        with patch("torch.cuda.is_available", return_value=False):
            assert resolve_device("auto") == "cpu"


class TestResolveDeviceExplicit:
    """显式指定设备：按需校验。"""

    def test_explicit_cpu_works_without_cuda(self):
        with patch("torch.cuda.is_available", return_value=False):
            assert resolve_device("cpu") == "cpu"

    def test_explicit_cuda_returns_cuda_when_available(self):
        with patch("torch.cuda.is_available", return_value=True):
            assert resolve_device("cuda") == "cuda"

    def test_explicit_cuda_raises_when_unavailable(self):
        with patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(RuntimeError, match="CUDA"):
                resolve_device("cuda")

    def test_explicit_cuda_raises_with_actionable_message(self):
        """报错信息需包含排查指引。"""
        with patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(RuntimeError) as exc_info:
                resolve_device("cuda")
            msg = str(exc_info.value)
            assert "torch.cuda.is_available" in msg
            assert "NVIDIA" in msg


class TestResolveDeviceCudaIndex:
    """cuda:N 索引模式。"""

    def test_cuda_index_within_range(self):
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.device_count", return_value=2):
            assert resolve_device("cuda:0") == "cuda:0"
            assert resolve_device("cuda:1") == "cuda:1"

    def test_cuda_index_out_of_range_raises(self):
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.device_count", return_value=2):
            with pytest.raises(RuntimeError, match="超出可用设备数"):
                resolve_device("cuda:5")

    def test_cuda_index_non_integer_raises(self):
        with patch("torch.cuda.is_available", return_value=True):
            with pytest.raises(ValueError, match="索引必须为整数"):
                resolve_device("cuda:abc")

    def test_cuda_index_unavailable_raises(self):
        with patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(RuntimeError, match="CUDA"):
                resolve_device("cuda:0")


class TestResolveDeviceInvalidMode:
    """非法 mode 应抛 ValueError。"""

    @pytest.mark.parametrize("bad_mode", ["gpu", "tpu", "cudaXYZ", "", "0"])
    def test_unknown_mode_raises_value_error(self, bad_mode):
        with pytest.raises(ValueError, match="未知 device"):
            resolve_device(bad_mode)