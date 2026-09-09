"""视频模块测试"""

import subprocess
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from maskperson.video import (
    VideoCapture,
    VideoWriter,
    extract_audio,
    has_audio_stream,
    merge_av,
)


class TestVideoCapture:
    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            VideoCapture(str(tmp_path / "nonexistent.mp4"))

    @patch("maskperson.video.cv2.VideoCapture")
    def test_info_property(self, mock_cap_cls):
        mock_cap = mock_cap_cls.return_value
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = [30.0, 1920, 1080, 1000]

        cap = VideoCapture("fake.mp4")
        info = cap.info

        assert info.fps == 30.0
        assert info.width == 1920
        assert info.height == 1080
        assert info.frame_count == 1000


class TestVideoWriter:
    @patch("maskperson.video.cv2.VideoWriter")
    def test_context_manager(self, mock_writer_cls, tmp_path: Path):
        mock_writer = mock_writer_cls.return_value
        mock_writer.isOpened.return_value = True

        path = tmp_path / "out.mp4"
        with VideoWriter(str(path), fps=30.0, width=640, height=480) as writer:
            writer.write(np.zeros((480, 640, 3), dtype=np.uint8))

        assert mock_writer.write.called
        assert mock_writer.release.called


class TestExtractAudio:
    @patch("maskperson.video.has_audio_stream", return_value=True)
    @patch("subprocess.run")
    def test_calls_ffmpeg(self, mock_run, _mock_probe, tmp_path: Path):
        out_audio = tmp_path / "audio.aac"
        mock_run.return_value = None

        extract_audio("input.mp4", str(out_audio))

        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert "ffmpeg" in args
        assert "-i" in args
        assert "input.mp4" in args
        assert "-vn" in args

    @patch("maskperson.video.has_audio_stream", return_value=False)
    @patch("subprocess.run")
    def test_skips_when_no_audio(self, mock_run, _mock_probe, tmp_path: Path):
        """无音频轨时不应调用 ffmpeg"""
        out_audio = tmp_path / "audio.aac"
        result = extract_audio("input.mp4", str(out_audio))
        assert result is False
        mock_run.assert_not_called()


class TestMergeAV:
    @patch("subprocess.run")
    def test_calls_ffmpeg(self, mock_run, tmp_path: Path):
        out = tmp_path / "merged.mp4"
        mock_run.return_value = None

        merge_av("video.mp4", "audio.aac", str(out), fps=30.0, width=1920, height=1080)

        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert "ffmpeg" in args
        assert "video.mp4" in args
        assert "audio.aac" in args
        assert "libx264" in args

    @patch("subprocess.run")
    def test_no_audio_skips_audio_args(self, mock_run, tmp_path: Path):
        """audio_path 为空时不加 -i audio / -c:a"""
        out = tmp_path / "merged.mp4"
        mock_run.return_value = None

        merge_av("video.mp4", "", str(out), fps=30.0, width=1920, height=1080)

        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert "video.mp4" in args
        assert "libx264" in args
        assert "-c:a" not in args


class TestHasAudioStream:
    @patch("subprocess.run")
    def test_true_when_stream_present(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="0\n", stderr=""
        )
        assert has_audio_stream("input.mp4") is True

    @patch("subprocess.run")
    def test_false_when_no_stream(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
        assert has_audio_stream("input.mp4") is False

    @patch("subprocess.run", side_effect=FileNotFoundError)
    def test_false_when_ffprobe_missing(self, _mock_run):
        assert has_audio_stream("input.mp4") is False
