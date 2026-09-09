"""跟踪器模块测试"""

import numpy as np

from maskperson.tracker import TrackCache


class TestTrackCache:
    def test_first_frame(self):
        """首帧返回原始 mask"""
        cache = TrackCache(smooth_window_size=5, max_age=30)
        mask = np.random.rand(100, 100)
        result = cache.update(track_id=1, mask=mask, frame_idx=0)
        np.testing.assert_array_almost_equal(result, mask > 0.5)

    def test_smooth_window(self):
        """窗口内多帧做平均"""
        cache = TrackCache(smooth_window_size=3, max_age=30)
        # 第 0 帧：上半部分有内容
        mask0 = np.zeros((10, 10))
        mask0[:5, :] = 1.0
        # 第 1、2 帧：全 1
        mask1 = np.ones((10, 10))
        mask2 = np.ones((10, 10))

        cache.update(1, mask0, 0)
        cache.update(1, mask1, 1)
        result = cache.update(1, mask2, 2)

        # 平均后阈值化
        expected = (mask0 + mask1 + mask2) / 3 > 0.5
        np.testing.assert_array_equal(result, expected)

    def test_old_tracks_cleaned(self):
        """超过 max_age 的 track 被清理"""
        cache = TrackCache(smooth_window_size=5, max_age=5)
        mask = np.ones((10, 10))
        for i in range(10):
            cache.update(i, mask, i)
        cache.clean_old(frame_idx=20)
        assert cache.active_count == 0

    def test_keep_recent_tracks(self):
        """max_age 内重新出现的 track 不清理"""
        cache = TrackCache(smooth_window_size=5, max_age=5)
        mask = np.ones((10, 10))
        cache.update(1, mask, 0)
        cache.update(2, mask, 1)
        cache.update(3, mask, 5)
        cache.update(1, mask, 8)  # track 1 重新出现
        cache.clean_old(frame_idx=10)
        assert 1 in cache.mask_history
        assert 1 in cache.last_seen

    def test_multiple_tracks(self):
        """多目标独立跟踪"""
        cache = TrackCache(smooth_window_size=3, max_age=10)
        mask1 = np.zeros((10, 10))
        mask1[:3, :] = 1.0
        mask2 = np.ones((10, 10))

        cache.update(10, mask1, 0)
        cache.update(20, mask2, 0)

        assert cache.active_count == 2
        assert 10 in cache.mask_history
        assert 20 in cache.mask_history

    def test_window_overflow(self):
        """历史窗口溢出时丢弃最早的帧"""
        cache = TrackCache(smooth_window_size=2, max_age=30)
        for i in range(5):
            m = np.full((10, 10), float(i))
            cache.update(1, m, i)
        # 窗口只保留最近 2 帧: i=3, i=4
        assert len(cache.mask_history[1]) == 2
        np.testing.assert_array_almost_equal(cache.mask_history[1][0], 3.0)
        np.testing.assert_array_almost_equal(cache.mask_history[1][1], 4.0)
