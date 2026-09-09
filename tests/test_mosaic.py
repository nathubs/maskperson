"""马赛克模块测试"""

import numpy as np

from maskperson.mosaic import create_pixel_mosaic


class TestCreatePixelMosaic:
    def test_no_mask(self):
        """全零 mask 不应修改图像"""
        img = np.full((100, 100, 3), 255, dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.float32)
        result = create_pixel_mosaic(img, mask, block_size=10)
        np.testing.assert_array_equal(result, img)

    def test_full_mask(self):
        """全 1 mask 应全部变为整图均值"""
        img = np.random.randint(50, 200, (10, 10, 3), dtype=np.uint8)
        mask = np.ones((10, 10), dtype=np.float32)
        result = create_pixel_mosaic(img, mask, block_size=10)
        # block_size=10 时整图只有一个块，所有像素都是整图均值
        expected = np.mean(img, axis=(0, 1)).astype(np.uint8)
        np.testing.assert_array_equal(result[0, 0], expected)

    def test_partial_mask(self):
        """部分 mask 只处理对应区域"""
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        img[:, :5] = 100
        img[:, 5:] = 200
        mask = np.zeros((10, 10), dtype=np.float32)
        mask[:, 5:] = 1.0
        result = create_pixel_mosaic(img, mask, block_size=5)
        # 右侧应被马赛克化
        assert result[0, 8].tolist() == result[1, 8].tolist()
        # 左侧保持原样
        np.testing.assert_array_equal(result[:, 2], 100)

    def test_block_size_1(self):
        """block_size=1 等价于恒等变换"""
        img = np.random.randint(0, 255, (20, 20, 3), dtype=np.uint8)
        mask = np.ones((20, 20), dtype=np.float32)
        result = create_pixel_mosaic(img, mask, block_size=1)
        np.testing.assert_array_equal(result, img)
