"""马赛克模块测试"""

import numpy as np

from maskperson.mosaic import create_pixel_mosaic, expand_mask


class TestCreatePixelMosaic:
    def test_no_mask(self):
        """全零 mask 不应修改图像"""
        img = np.full((100, 100, 3), 255, dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.float32)
        result = create_pixel_mosaic(img, mask, block_size=10)
        np.testing.assert_array_equal(result, img)

    def test_full_mask(self):
        """全 1 mask 应全部变为整图均值（允许 1 单位舍入误差）"""
        img = np.random.randint(50, 200, (10, 10, 3), dtype=np.uint8)
        mask = np.ones((10, 10), dtype=np.float32)
        result = create_pixel_mosaic(img, mask, block_size=10)
        # block_size=10 时整图只有一个块，所有像素都是整图均值
        expected = np.mean(img, axis=(0, 1))
        # 新实现用 np.round 而非 truncate，允许 ±1 误差
        np.testing.assert_allclose(result[0, 0].astype(np.int32),
                                   expected.astype(np.int32), atol=1)

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

    def test_non_divisible_dimensions(self):
        """尺寸不整除 block_size 时不崩溃，剩余行/列保留原值"""
        img = np.full((23, 27, 3), 100, dtype=np.uint8)
        mask = np.ones((23, 27), dtype=np.float32)
        result = create_pixel_mosaic(img, mask, block_size=10)
        assert result.shape == img.shape
        # 主要区域是均值 100
        np.testing.assert_array_equal(result[:20, :20], 100)
        # 不抛异常即视为通过

    def test_invalid_block_size(self):
        """block_size <= 0 应抛 ValueError"""
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        mask = np.zeros((10, 10), dtype=np.float32)
        with np.testing.assert_raises(ValueError):
            create_pixel_mosaic(img, mask, block_size=0)


class TestExpandMask:
    def test_zero_expand_is_identity(self):
        mask = np.zeros((20, 20), dtype=np.float32)
        mask[5:15, 5:15] = 1.0
        out = expand_mask(mask, expand_pixels=0)
        np.testing.assert_array_equal(out, mask.astype(bool))

    def test_positive_expand_grows_mask(self):
        mask = np.zeros((40, 40), dtype=np.float32)
        mask[20, 20] = 1.0  # 单像素
        out = expand_mask(mask, expand_pixels=3)
        # 膨胀后应该有显著大于 1 像素的 True 区域
        assert out.sum() > 10
        assert out[20, 20]  # 原点保留
