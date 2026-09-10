"""马赛克模块测试"""

import numpy as np

from maskperson.mosaic import (
    apply_mosaic,
    create_checkerboard_mosaic,
    create_pixel_mosaic,
    create_solid_black_mosaic,
    expand_mask,
)


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


class TestCreateSolidBlackMosaic:
    """最强脱敏风格：实心黑块。"""

    def test_full_mask_becomes_all_black(self):
        img = np.full((20, 20, 3), 200, dtype=np.uint8)
        mask = np.ones((20, 20), dtype=np.float32)
        result = create_solid_black_mosaic(img, mask)
        np.testing.assert_array_equal(result, np.zeros_like(img))

    def test_empty_mask_preserves_image(self):
        img = np.full((20, 20, 3), 200, dtype=np.uint8)
        mask = np.zeros((20, 20), dtype=np.float32)
        np.testing.assert_array_equal(create_solid_black_mosaic(img, mask), img)

    def test_partial_mask_only_blackens_region(self):
        img = np.full((10, 10, 3), 200, dtype=np.uint8)
        mask = np.zeros((10, 10), dtype=np.float32)
        mask[2:8, 2:8] = 1.0
        result = create_solid_black_mosaic(img, mask)
        np.testing.assert_array_equal(result[2:8, 2:8], 0)
        np.testing.assert_array_equal(result[0], img[0])

    def test_expand_grows_black_region(self):
        img = np.full((10, 10, 3), 200, dtype=np.uint8)
        mask = np.zeros((10, 10), dtype=np.float32)
        mask[5, 5] = 1.0
        result = create_solid_black_mosaic(img, mask, expand_pixels=2)
        # 中心像素周围应被黑色覆盖（膨胀后）
        assert result[5, 5, 0] == 0
        assert result[4, 5, 0] == 0
        # 远离中心的像素保持原值
        np.testing.assert_array_equal(result[0], 200)


class TestCreateCheckerboardMosaic:
    """黑白棋盘格风格。"""

    def test_full_mask_alternates(self):
        img = np.full((20, 20, 3), 200, dtype=np.uint8)
        mask = np.ones((20, 20), dtype=np.float32)
        result = create_checkerboard_mosaic(img, mask, block_size=10)
        # 块 (0,0) 应为黑、(0,1) 白、(1,0) 白、(1,1) 黑
        np.testing.assert_array_equal(result[0:10, 0:10], 0)
        np.testing.assert_array_equal(result[0:10, 10:20], 255)
        np.testing.assert_array_equal(result[10:20, 0:10], 255)
        np.testing.assert_array_equal(result[10:20, 10:20], 0)

    def test_empty_mask_preserves_image(self):
        img = np.full((20, 20, 3), 200, dtype=np.uint8)
        mask = np.zeros((20, 20), dtype=np.float32)
        np.testing.assert_array_equal(
            create_checkerboard_mosaic(img, mask, block_size=10), img,
        )

    def test_block_size_zero_falls_back_to_one(self):
        """block_size<=0 应被当作 1（不崩溃）。"""
        img = np.full((10, 10, 3), 200, dtype=np.uint8)
        mask = np.ones((10, 10), dtype=np.float32)
        # 不抛异常即视为通过
        create_checkerboard_mosaic(img, mask, block_size=0)


class TestApplyMosaic:
    """apply_mosaic 工厂函数的分派逻辑。"""

    def test_pixel_dispatches_to_pixel(self):
        img = np.full((10, 10, 3), 100, dtype=np.uint8)
        mask = np.ones((10, 10), dtype=np.float32)
        result = apply_mosaic(img, mask, "pixel", block_size=10)
        # block_size=10 全图为均值
        np.testing.assert_allclose(result[0, 0].astype(int), [100, 100, 100], atol=1)

    def test_solid_black_dispatches_to_solid_black(self):
        img = np.full((10, 10, 3), 100, dtype=np.uint8)
        mask = np.ones((10, 10), dtype=np.float32)
        result = apply_mosaic(img, mask, "solid_black", block_size=10)
        np.testing.assert_array_equal(result, 0)

    def test_checkerboard_dispatches_to_checkerboard(self):
        img = np.full((10, 10, 3), 100, dtype=np.uint8)
        mask = np.ones((10, 10), dtype=np.float32)
        result = apply_mosaic(img, mask, "checkerboard", block_size=5)
        # (0,0) 黑、(0,5) 白、(5,0) 白、(5,5) 黑
        np.testing.assert_array_equal(result[0:5, 0:5], 0)
        np.testing.assert_array_equal(result[0:5, 5:10], 255)
        np.testing.assert_array_equal(result[5:10, 0:5], 255)
        np.testing.assert_array_equal(result[5:10, 5:10], 0)

    def test_unknown_style_raises(self):
        img = np.full((10, 10, 3), 100, dtype=np.uint8)
        mask = np.ones((10, 10), dtype=np.float32)
        with np.testing.assert_raises(ValueError):
            apply_mosaic(img, mask, "nonsense", block_size=5)
