"""
Tests for server.py — post-processing functions (pixelate, background removal, prompt building).

Does NOT test model loading or generation (requires GPU).
"""

import os
import sys

import numpy as np
from PIL import Image

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import server


class TestBuildPrompt:
    def test_basic_prompt(self):
        result = server._build_prompt("a brave knight")
        assert "pixel art" in result
        assert "a brave knight" in result
        assert "game asset" in result

    def test_empty_prompt(self):
        result = server._build_prompt("")
        assert "pixel art" in result


class TestPixelate:
    def _make_image(self, size=512):
        arr = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
        return Image.fromarray(arr, mode="RGB")

    def test_pixelate_preserves_size(self):
        img = self._make_image(512)
        result = server._pixelate(img, pixel_size=4)
        assert result.size == (512, 512)

    def test_pixelate_zero_returns_original(self):
        img = self._make_image(64)
        result = server._pixelate(img, pixel_size=0)
        assert result.size == (64, 64)

    def test_pixelate_reduces_unique_colors(self):
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        for y in range(64):
            for x in range(64):
                arr[y, x] = [x * 4, y * 4, 128]
        img = Image.fromarray(arr, mode="RGB")

        result = server._pixelate(img, pixel_size=8)

        original_colors = len(np.unique(np.array(img).reshape(-1, 3), axis=0))
        result_colors = len(np.unique(np.array(result).reshape(-1, 3), axis=0))
        assert result_colors <= original_colors

    def test_pixelate_creates_blocks(self):
        arr = np.zeros((8, 8, 3), dtype=np.uint8)
        arr[:4, :4] = [255, 0, 0]
        arr[4:, 4:] = [0, 255, 0]
        img = Image.fromarray(arr, mode="RGB")

        result = server._pixelate(img, pixel_size=4)
        result_arr = np.array(result)

        block_tl = result_arr[:4, :4]
        assert np.all(block_tl == block_tl[0, 0])

        block_br = result_arr[4:, 4:]
        assert np.all(block_br == block_br[0, 0])


class TestRemoveBackground:
    def _make_sprite_on_white(self, size=64):
        arr = np.full((size, size, 3), 255, dtype=np.uint8)
        arr[16:48, 16:48] = [200, 50, 50]
        return Image.fromarray(arr, mode="RGB")

    def test_returns_rgba(self):
        img = self._make_sprite_on_white()
        result = server._remove_background(img)
        assert result.mode == "RGBA"

    def test_white_background_becomes_transparent(self):
        img = self._make_sprite_on_white()
        result = server._remove_background(img)
        arr = np.array(result)
        alpha = arr[:, :, 3]

        corner_alpha = alpha[0, 0]
        assert corner_alpha == 0

    def test_sprite_pixels_remain_opaque(self):
        img = self._make_sprite_on_white()
        result = server._remove_background(img)
        arr = np.array(result)
        alpha = arr[:, :, 3]

        center_alpha = alpha[32, 32]
        assert center_alpha == 255

    def test_preserves_sprite_colors(self):
        img = self._make_sprite_on_white()
        result = server._remove_background(img)
        arr = np.array(result)
        center_pixel = arr[32, 32]
        assert center_pixel[0] == 200
        assert center_pixel[1] == 50
        assert center_pixel[2] == 50

    def test_interior_bright_pixel_not_removed(self):
        # Sprite is dark red on white background, with a white highlight inside
        arr = np.full((64, 64, 3), 255, dtype=np.uint8)
        arr[8:56, 8:56] = [180, 30, 30]
        # White highlight inside the sprite (like a shine on armor)
        arr[30:34, 30:34] = [255, 255, 255]
        img = Image.fromarray(arr, mode="RGB")

        result = server._remove_background(img)
        result_arr = np.array(result)
        alpha = result_arr[:, :, 3]

        # Background corners should be transparent
        assert alpha[0, 0] == 0
        # Interior highlight should be opaque (not removed as background)
        assert alpha[32, 32] == 255
        # Sprite body should be opaque
        assert alpha[20, 20] == 255

    def test_black_background_removed(self):
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        arr[16:48, 16:48] = [200, 50, 50]
        img = Image.fromarray(arr, mode="RGB")

        result = server._remove_background(img)
        result_arr = np.array(result)
        alpha = result_arr[:, :, 3]

        assert alpha[0, 0] == 0
        assert alpha[32, 32] == 255


class TestEnsureDir:
    def test_ensure_dir_creates_nested(self, tmp_path):
        path = str(tmp_path / "a" / "b" / "c" / "test.png")
        server._ensure_dir(path)
        assert os.path.isdir(str(tmp_path / "a" / "b" / "c"))

    def test_ensure_dir_empty_dir(self):
        server._ensure_dir("test.png")


class TestEnvPaths:
    def test_default_model_dir(self):
        assert "models" in server.MODEL_DIR

    def test_default_lora_dir(self):
        assert "models" in server.LORA_DIR

    def test_default_output_dir(self):
        assert "output" in server.OUTPUT_DIR

    def test_default_db_path(self):
        assert "feedback.db" in server.DB_PATH
