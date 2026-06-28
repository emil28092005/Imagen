#!/usr/bin/env python3
"""
MCP server for generating pixel-art sprites using FLUX.2-klein-4B + pixel-art-lora.

Tools:
  - generate_sprite: Generate a single pixel-art sprite
  - batch_generate:  Generate multiple sprites in one call

Model is loaded lazily on first call (~6s), then stays in VRAM for speed.
Background is removed post-generation to produce transparent PNG.
"""

import os
import sys
import time
from typing import Optional

import numpy as np
from PIL import Image
from mcp.server.fastmcp import FastMCP

# Paths — models live in a shared location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.environ.get(
    "IMAGEGEN_MODEL_DIR",
    os.path.join(os.path.expanduser("~"), "models", "flux2-klein-4b"),
)
LORA_DIR = os.environ.get(
    "IMAGEGEN_LORA_DIR",
    os.path.join(os.path.expanduser("~"), "models", "pixel-art-lora"),
)
OUTPUT_DIR = os.environ.get("IMAGEGEN_OUTPUT_DIR", os.path.join(BASE_DIR, "output"))

# rsLoRA requires much lower scale in diffusers — 1.0 produces black images
LORA_SCALE = 0.1

# Global state — model loaded lazily
_pipe = None
_device = None


def _get_device():
    global _device
    if _device is None:
        import torch

        if torch.cuda.is_available():
            _device = "cuda"
        else:
            _device = "cpu"
            sys.stderr.write(
                "[pixel-art] WARNING: CUDA not available, using CPU (very slow)\n"
            )
    return _device


def _load_model():
    global _pipe
    if _pipe is not None:
        return _pipe

    sys.stderr.write("[pixel-art] Loading FLUX.2-klein-4B + LoRA (first call)...\n")
    t0 = time.time()

    import torch
    from diffusers import Flux2KleinPipeline

    _pipe = Flux2KleinPipeline.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.bfloat16,
    )
    _pipe.load_lora_weights(LORA_DIR)

    if _get_device() == "cuda":
        _pipe.enable_model_cpu_offload()
    else:
        _pipe.to(_get_device())

    elapsed = time.time() - t0
    sys.stderr.write(f"[pixel-art] Model loaded in {elapsed:.1f}s\n")
    return _pipe


def _build_prompt(user_prompt: str) -> str:
    return f"pixel art sprite, {user_prompt}, game asset, transparent background"


def _generate(
    pipe, prompt: str, seed: Optional[int], width: int, height: int, steps: int
):
    import torch

    generator = None
    if seed is not None:
        generator = torch.Generator(device=_get_device()).manual_seed(seed)

    image = pipe(
        prompt=prompt,
        num_inference_steps=steps,
        guidance_scale=1.0,
        height=height,
        width=width,
        generator=generator,
        attention_kwargs={"scale": LORA_SCALE},
    ).images[0]

    return image


def _remove_background(image: Image.Image, threshold: int = 30) -> Image.Image:
    """Remove background using flood-fill from edges.

    Two-pass approach:
    1. Detect border color, replace all near-border pixels with a flat fill color
    2. Flood-fill from edges to remove the flat color cleanly

    This normalizes gradient/noisy backgrounds into one solid color,
    making flood-fill removal much cleaner.
    """
    from collections import deque

    rgb = image.convert("RGB")
    arr = np.array(rgb).astype(int)
    h, w = arr.shape[:2]

    # Sample border colors from all 4 edges
    border_colors = []
    for x in range(w):
        border_colors.append(arr[0, x])
        border_colors.append(arr[h - 1, x])
    for y in range(h):
        border_colors.append(arr[y, 0])
        border_colors.append(arr[y, w - 1])

    border_colors = np.array(border_colors)
    bg_color = np.median(border_colors, axis=0).astype(int)

    # Pass 1: normalize background — replace all pixels within threshold
    # of border color with a flat fill color (pure magenta, unlikely in sprites)
    fill_color = np.array([255, 0, 255], dtype=int)
    dist_to_bg = np.abs(arr - bg_color).sum(axis=2)
    bg_mask = dist_to_bg < threshold * 3
    arr[bg_mask] = fill_color

    # Pass 2: flood-fill from edges to remove connected fill_color regions
    alpha = np.full((h, w), 255, dtype=np.uint8)
    visited = np.zeros((h, w), dtype=bool)
    queue = deque()

    fill_dist_threshold = 30  # tolerance for near-fill pixels

    # Seed from all border pixels
    for x in range(w):
        for y in [0, h - 1]:
            if not visited[y, x]:
                queue.append((y, x))
                visited[y, x] = True
    for y in range(h):
        for x in [0, w - 1]:
            if not visited[y, x]:
                queue.append((y, x))
                visited[y, x] = True

    # BFS flood-fill
    while queue:
        y, x = queue.popleft()
        dist = np.abs(arr[y, x] - fill_color).sum()
        if dist > fill_dist_threshold:
            continue
        alpha[y, x] = 0

        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                visited[ny, nx] = True
                queue.append((ny, nx))

    # Clean up: any remaining near-magenta pixels that weren't flood-filled
    # (small isolated background pockets) get removed too
    remaining_bg = np.abs(arr - fill_color).sum(axis=2) < fill_dist_threshold
    alpha[remaining_bg] = 0

    rgba = np.dstack([arr.astype(np.uint8), alpha])
    return Image.fromarray(rgba, mode="RGBA")


def _pixelate(image: Image.Image, pixel_size: int = 8) -> Image.Image:
    """Downscale then upscale with NEAREST to create chunky pixel-art effect.

    pixel_size=8 means each "pixel" in the result is an 8x8 block.
    """
    w, h = image.size
    small = image.resize((w // pixel_size, h // pixel_size), Image.LANCZOS)
    return small.resize((w, h), Image.NEAREST)


def _ensure_dir(path: str):
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)


# Create MCP server
mcp = FastMCP("pixel-art")


@mcp.tool()
def generate_sprite(
    prompt: str,
    output_path: str,
    seed: Optional[int] = None,
    width: int = 512,
    height: int = 512,
    steps: int = 4,
    remove_bg: bool = True,
    pixel_size: int = 4,
) -> dict:
    """Generate a pixel-art sprite and save it as PNG with transparent background.

    Args:
        prompt:       Description of the sprite (e.g. "a crystal warrior with geometric armor")
        output_path:  Where to save the PNG file (relative to output dir or absolute)
        seed:         Optional seed for reproducibility
        width:        Image width in pixels (default 512)
        height:       Image height in pixels (default 512)
        steps:        Inference steps (default 4, FLUX.2-klein is distilled)
        remove_bg:    Remove background and make transparent (default True)
        pixel_size:   Size of each pixel block for pixel-art effect (default 4, 0=off)

    Returns:
        Dict with output_path, seed_used, generation_time, prompt, size.
    """
    pipe = _load_model()
    full_prompt = _build_prompt(prompt)

    if not os.path.isabs(output_path):
        output_path = os.path.join(OUTPUT_DIR, output_path)

    _ensure_dir(output_path)

    t0 = time.time()
    image = _generate(pipe, full_prompt, seed, width, height, steps)

    if pixel_size > 0:
        image = _pixelate(image, pixel_size)

    if remove_bg:
        image = _remove_background(image)

    image.save(output_path)
    elapsed = time.time() - t0

    return {
        "output_path": output_path,
        "seed_used": seed,
        "generation_time": f"{elapsed:.1f}s",
        "prompt": full_prompt,
        "size": f"{width}x{height}",
        "transparent": remove_bg,
        "pixel_size": pixel_size,
    }


@mcp.tool()
def batch_generate(
    specs: list[dict],
) -> list[dict]:
    """Generate multiple pixel-art sprites in one call.

    Args:
        specs: List of dicts, each with:
          - prompt:      str   (required) — sprite description
          - output_path: str   (required) — PNG save path
          - seed:        int   (optional)
          - width:       int   (optional, default 512)
          - height:      int   (optional, default 512)
          - steps:       int   (optional, default 4)
          - remove_bg:   bool  (optional, default True)
          - pixel_size:  int   (optional, default 4, 0=off)

    Returns:
        List of dicts with output_path, seed_used, generation_time, prompt, size, transparent.
    """
    pipe = _load_model()
    results = []

    for spec in specs:
        prompt = spec["prompt"]
        output_path = spec["output_path"]
        seed = spec.get("seed")
        width = spec.get("width", 512)
        height = spec.get("height", 512)
        steps = spec.get("steps", 4)
        remove_bg = spec.get("remove_bg", True)
        pixel_size = spec.get("pixel_size", 4)

        full_prompt = _build_prompt(prompt)

        if not os.path.isabs(output_path):
            output_path = os.path.join(OUTPUT_DIR, output_path)

        _ensure_dir(output_path)

        t0 = time.time()
        image = _generate(pipe, full_prompt, seed, width, height, steps)

        if pixel_size > 0:
            image = _pixelate(image, pixel_size)

        if remove_bg:
            image = _remove_background(image)

        image.save(output_path)
        elapsed = time.time() - t0

        results.append(
            {
                "output_path": output_path,
                "seed_used": seed,
                "generation_time": f"{elapsed:.1f}s",
                "prompt": full_prompt,
                "size": f"{width}x{height}",
                "transparent": remove_bg,
            }
        )

    return results


if __name__ == "__main__":
    mcp.run(transport="stdio")
