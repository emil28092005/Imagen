#!/usr/bin/env python3
"""
MCP server for generating pixel-art sprites using SDXL + LCM + pixel-art-xl LoRA.

Tools:
  - generate_sprite:       Generate a single pixel-art sprite
  - batch_generate:        Generate multiple sprites in one call
  - rate_sprite:           Rate a generated sprite (1-5 stars) with optional feedback
  - get_reference_sprites: Get highly-rated reference sprites for a prompt
  - list_sprites:          List sprites in the feedback DB (all, unrated, or top-rated)
  - db_stats:              Get feedback database statistics

Model is loaded lazily on first call (~12s), then stays in VRAM for speed.
Background is removed post-generation to produce transparent PNG.
Every generated sprite is automatically saved to the feedback DB (unrated).
"""

import os
import sys
import time
from typing import Optional

import numpy as np
from PIL import Image
from mcp.server.fastmcp import FastMCP

from feedback import FeedbackDB

# Paths — models live in a shared location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.environ.get(
    "IMAGEGEN_MODEL_DIR",
    os.path.join(os.path.expanduser("~"), "models", "sdxl-base"),
)
LORA_DIR = os.environ.get(
    "IMAGEGEN_LORA_DIR",
    os.path.join(os.path.expanduser("~"), "models", "pixel-art-xl"),
)
LCM_LORA_DIR = os.environ.get(
    "IMAGEGEN_LCM_LORA_DIR",
    os.path.join(os.path.expanduser("~"), "models", "lcm-lora-sdxl"),
)
OUTPUT_DIR = os.environ.get("IMAGEGEN_OUTPUT_DIR", os.path.join(BASE_DIR, "output"))
DB_PATH = os.environ.get("IMAGEGEN_DB_PATH", os.path.join(BASE_DIR, "feedback.db"))

# LoRA scales — pixel-art-xl needs 1.2, LCM needs 1.0
PIXEL_LORA_SCALE = 1.2
LCM_LORA_SCALE = 1.0

# Negative prompt to improve quality
NEGATIVE_PROMPT = "3d render, realistic, detailed, noise, artifacts, blurry, text"

# Global state — model loaded lazily
_pipe = None
_device = None
_db: Optional[FeedbackDB] = None


def _get_db() -> FeedbackDB:
    global _db
    if _db is None:
        _db = FeedbackDB.open(DB_PATH)
        sys.stderr.write(f"[pixel-art] Feedback DB: {DB_PATH}\n")
    return _db


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

    sys.stderr.write(
        "[pixel-art] Loading SDXL + LCM + pixel-art-xl LoRA (first call)...\n"
    )
    t0 = time.time()

    import torch
    from diffusers import StableDiffusionXLPipeline, LCMScheduler

    _pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )
    _pipe.scheduler = LCMScheduler.from_config(_pipe.scheduler.config)

    _pipe.load_lora_weights(LCM_LORA_DIR, adapter_name="lcm")
    _pipe.load_lora_weights(LORA_DIR, adapter_name="pixel")
    _pipe.set_adapters(
        ["lcm", "pixel"], adapter_weights=[LCM_LORA_SCALE, PIXEL_LORA_SCALE]
    )

    if _get_device() == "cuda":
        _pipe.to("cuda")
    else:
        _pipe.to(_get_device())

    elapsed = time.time() - t0
    sys.stderr.write(f"[pixel-art] Model loaded in {elapsed:.1f}s\n")
    return _pipe


def _build_prompt(user_prompt: str) -> str:
    return f"pixel art, {user_prompt}, simple, flat colors, game asset"


def _generate(
    pipe, prompt: str, seed: Optional[int], width: int, height: int, steps: int
):
    import torch

    generator = None
    if seed is not None:
        generator = torch.Generator(device="cuda").manual_seed(seed)

    image = pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=steps,
        guidance_scale=1.5,
        height=height,
        width=width,
        generator=generator,
    ).images[0]

    return image


def _remove_background(image: Image.Image, threshold: int = 30) -> Image.Image:
    """Remove background using flood-fill from edges.

    Detects the border color, then flood-fills from all edge pixels,
    removing any pixel that is within threshold of the border color AND
    connected to the border. Interior pixels with similar colors (e.g.
    highlights on armor) are preserved because they are not connected
    to the border through similar-colored pixels.
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

    alpha = np.full((h, w), 255, dtype=np.uint8)
    visited = np.zeros((h, w), dtype=bool)
    queue = deque()

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

    # BFS flood-fill: remove pixels close to bg_color that are connected to border
    while queue:
        y, x = queue.popleft()
        dist = np.abs(arr[y, x] - bg_color).sum()
        if dist > threshold * 3:
            continue
        alpha[y, x] = 0

        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                visited[ny, nx] = True
                queue.append((ny, nx))

    rgba = np.dstack([arr.astype(np.uint8), alpha])
    return Image.fromarray(rgba, mode="RGBA")


def _pixelate(image: Image.Image, pixel_size: int = 8) -> Image.Image:
    """Downscale then upscale with NEAREST to create chunky pixel-art effect.

    pixel_size=8 means each "pixel" in the result is an 8x8 block.
    pixel_size=0 returns the original image unchanged.
    """
    if pixel_size <= 0:
        return image
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
    steps: int = 8,
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

    db = _get_db()
    entry_id = db.add(
        prompt=prompt,
        params={
            "seed": seed,
            "width": width,
            "height": height,
            "steps": steps,
            "remove_bg": remove_bg,
            "pixel_size": pixel_size,
        },
        image_path=output_path,
    )

    return {
        "output_path": output_path,
        "seed_used": seed,
        "generation_time": f"{elapsed:.1f}s",
        "prompt": full_prompt,
        "size": f"{width}x{height}",
        "transparent": remove_bg,
        "pixel_size": pixel_size,
        "db_id": entry_id,
        "rated": False,
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
          - steps:       int   (optional, default 8)
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
        steps = spec.get("steps", 8)
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

        db = _get_db()
        entry_id = db.add(
            prompt=prompt,
            params={
                "seed": seed,
                "width": width,
                "height": height,
                "steps": steps,
                "remove_bg": remove_bg,
                "pixel_size": pixel_size,
            },
            image_path=output_path,
        )

        results.append(
            {
                "output_path": output_path,
                "seed_used": seed,
                "generation_time": f"{elapsed:.1f}s",
                "prompt": full_prompt,
                "size": f"{width}x{height}",
                "transparent": remove_bg,
                "db_id": entry_id,
                "rated": False,
            }
        )

    return results


@mcp.tool()
def rate_sprite(
    db_id: str,
    rating: int,
    feedback: Optional[str] = None,
) -> dict:
    """Rate a generated sprite (1-5 stars) with optional feedback text.

    Use this after reviewing a sprite to teach the system what looks good.
    The AI uses high-rated sprites as reference when generating similar ones.

    Args:
        db_id:     The ID returned by generate_sprite or batch_generate
        rating:    1-5 stars (5 = excellent, 1 = terrible)
        feedback:  Optional text feedback (e.g. "great colors, bad proportions")

    Returns:
        Dict with db_id, rating, feedback, and status.
    """
    db = _get_db()
    db.update_rating(db_id, rating, feedback)

    return {
        "db_id": db_id,
        "rating": rating,
        "feedback": feedback,
        "status": "saved",
    }


@mcp.tool()
def get_reference_sprites(
    prompt: str,
    limit: int = 5,
    min_rating: int = 4,
) -> list[dict]:
    """Get highly-rated reference sprites from the feedback DB for a given prompt.

    Use these as examples when generating similar sprites to improve quality.
    Returns sprites with similar prompt keywords that have been rated >= min_rating.

    Args:
        prompt:      The prompt to search for (e.g. "knight", "crystal warrior")
        limit:       Max number of results (default 5)
        min_rating:  Minimum rating (1-5, default 4)

    Returns:
        List of dicts with db_id, prompt, rating, feedback, image_path, params.
    """
    db = _get_db()
    entries = db.search_similar(prompt, limit * 2)
    entries = [e for e in entries if e.rating >= min_rating][:limit]

    if not entries:
        return []

    return [
        {
            "db_id": e.id,
            "prompt": e.prompt,
            "rating": e.rating,
            "feedback": e.feedback,
            "image_path": e.image_path,
            "params": e.params,
        }
        for e in entries
    ]


@mcp.tool()
def list_sprites(
    filter: str = "all",
    limit: int = 20,
) -> list[dict]:
    """List sprites in the feedback database.

    Args:
        filter:  "all" = all sprites, "unrated" = only unrated, "top" = highest rated
        limit:   Max number of results (default 20)

    Returns:
        List of dicts with db_id, prompt, rating, image_path, created_at.
    """
    db = _get_db()

    if filter == "unrated":
        entries = db.get_unrated()
    elif filter == "top":
        entries = db.top_rated(limit, 1)
    else:
        entries = db.get_all()

    entries = entries[:limit]

    return [
        {
            "db_id": e.id,
            "prompt": e.prompt,
            "rating": e.rating,
            "image_path": e.image_path,
            "created_at": e.created_at,
        }
        for e in entries
    ]


@mcp.tool()
def db_stats() -> dict:
    """Get feedback database statistics.

    Returns:
        Dict with total, rated, unrated, avg_rating.
    """
    db = _get_db()
    stats = db.stats()

    return {
        "total": stats.total,
        "rated": stats.rated,
        "unrated": stats.unrated,
        "avg_rating": stats.avg_rating,
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
