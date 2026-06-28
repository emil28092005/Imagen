# Imagen

MCP server for generating pixel-art sprites with transparent backgrounds. Bring your own model — any [diffusers](https://github.com/huggingface/diffusers)-compatible text-to-image pipeline works.

> The default configuration uses [FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) + [pixel-art-lora](https://huggingface.co/Limbicnation/pixel-art-lora), but you can swap in any model you like.

## Features

- **Bring your own model** — any diffusers-compatible pipeline (FLUX, SDXL, SD3, etc.)
- **Text-to-sprite generation** — describe any character, get a pixel-art PNG
- **Transparent background** — automatic background removal via flood-fill
- **Pixel-art effect** — downscale/upscale with NEAREST interpolation
- **Reproducible** — optional seed for consistent results
- **Batch generation** — generate multiple sprites in one call
- **MCP integration** — works with any MCP-compatible client (opencode, Claude, etc.)
- **Feedback loop** — rate generated sprites, AI uses high-rated ones as reference

## Quick Start

### 1. Install dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Download a model

You need a base text-to-image model. Optionally, a LoRA adapter for pixel-art style.

**Example: FLUX.2-klein-4B + pixel-art-lora (default)**

```bash
mkdir -p ~/models

# Base model (~23 GB)
huggingface-cli download black-forest-labs/FLUX.2-klein-4b \
    --local-dir ~/models/flux2-klein-4b

# LoRA adapter (~625 MB) — optional but recommended for pixel-art
huggingface-cli download Limbicnation/pixel-art-lora \
    --local-dir ~/models/pixel-art-lora
```

**Other models that work:**

| Model | Size | LoRA support | Notes |
|---|---|---|---|
| [FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4b) | ~23 GB | Yes | Default, distilled (4 steps) |
| [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) | ~23 GB | Yes | More detail, slower (20+ steps) |
| [SDXL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) | ~7 GB | Yes | Lighter, good for 8 GB VRAM |
| [SD3.5-large](https://huggingface.co/stabilityai/stable-diffusion-3.5-large) | ~16 GB | Yes | Good quality/speed balance |

> **Note:** You may need to adjust `LORA_SCALE` and pipeline class in `server.py` depending on your model. See [Configuration](#configuration).

### 3. Configure paths

By default, models are expected at `~/models/`. Override with environment variables:

```bash
export IMAGEGEN_MODEL_DIR=/path/to/your/base-model
export IMAGEGEN_LORA_DIR=/path/to/your/lora        # optional, set empty to disable
export IMAGEGEN_OUTPUT_DIR=/path/to/output
```

### 4. Run as MCP server

```bash
./venv/bin/python server.py
```

Or configure in your MCP client:

```json
{
  "mcp": {
    "pixel-art": {
      "type": "local",
      "command": ["./venv/bin/python", "server.py"],
      "enabled": true
    }
  }
}
```

## Tools

### Generation

#### `generate_sprite`

Generate a single pixel-art sprite. Automatically saved to feedback DB (unrated).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `prompt` | str | required | Sprite description (e.g. "a brave knight in armor") |
| `output_path` | str | required | PNG save path (relative to output dir or absolute) |
| `seed` | int? | null | Seed for reproducibility |
| `width` | int | 512 | Image width |
| `height` | int | 512 | Image height |
| `steps` | int | 4 | Inference steps (lower = faster, less detail) |
| `remove_bg` | bool | true | Remove background, make transparent |
| `pixel_size` | int | 4 | Pixel block size (0 = off, 4 = chunky pixel-art) |

Returns: `output_path`, `db_id`, `generation_time`, and other metadata.

#### `batch_generate`

Generate multiple sprites in one call. Each is saved to the feedback DB.

### Feedback

#### `rate_sprite`

Rate a generated sprite 1-5 stars with optional feedback.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `db_id` | str | required | ID returned by generate_sprite / batch_generate |
| `rating` | int | required | 1-5 stars |
| `feedback` | str? | null | Optional text feedback |

#### `get_reference_sprites`

Get highly-rated reference sprites for a prompt. The AI uses these as examples when generating similar sprites.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `prompt` | str | required | Search query (e.g. "knight") |
| `limit` | int | 5 | Max results |
| `min_rating` | int | 4 | Minimum rating threshold |

#### `list_sprites`

List sprites in the feedback database.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `filter` | str | "all" | "all", "unrated", or "top" |
| `limit` | int | 20 | Max results |

#### `db_stats`

Get database statistics: total sprites, rated, unrated, average rating.

## Configuration

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `IMAGEGEN_MODEL_DIR` | `~/models/flux2-klein-4b` | Path to base model |
| `IMAGEGEN_LORA_DIR` | `~/models/pixel-art-lora` | Path to LoRA adapter |
| `IMAGEGEN_OUTPUT_DIR` | `./output` | Default output directory |

### Swapping models

The server is configured for FLUX.2-klein by default. To use a different model, edit `server.py`:

1. **Pipeline class** — replace `Flux2KleinPipeline` with your model's pipeline (e.g. `StableDiffusionXLPipeline` for SDXL)
2. **LoRA scale** — adjust `LORA_SCALE` (rsLoRA needs ~0.1, regular LoRA typically 0.7-1.0)
3. **Guidance scale** — distilled models ignore it; standard models need 5-8
4. **Steps** — distilled models work at 4; standard models need 20-30

## How It Works

1. **Generation** — text-to-image model generates a 512x512 image
2. **Pixelation** — downscale with LANCZOS, upscale with NEAREST → chunky pixel-art blocks
3. **Background removal** — detect border color, normalize to solid fill, flood-fill from edges → transparent PNG

## Requirements

- **GPU:** NVIDIA with >= 8 GB VRAM (uses CPU offload automatically)
- **Python:** 3.12+
- **CUDA:** 12.0+

## Credits

- Default model: [FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) by Black Forest Labs (Apache 2.0)
- Default LoRA: [pixel-art-lora](https://huggingface.co/Limbicnation/pixel-art-lora) by Limbicnation (Apache 2.0)
- MCP SDK: [modelcontextprotocol/python-sdk](https://github.com/modelcontextprotocol/python-sdk)

## License

MIT — see [LICENSE](LICENSE)

Model licenses are separate from this project. Check each model's license card for usage terms.
