# Imagegen

MCP server for generating pixel-art sprites using FLUX.2-klein-4B + [pixel-art-lora](https://huggingface.co/Limbicnation/pixel-art-lora).

## Features

- **Text-to-sprite generation** — describe any character, get a pixel-art PNG
- **Transparent background** — automatic background removal via flood-fill
- **Pixel-art effect** — downscale/upscale with NEAREST interpolation
- **Reproducible** — optional seed for consistent results
- **Batch generation** — generate multiple sprites in one call
- **MCP integration** — works with any MCP-compatible client (opencode, Claude, etc.)

## Quick Start

### 1. Install dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Download models

```bash
# Create models directory
mkdir -p ~/models

# Download base model (~23 GB)
huggingface-cli download black-forest-labs/FLUX.2-klein-4b \
    --local-dir ~/models/flux2-klein-4b

# Download LoRA adapter (~625 MB)
huggingface-cli download Limbicnation/pixel-art-lora \
    --local-dir ~/models/pixel-art-lora
```

### 3. Configure paths (optional)

By default, models are expected at `~/models/`. Override with environment variables:

```bash
export IMAGEGEN_MODEL_DIR=/path/to/flux2-klein-4b
export IMAGEGEN_LORA_DIR=/path/to/pixel-art-lora
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

### `generate_sprite`

Generate a single pixel-art sprite.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `prompt` | str | required | Sprite description (e.g. "a brave knight in armor") |
| `output_path` | str | required | PNG save path (relative to output dir or absolute) |
| `seed` | int? | null | Seed for reproducibility |
| `width` | int | 512 | Image width |
| `height` | int | 512 | Image height |
| `steps` | int | 4 | Inference steps (FLUX.2-klein is distilled) |
| `remove_bg` | bool | true | Remove background, make transparent |
| `pixel_size` | int | 4 | Pixel block size (0 = off, 4 = chunky pixel-art) |

### `batch_generate`

Generate multiple sprites in one call. Accepts a list of specs with the same parameters.

## How It Works

1. **Generation** — FLUX.2-klein-4B (4B params, distilled to 4 steps) with pixel-art LoRA (scale 0.1 for rsLoRA compatibility)
2. **Pixelation** — downscale with LANCZOS, upscale with NEAREST → chunky pixel-art blocks
3. **Background removal** — detect border color, normalize to magenta fill, flood-fill from edges → transparent PNG

## Requirements

- **GPU:** NVIDIA with >= 8 GB VRAM (uses CPU offload)
- **Python:** 3.12+
- **CUDA:** 12.0+

## Credits

- Base model: [FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) by Black Forest Labs (Apache 2.0)
- LoRA: [pixel-art-lora](https://huggingface.co/Limbicnation/pixel-art-lora) by Limbicnation (Apache 2.0)
- MCP SDK: [modelcontextprotocol/python-sdk](https://github.com/modelcontextprotocol/python-sdk)

## License

MIT — see [LICENSE](LICENSE)

Model licenses are separate (Apache 2.0). Check model cards for details.
