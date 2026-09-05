import base64
import io
import os
import random
import time

import runpod
import torch
from diffusers import AutoPipelineForText2Image

MODEL_ID = os.getenv("MODEL_ID", "stabilityai/sdxl-turbo")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is required for this worker.")

print(f"Loading model: {MODEL_ID}")

pipe = AutoPipelineForText2Image.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
)
pipe = pipe.to("cuda")
pipe.set_progress_bar_config(disable=True)

print("Model loaded successfully.")


def _validate_dimension(value, name):
    value = int(value)
    if value < 256 or value > 1024 or value % 64 != 0:
        raise ValueError(f"{name} must be between 256 and 1024 and divisible by 64.")
    return value


def handler(event):
    inp = event.get("input", {})
    prompt = str(inp.get("prompt", "")).strip()

    if not prompt:
        return {"error": "input.prompt is required"}

    try:
        width = _validate_dimension(inp.get("width", 512), "width")
        height = _validate_dimension(inp.get("height", 512), "height")

        steps = int(inp.get("steps", 2))
        if steps < 1 or steps > 4:
            raise ValueError("steps must be between 1 and 4 for SDXL-Turbo.")

        seed = int(inp.get("seed", random.randint(0, 2**31 - 1)))
        if seed < 0:
            raise ValueError("seed must be >= 0")

        generator = torch.Generator(device="cuda").manual_seed(seed)

        started = time.perf_counter()

        with torch.inference_mode():
            image = pipe(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=0.0,
                generator=generator,
            ).images[0]

        elapsed = time.perf_counter() - started

        buf = io.BytesIO()
        image.save(buf, format="PNG", optimize=True)
        image_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        return {
            "image": f"data:image/png;base64,{image_b64}",
            "seed": seed,
            "width": width,
            "height": height,
            "steps": steps,
            "model": MODEL_ID,
            "generation_time_seconds": round(elapsed, 3),
        }

    except Exception as exc:
        return {
            "error": str(exc),
            "type": exc.__class__.__name__,
        }


runpod.serverless.start({"handler": handler})
