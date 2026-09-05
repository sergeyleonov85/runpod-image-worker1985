import os
import uuid
import boto3
import torch
import runpod
from diffusers import StableDiffusionXLPipeline

MODEL_ID = os.getenv("MODEL_ID", "stabilityai/stable-diffusion-xl-base-1.0")

print(f"Loading model: {MODEL_ID}")
pipe = StableDiffusionXLPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    use_safetensors=True,
)
pipe = pipe.to("cuda")
pipe.enable_attention_slicing()
print("Model loaded.")

S3_ENDPOINT = os.getenv("S3_ENDPOINT")
S3_BUCKET = os.getenv("S3_BUCKET")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY")

def upload_image(path, key):
    if not all([S3_ENDPOINT, S3_BUCKET, S3_ACCESS_KEY, S3_SECRET_KEY]):
        raise RuntimeError(
            "S3 is not configured. Set S3_ENDPOINT, S3_BUCKET, "
            "S3_ACCESS_KEY and S3_SECRET_KEY as RunPod secrets/environment variables."
        )

    s3 = boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
        region_name="us-ca-2",
    )
    s3.upload_file(path, S3_BUCKET, key, ExtraArgs={"ContentType": "image/png"})

    # RunPod S3-compatible storage may not expose objects publicly.
    # Return the object coordinates so the next integration layer can fetch/sign it.
    return {
        "bucket": S3_BUCKET,
        "key": key,
        "endpoint": S3_ENDPOINT,
    }

def handler(event):
    inp = event.get("input", {})
    prompt = inp.get("prompt")
    if not prompt:
        return {"error": "input.prompt is required"}

    width = int(inp.get("width", 512))
    height = int(inp.get("height", 512))
    steps = int(inp.get("steps", 20))
    seed = int(inp.get("seed", 1985))
    guidance = float(inp.get("guidance_scale", 7.0))

    generator = torch.Generator(device="cuda").manual_seed(seed)

    image = pipe(
        prompt=prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
    ).images[0]

    filename = f"{uuid.uuid4().hex}.png"
    local_path = f"/tmp/{filename}"
    image.save(local_path, format="PNG")

    try:
        obj = upload_image(local_path, f"generated/{filename}")
    finally:
        try:
            os.remove(local_path)
        except OSError:
            pass

    return {
        "status": "ok",
        "image": obj,
        "seed": seed,
        "width": width,
        "height": height,
        "steps": steps,
        "model": MODEL_ID,
    }

runpod.serverless.start({"handler": handler})
