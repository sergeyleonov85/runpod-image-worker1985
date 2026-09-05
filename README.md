# runpod-image-worker1985

Minimal RunPod Serverless text-to-image worker using SDXL-Turbo.

## API input

```json
{
  "input": {
    "prompt": "your prompt",
    "width": 512,
    "height": 512,
    "steps": 2,
    "seed": 1985
  }
}
```

- `prompt`: required
- `width`, `height`: 256-1024, divisible by 64
- `steps`: 1-4
- `seed`: optional

The worker returns a PNG as a `data:image/png;base64,...` string.

## RunPod deployment

Create a Serverless endpoint from this GitHub repository and use:

- Branch: `main`
- Dockerfile path: `/Dockerfile`
- Endpoint type: `Queue`
- Active workers: `0`
- Max workers: `1`
- GPU: 24 GB or larger recommended

The model is loaded once when the worker starts, outside the handler, to avoid reloading it for every job.
