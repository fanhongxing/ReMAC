import os
import io
import asyncio
from typing import Optional

# Ensure HF mirror endpoint for faster model access if available
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from PIL import Image
from diffusers.utils import check_min_version
import zipfile
import time

# Local modules
from controlnet_flux import FluxControlNetModel
from transformer_flux import FluxTransformer2DModel
from pipeline_flux_controlnet_inpaint import FluxControlNetInpaintingPipeline

# Validate diffusers version
check_min_version("0.30.2")

app = FastAPI(title="FLUX ControlNet Inpainting API")

# Global state for model/pipeline
_device = "cuda:0" if torch.cuda.is_available() else "cpu"
_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
_pipe: Optional[FluxControlNetInpaintingPipeline] = None
_pipe_lock = asyncio.Lock()

# Paths and model IDs (adapt if your paths differ)
TRANSFORMER_REPO_PATH = "/data/fanhongxing/ckpt/FLUX.1-dev"
CONTROLNET_MODEL_ID = "alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta"


def _require_cuda():
    if not torch.cuda.is_available():
        raise HTTPException(status_code=503, detail="CUDA GPU is required for this pipeline but not available.")


def _load_pipeline_if_needed():
    global _pipe
    if _pipe is not None:
        return _pipe

    _require_cuda()

    controlnet = FluxControlNetModel.from_pretrained(CONTROLNET_MODEL_ID, torch_dtype=_dtype)
    transformer = FluxTransformer2DModel.from_pretrained(
        TRANSFORMER_REPO_PATH, subfolder="transformer", torch_dtype=_dtype
    )

    pipe = FluxControlNetInpaintingPipeline.from_pretrained(
        TRANSFORMER_REPO_PATH,
        controlnet=controlnet,
        transformer=transformer,
        torch_dtype=_dtype,
    ).to(_device)

    # Ensure modules use the expected dtype
    pipe.transformer.to(_dtype)
    pipe.controlnet.to(_dtype)

    _pipe = pipe
    return _pipe


def _read_image(file: UploadFile) -> Image.Image:
    try:
        data = file.file.read()
        img = Image.open(io.BytesIO(data))
        return img.convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file '{file.filename}': {e}")


@app.get("/health")
async def health():
    # Lightweight health without model load
    return JSONResponse({"status": "ok", "device": _device})


@app.post("/inpaint")
async def inpaint(
    prompt: str = Form(..., description="Text prompt to guide inpainting"),
    image: UploadFile = File(..., description="RGB input image (PNG/JPG)", alias="image"),
    mask: UploadFile = File(..., description="Mask image (white = keep/visible, black = inpaint)", alias="mask"),
    steps: int = Form(28, description="Number of inference steps"),
    guidance_scale: float = Form(3.5, description="Classifier-free guidance scale"),
    controlnet_conditioning_scale: float = Form(0.9, description="ControlNet conditioning scale"),
    true_guidance_scale: float = Form(1.0, description="True guidance scale for beta pipeline"),
    num_images: int = Form(1, description="Number of images to generate (1-8)"),
    seed: Optional[int] = Form(None, description="Random seed; fixed for reproducibility if provided"),
):
    # Parse images
    base_img = _read_image(image)
    mask_img = _read_image(mask)

    # Keep original size to resize result back
    w, h = base_img.size

    # Resize both inputs to 1024x1024 as in the reference script
    target_size = (1024, 1024)
    base_resized = base_img.resize(target_size)
    mask_resized = mask_img.resize(target_size)

    # Validate and prepare generators
    if num_images < 1 or num_images > 8:
        raise HTTPException(status_code=400, detail="num_images must be between 1 and 8")

    generator = None
    if seed is not None:
        _require_cuda()
        # Ensure different streams when requesting multiple images with a fixed seed
        if num_images == 1:
            generator = torch.Generator(device=_device).manual_seed(int(seed))
        else:
            base_seed = int(seed)
            generator = [
                torch.Generator(device=_device).manual_seed(base_seed + i)
                for i in range(num_images)
            ]

    # Run pipeline under a lock to avoid GPU memory thrash on concurrent requests
    async with _pipe_lock:
        pipe = _load_pipeline_if_needed()
        try:
            out = pipe(
                prompt=prompt,
                height=target_size[1],
                width=target_size[0],
                control_image=base_resized,
                control_mask=mask_resized,
                num_inference_steps=int(steps),
                generator=generator,
                controlnet_conditioning_scale=float(controlnet_conditioning_scale),
                guidance_scale=float(guidance_scale),
                negative_prompt="",
                true_guidance_scale=float(true_guidance_scale),
                num_images_per_prompt=int(num_images),
            )
            images = out.images
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Pipeline inference failed: {e}")

    # Resize images back to original resolution
    images = [img.resize((w, h)) for img in images]

    # If only one image requested, return single PNG for backward compatibility
    if num_images == 1:
        buf = io.BytesIO()
        images[0].save(buf, format="PNG")
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")

    # Otherwise, package multiple images into a ZIP archive
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        timestamp = int(time.time())
        for i, img in enumerate(images, start=1):
            img_buf = io.BytesIO()
            img.save(img_buf, format="PNG")
            img_buf.seek(0)
            filename = f"inpaint_{timestamp}_{i:02d}.png"
            zf.writestr(filename, img_buf.read())
    zip_buf.seek(0)

    headers = {"Content-Disposition": "attachment; filename=results.zip"}
    return StreamingResponse(zip_buf, media_type="application/zip", headers=headers)


if __name__ == "__main__":
    # Optional: run with `python api_server.py` for local testing
    import uvicorn

    # Do not eagerly load models on startup unless desired; they will load on first request
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8041,
        reload=False,
        workers=1,
    )
