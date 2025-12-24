import os
import io
import zipfile
import time
import random
from typing import Dict, Any, Optional, List
import requests
from PIL import Image, ImageFilter

class InpaintAdapter:
    """
    A client for FLUX-Controlnet-Inpainting api_server.py.
    Expected: POST /inpaint with image + mask + prompt
    """
    def __init__(self, base_url: Optional[str] = None, timeout: int = 120, dry_run: bool = False):
        # Default ports as requested: inpainting service at 8041
        self.base_url = base_url or os.getenv("INPAINT_API_BASE", "http://127.0.0.1:8041")
        self.timeout = timeout
        self.dry_run = dry_run
        # Allow overriding endpoint path
        self.inpaint_path = os.getenv("INPAINT_API_PATH", "/inpaint")
        self._fallback_paths = [
            "/inpaint",
            "/api/inpaint",
            "/edit",
            "/run",
            "/predict",
            "/flux/inpaint",
        ]

    def _parse_response_images(self, resp: requests.Response, expect_multiple: bool = False) -> List[Image.Image]:
        ctype = (resp.headers.get("content-type") or "").lower()
        raw = resp.content
        # ZIP path
        if "application/zip" in ctype or (expect_multiple and raw[:2] == b"PK"):
            imgs: List[Image.Image] = []
            try:
                with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                    names = [n for n in zf.namelist() if not n.endswith("/")]
                    # try stable ordering
                    names.sort()
                    for n in names:
                        try:
                            with zf.open(n) as f:
                                data = f.read()
                            im = Image.open(io.BytesIO(data)).convert("RGB")
                            imgs.append(im)
                        except Exception:
                            continue
            except Exception as e:
                # Fall back to single image decode if zip parsing fails
                try:
                    im = Image.open(io.BytesIO(raw)).convert("RGB")
                    return [im]
                except Exception:
                    raise e
            return imgs
        # Single image path
        im = Image.open(io.BytesIO(raw)).convert("RGB")
        return [im]

    def inpaint_many(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        num_images: int = 1,
        extra: Optional[Dict[str, Any]] = None,
    ) -> List[Image.Image]:
        if self.dry_run:
            # Produce simple blurred variants according to num_images
            num = max(1, int(num_images or 1))
            imgs: List[Image.Image] = []
            for i in range(num):
                img = image.convert("RGBA").copy()
                # vary blur radius slightly by index
                radius = 1 + (i % 3)
                blurred = image.convert("RGBA").filter(ImageFilter.GaussianBlur(radius=radius))
                mask_rgba = mask.convert("L")
                img.paste(blurred, (0, 0), mask_rgba)
                imgs.append(img.convert("RGB"))
            return imgs

        # Prepare immutable bytes and rebuild file objects on each retry to avoid pointer exhaustion
        img_buf = io.BytesIO()
        image.save(img_buf, format="PNG")
        img_bytes = img_buf.getvalue()
        mask_buf = io.BytesIO()
        mask.save(mask_buf, format="PNG")
        mask_bytes = mask_buf.getvalue()

        # include both 'prompt' and 'text' to match various servers
        data = {"prompt": prompt, "text": prompt}
        try:
            n = int(num_images or 1)
        except Exception:
            n = 1
        if n < 1:
            n = 1
        data_bulk = dict(data)
        if extra:
            data_bulk.update(extra)
        data_bulk["num_images"] = str(n)

        # Single bulk request to configured path only (no noisy fallbacks)
        url = f"{self.base_url}{self.inpaint_path}"
        files = {
            "image": ("image.png", io.BytesIO(img_bytes), "image/png"),
            "mask": ("mask.png", io.BytesIO(mask_bytes), "image/png"),
        }
        r = requests.post(url, data=data_bulk, files=files, timeout=self.timeout)
        if r.status_code >= 400:
            detail = r.text.strip()[:512]
            raise RuntimeError(f"Inpaint server error {r.status_code} at {url}: {detail}")
        imgs = self._parse_response_images(r, expect_multiple=(n > 1))
        return imgs[:n] if imgs else []

    def inpaint(self, image: Image.Image, mask: Image.Image, prompt: str, extra: Optional[Dict[str, Any]] = None) -> Image.Image:
        # Backward-compatible single image API (no fallbacks, configured path only)
        img_buf = io.BytesIO()
        image.save(img_buf, format="PNG")
        img_bytes = img_buf.getvalue()
        mask_buf = io.BytesIO()
        mask.save(mask_buf, format="PNG")
        mask_bytes = mask_buf.getvalue()
        data = {"prompt": prompt, "text": prompt, "num_images": "1"}
        if extra:
            data.update(extra)
        url = f"{self.base_url}{self.inpaint_path}"
        files = {
            "image": ("image.png", io.BytesIO(img_bytes), "image/png"),
            "mask": ("mask.png", io.BytesIO(mask_bytes), "image/png"),
        }
        r = requests.post(url, data=data, files=files, timeout=self.timeout)
        if r.status_code >= 400:
            detail = r.text.strip()[:512]
            raise RuntimeError(f"Inpaint server error {r.status_code} at {url}: {detail}")
        out = Image.open(io.BytesIO(r.content)).convert("RGB")
        return out
