"""
Common tools for coding.
"""
# Image related
import io
import base64
import pathlib

import numpy as np
import requests
from PIL import Image


def _to_pil(image: object) -> Image.Image:
    """Convert various input types to a PIL Image."""
    if isinstance(image, Image.Image):
        return image

    if isinstance(image, np.ndarray):
        return Image.fromarray(image)

    if isinstance(image, bytes):
        return Image.open(io.BytesIO(image))

    if isinstance(image, pathlib.Path):
        return Image.open(image)

    if isinstance(image, str):
        if image.startswith(("http://", "https://")):
            resp = requests.get(image, timeout=10)
            resp.raise_for_status()
            return Image.open(io.BytesIO(resp.content))

        if image.startswith("data:image/"):
            # data URI: data:image/png;base64,<data>
            _, data = image.split(",", 1)
            return Image.open(io.BytesIO(base64.b64decode(data)))

        # Try as a file path first
        p = pathlib.Path(image)
        if p.exists():
            return Image.open(p)

        # Last resort: raw base64 string
        try:
            return Image.open(io.BytesIO(base64.b64decode(image)))
        except Exception:
            pass

        raise ValueError(f"Cannot interpret string as image: {image[:80]!r}")

    raise TypeError(f"Unsupported image type: {type(image)}")


def parse_image(image: object, fmt: str | None = None) -> dict:
    """Parse any image input into a unified dict with pil, base64, base64_url, and bytes.

    Args:
        image: PIL Image, numpy array, bytes, URL string, file path, or base64 string.
        fmt:   Output encoding format (e.g. "PNG", "JPEG"). Inferred from img.format
               when available, defaulting to "PNG". JPEG is auto-upgraded to PNG for
               images with an alpha channel.

    Returns:
        {
            "pil":        PIL.Image.Image,
            "bytes":      raw image bytes in the resolved format,
            "base64":     plain base64 string,
            "base64_url": data URI  "data:image/png;base64,..."
        }
    """
    pil_img = _to_pil(image)

    # Resolve format: explicit arg → img.format → PNG
    resolved_fmt = (fmt or pil_img.format or "PNG").upper()

    # JPEG cannot encode RGBA/P-with-transparency → fall back to PNG
    if resolved_fmt == "JPEG" and pil_img.mode in ("RGBA", "LA", "PA"):
        resolved_fmt = "PNG"

    buf = io.BytesIO()
    pil_img.save(buf, format=resolved_fmt)
    raw_bytes = buf.getvalue()

    b64_str = base64.b64encode(raw_bytes).decode("ascii")
    mime = f"image/{resolved_fmt.lower()}"

    return {
        "format": resolved_fmt,
        "pil": pil_img,
        "bytes": raw_bytes,
        "base64": b64_str,
        "base64_url": f"data:{mime};base64,{b64_str}",
    }
