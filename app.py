import csv
import json
import os
import re
import statistics
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Queue, Empty
from typing import Dict, Any, Optional, List, Tuple

import requests
from dotenv import load_dotenv
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

# --- Leonardo REST base ---
BASE_URL = "https://cloud.leonardo.ai/api/rest/v1"  # official base

# Load .env early from the same folder as this script (works regardless of CWD)
ENV_PATH = Path(__file__).with_name(".env")
load_dotenv(dotenv_path=ENV_PATH, override=False)

# Model IDs (hardcoded for runtime switching)
MODEL_CHEAP = "b24e16ff-06e3-43eb-8d33-4416c2d75876"  # Lightning XL
MODEL_HQ = "5c232a9e-9061-4777-980a-ddc8e65647c6"  # Vision XL

def get_model_id() -> str:
    """
    Read modelId from env dynamically (prevents 'env loaded too late' issues).
    Default uses a low-cost SDXL-capable model suitable for init_image_id workflows.
    """
    mid = (os.getenv("LEONARDO_MODEL_ID") or "").strip()
    return mid or "b24e16ff-06e3-43eb-8d33-4416c2d75876"  # Lightning XL (cheap)

def get_hq_model_id() -> str:
    mid = (os.getenv("LEONARDO_HQ_MODEL_ID") or "").strip()
    return mid or "5c232a9e-9061-4777-980a-ddc8e65647c6"  # Vision XL

def get_i2i_fallback_model_id() -> str:
    mid = (os.getenv("LEONARDO_I2I_FALLBACK_MODEL_ID") or "").strip()
    return mid or get_model_id()

I2I_UNSUPPORTED_MODEL_IDS = {
    "05ce0082-2d80-4a2d-8653-4d1c85e2418e": "Lucid Realism",
    "7b592283-e8a7-4c5a-9ba6-d18c31f258b9": "Lucid Origin",
}

ALLOWED_EXTS = [".jpg", ".jpeg", ".png", ".webp"]

DEFAULT_PACK_PROMPT = (
    "photorealistic studio packshot, same packaging design and same text as reference, "
    "centered, pure white seamless background, soft natural shadow, sharp focus, "
    "realistic lighting, slight natural wrinkles/reflections, no redesign, no new labels"
)

DEFAULT_PIECE_PROMPT = (
    "photorealistic macro product photo, same treat piece as reference, "
    "same shape/texture/color, centered on pure white seamless background, "
    "soft shadow, sharp focus, realistic details, no plate, no props"
)

DEFAULT_PACK_NEGATIVE = (
    "watermark, website logo overlay, extra stickers, new label, redesign, "
    "cartoon, illustration, 3d, cgi, blurry, low quality"
)

DEFAULT_PIECE_NEGATIVE = (
    "watermark, text, logo overlay, cartoon, illustration, 3d, cgi, blurry, "
    "low quality, extra objects"
)

DEFAULT_NEGATIVE = DEFAULT_PACK_NEGATIVE  # Keep for backward compatibility

MANDATORY_NEGATIVE_TOKENS = (
    "placeholder, mockup, template, generic, stock photo, "
    "cheese, dairy, yogurt, dessert, jam, sauce, fruit, garnish, "
    "plate, bowl, napkin, table, cutting board, spoon, fork, "
    "cartoon, illustration, anime, 3d render, cgi, low quality, blurry, "
    "watermark, logo overlay, extra text, fake label, distorted packaging"
)

@dataclass
class Settings:
    csv_path: Path
    pack_dir: Path
    piece_dir: Path
    output_dir: Path
    gen_pack: bool = True
    gen_piece: bool = True
    width: int = 1024
    height: int = 1024
    pack_strength: float = 0.90
    piece_strength: float = 0.90
    alchemy: bool = True
    skip_existing: bool = True
    inference_steps: int = 12
    model_profile: str = "CHEAP"  # CHEAP | HQ
    pack_num_images: int = 1
    piece_num_images: int = 1
    # note: resolution controlled by width/height
    poll_s: float = 2.0
    timeout_s: int = 240
    pack_prompt: str = DEFAULT_PACK_PROMPT
    piece_prompt: str = DEFAULT_PIECE_PROMPT
    negative_prompt: str = DEFAULT_NEGATIVE
    reject_watermarks: bool = True  # Reject refs with text/watermark overlays
    studio_mode: bool = True  # Studio Photo Mode
    normalize_framing: bool = False  # Optional postprocess for consistent framing
    enhance_refs: bool = True  # Enhance blurry refs (denoise+upscale+sharpen)

class LeonardoClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()
        self.headers_json = {
            "accept": "application/json",
            "authorization": f"Bearer {self.api_key}",
            "content-type": "application/json",
        }
        self.headers_get = {
            "accept": "application/json",
            "authorization": f"Bearer {self.api_key}",
        }

    def get_me(self) -> Dict[str, Any]:
        # official endpoint: GET /me
        r = self.session.get(f"{BASE_URL}/me", headers=self.headers_get, timeout=30)
        r.raise_for_status()
        return r.json()

    def init_image_upload(self, extension: str) -> Dict[str, Any]:
        # POST /init-image returns presigned S3 details
        payload = {"extension": extension}
        r = self.session.post(f"{BASE_URL}/init-image", headers=self.headers_json, json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
        return data["uploadInitImage"]

    def upload_init_image(self, file_path: Path) -> str:
        ext = file_path.suffix.lower()
        if ext not in ALLOWED_EXTS:
            raise ValueError(f"Unsupported extension: {ext}. Use {ALLOWED_EXTS}")

        # Leonardo expects extension string like "jpg/png/webp"
        ext_norm = ext.lstrip(".")
        if ext_norm == "jpeg":
            ext_norm = "jpg"

        info = self.init_image_upload(extension=ext_norm)

        init_image_id = info["id"]
        presigned_url = info["url"]
        fields = info["fields"]
        if isinstance(fields, str):
            fields = json.loads(fields)

        # Upload to S3 (no auth header, it's presigned)
        with file_path.open("rb") as f:
            files = {"file": (file_path.name, f)}
            resp = requests.post(presigned_url, data=fields, files=files, timeout=60)

        if resp.status_code not in (200, 204):
            raise RuntimeError(f"S3 upload failed ({resp.status_code}): {resp.text}")

        return init_image_id

    def create_generation(
        self,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        init_image_id: str,
        init_strength: float,
        alchemy: bool,
        num_images: int = 1,
        inference_steps: int = 12,
        model_profile: str = "CHEAP",
        studio_mode: bool = True,
        variant: str = "pack",
    ) -> Tuple[str, Optional[int]]:
        """
        Create generation with strict requirements:
        - init_image_id is REQUIRED (no text-to-image fallback)
        - Only uses init_image_id + init_strength (no imagePrompts/content reference)
        """
        if not init_image_id:
            raise ValueError("init_image_id is REQUIRED. Cannot generate without reference image.")
        
        # Clamp init_strength to safe range
        clamped_strength = clamp_init_strength(float(init_strength))
        
        model_id = MODEL_HQ if model_profile == "HQ" else MODEL_CHEAP
        
        # Build strict negative prompt (per-variant)
        strict_negative = build_strict_negative(negative_prompt, variant=variant, studio_mode=studio_mode)
        
        payload: Dict[str, Any] = {
            "prompt": prompt,
            "negative_prompt": strict_negative,
            "modelId": model_id,
            "width": int(width),
            "height": int(height),
            "num_images": int(num_images),
            "alchemy": bool(alchemy),
            "num_inference_steps": int(inference_steps),
            "init_image_id": init_image_id,
            "init_strength": clamped_strength,
        }

        # Check model compatibility
        if model_id in I2I_UNSUPPORTED_MODEL_IDS:
            mname = I2I_UNSUPPORTED_MODEL_IDS[model_id]
            raise ValueError(
                f"Model '{mname}' (modelId={model_id}) nepodporuje image-to-image cez init_image_id. "
                f"Nastav LEONARDO_MODEL_ID na SDXL model, napr. Leonardo Vision XL "
                f"(5c232a9e-9061-4777-980a-ddc8e65647c6) alebo Leonardo Diffusion XL "
                f"(1e60896f-3c26-4296-8ecc-53e2afecc132)."
            )

        r = self.session.post(f"{BASE_URL}/generations", headers=self.headers_json, json=payload, timeout=60)
        if not r.ok:
            txt = r.text or ""
            # Actionable message for your exact failure
            if r.status_code == 400 and "Image to image defaults do not exist for model" in txt:
                raise RuntimeError(
                    "Leonardo API 400: Vybraný model nepodporuje image-to-image cez init_image_id.\n"
                    f"Aktuálne modelId: {model_id}\n"
                    "Fix: nastav v .env LEONARDO_MODEL_ID na SDXL model (napr. Leonardo Vision XL) "
                    "a reštartuj appku.\n"
                    f"Raw: {txt}"
                )
            # Toto ti povie presný dôvod 400 (chýbajúci field, zlá hodnota, atď.)
            raise RuntimeError(f"Leonardo API error {r.status_code}: {r.text}")
        r.raise_for_status()
        data = r.json()

        job = data.get("sdGenerationJob") or {}
        gen_id = job.get("generationId")
        cost = job.get("apiCreditCost")

        if not gen_id:
            raise RuntimeError(f"No generationId in response: {data}")
        return gen_id, cost

    def get_generation(self, gen_id: str) -> Dict[str, Any]:
        r = self.session.get(f"{BASE_URL}/generations/{gen_id}", headers=self.headers_get, timeout=30)
        r.raise_for_status()
        return r.json()

    def wait_for_urls(self, gen_id: str, poll_s: float, timeout_s: int) -> List[str]:
        start = time.time()
        while True:
            data = self.get_generation(gen_id)
            g = data.get("generations_by_pk") or {}
            status = g.get("status")
            images = g.get("generated_images") or []

            if status == "COMPLETE" and images:
                urls = [img.get("url") for img in images if img.get("url")]
                if urls:
                    return urls

            if status == "FAILED":
                raise RuntimeError(f"Generation FAILED: {data}")

            if time.time() - start > timeout_s:
                raise TimeoutError(f"Timeout waiting for generation {gen_id}, last status={status}")

            time.sleep(poll_s)

    def download(self, url: str, out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with self.session.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with out_path.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 256):
                    if chunk:
                        f.write(chunk)

def load_api_key() -> str:
    # already loaded at import time, but keep as safety no-op
    load_dotenv(dotenv_path=ENV_PATH, override=False)
    api_key = os.getenv("LEONARDO_API_KEY")
    if not api_key:
        raise RuntimeError("Missing LEONARDO_API_KEY. Create .env from .env.example and paste your key.")
    return api_key

def safe_folder_name(sku: str, name: str, max_len: int = 120) -> str:
    """
    Create a safe folder name from SKU and name for Windows filesystem.
    Returns {sku}_{sanitized_name} or just {sku} if name is empty.
    Sanitized name: lowercase, spaces -> '-', remove invalid chars, limit length.
    Always keeps full SKU, truncates name if needed.
    """
    if not name or not name.strip():
        return sku
    
    # Clean name: replace invalid Windows chars, trim, collapse spaces
    invalid_chars = '<>:"/\\|?*'
    cleaned = name.strip()
    for char in invalid_chars:
        cleaned = cleaned.replace(char, "-")
    
    # Collapse multiple spaces to single space, then replace with dashes
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = cleaned.replace(' ', '-')
    
    # Lowercase
    cleaned = cleaned.lower()
    
    # Remove multiple consecutive dashes
    cleaned = re.sub(r'-+', '-', cleaned)
    
    # Remove leading/trailing dashes
    cleaned = cleaned.strip('-')
    
    # Build folder name
    folder_name = f"{sku}_{cleaned}"
    
    # Truncate if too long, but always keep full SKU
    if len(folder_name) > max_len:
        # Keep SKU + "_" + truncated name
        available_for_name = max_len - len(sku) - 1  # -1 for underscore
        if available_for_name > 0:
            folder_name = f"{sku}_{cleaned[:available_for_name]}"
        else:
            # If SKU itself is too long, just return SKU (shouldn't happen normally)
            folder_name = sku
    
    return folder_name

def find_ref_image(directory: Path, sku: str, suffix: str) -> Optional[Path]:
    # looks for <SKU>_<suffix>.<ext>
    for ext in ALLOWED_EXTS:
        p = directory / f"{sku}_{suffix}{ext}"
        if p.exists():
            return p
    return None

def detect_watermark_overlay(image_path: Path, edge_threshold: float = 15.0, non_white_threshold: float = 0.15) -> bool:
    """
    Detect if image contains text/watermark overlays using lightweight heuristics.
    Checks bottom-right region and bottom-center band for edge density and non-white pixels.
    
    Returns True if watermark/text overlay is detected.
    """
    try:
        with Image.open(image_path) as img:
            # Convert to grayscale
            gray = img.convert('L')
            width, height = gray.size
            
            # Region 1: Bottom-right (last 30% width × last 25% height)
            br_x0 = int(width * 0.70)
            br_y0 = int(height * 0.75)
            br_region = gray.crop((br_x0, br_y0, width, height))
            
            # Region 2: Bottom-center band (last 18% height × middle 60% width)
            bc_x0 = int(width * 0.20)
            bc_x1 = int(width * 0.80)
            bc_y0 = int(height * 0.82)
            bc_region = gray.crop((bc_x0, bc_y0, bc_x1, height))
            
            # Check both regions
            for region in [br_region, bc_region]:
                if region.size[0] == 0 or region.size[1] == 0:
                    continue
                
                # Compute edge density using FIND_EDGES
                edges = region.filter(ImageFilter.FIND_EDGES)
                edge_pixels = list(edges.getdata())
                edge_mean = statistics.mean(edge_pixels) if edge_pixels else 0
                
                # Compute non-white pixel ratio (threshold: >240 is near-white)
                region_pixels = list(region.getdata())
                non_white_count = sum(1 for p in region_pixels if p < 240)
                non_white_ratio = non_white_count / len(region_pixels) if region_pixels else 0
                
                # Flag if edge density OR non-white ratio exceeds threshold
                if edge_mean > edge_threshold or non_white_ratio > non_white_threshold:
                    return True
            
            return False
    except Exception as e:
        # If we can't analyze, assume no watermark (don't block generation)
        return False

# --- Image Preprocessing Functions ---

BLUR_THRESHOLD = 0.0035  # Threshold for sharpness score (variance of edges)

def estimate_sharpness(img: Image.Image) -> float:
    """
    Estimate image sharpness using edge variance.
    Convert to grayscale, apply FIND_EDGES, return normalized variance.
    """
    try:
        gray = img.convert('L')
        edges = gray.filter(ImageFilter.FIND_EDGES)
        edge_array = np.array(edges, dtype=np.float32)
        variance = float(np.var(edge_array))
        # Normalize by dividing by max possible variance (255^2 = 65025)
        normalized = variance / 65025.0
        return normalized
    except Exception:
        return 0.0

def autocrop_nonwhite(img: Image.Image, threshold: int = 248, pad_ratio: float = 0.10) -> Image.Image:
    """
    Auto-crop image to remove large white margins.
    Finds bounding box of non-white pixels and crops with padding.
    """
    try:
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Convert to numpy array
        arr = np.array(img)
        h, w, c = arr.shape
        
        # Create mask: True where any channel < threshold (non-white)
        mask = np.any(arr < threshold, axis=2)
        
        # If no non-white pixels, return original
        if not np.any(mask):
            return img
        
        # Find bounding box
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            return img
        
        top = np.argmax(rows)
        bottom = len(rows) - np.argmax(rows[::-1])
        left = np.argmax(cols)
        right = len(cols) - np.argmax(cols[::-1])
        
        # Add padding
        bbox_w = right - left
        bbox_h = bottom - top
        pad = int(max(bbox_w, bbox_h) * pad_ratio)
        
        # Clamp to image bounds
        crop_left = max(0, left - pad)
        crop_top = max(0, top - pad)
        crop_right = min(w, right + pad)
        crop_bottom = min(h, bottom + pad)
        
        cropped = img.crop((crop_left, crop_top, crop_right, crop_bottom))
        return cropped
    except Exception:
        return img

def preprocess_ref_image(
    in_path: Path,
    out_path: Path,
    target_w: int,
    target_h: int,
    enhance: bool = True
) -> Tuple[Path, float]:
    """
    Preprocess reference image: auto-crop, resize, enhance, and compute sharpness.
    Returns (out_path, sharpness_score).
    """
    try:
        # Load image
        img = Image.open(in_path)
        
        # Convert to RGB, composite on white if has alpha
        if img.mode == 'RGBA':
            white_bg = Image.new('RGB', img.size, (255, 255, 255))
            white_bg.paste(img, mask=img.split()[3])  # Use alpha channel as mask
            img = white_bg
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Auto-crop to remove huge white margins
        img = autocrop_nonwhite(img)
        
        # Resize to fit inside target dimensions while keeping aspect ratio
        img_w, img_h = img.size
        scale = min(target_w / img_w, target_h / img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Paste centered on pure white canvas
        canvas = Image.new('RGB', (target_w, target_h), (255, 255, 255))
        paste_x = (target_w - new_w) // 2
        paste_y = (target_h - new_h) // 2
        canvas.paste(img, (paste_x, paste_y))
        img = canvas
        
        # Enhance if requested
        if enhance:
            # Mild denoise for JPEG artifacts
            img = img.filter(ImageFilter.MedianFilter(size=3))
            
            # Unsharp mask for sharpening
            img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=180, threshold=3))
            
            # Contrast enhancement
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.08)  # 1.05-1.10 range
            
            # Sharpness enhancement
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(1.10)  # 1.05-1.15 range
        
        # Compute sharpness score on final image
        sharpness_score = estimate_sharpness(img)
        
        # Save to output path (ensure parent dirs exist)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path, 'PNG')
        
        return (out_path, sharpness_score)
    except Exception as e:
        # If preprocessing fails, return original path and low sharpness
        return (in_path, 0.0)

def build_strict_prompt(kind: str, base_prompt: str, name: str, studio_mode: bool = True) -> str:
    """
    Build a strict anti-placeholder prompt that forces real photography look.
    kind: "pack" or "piece"
    base_prompt: original prompt from settings
    name: product name from CSV (may be empty)
    studio_mode: if True, wrap with strict studio photo constraints
    """
    if studio_mode:
        constraints = [
            "real product photo, studio product photography, DSLR",
            "pure white seamless background",
            "single product only",
            "natural soft shadow under product",
            "centered, sharp focus",
            "do not add any text, logos, labels, props, plates, bowls, hands",
            "match the reference image shape, material, texture and color"
        ]
        constraint_text = ", ".join(constraints)
        
        if name:
            prompt = f"{name}. {constraint_text}. {base_prompt}"
        else:
            prompt = f"{constraint_text}. {base_prompt}"
    else:
        # No wrapping, use base prompt as-is
        if name:
            prompt = f"{name}. {base_prompt}"
        else:
            prompt = base_prompt
    
    return prompt

def build_strict_negative(user_negative: str, variant: str = "pack", studio_mode: bool = True) -> str:
    """
    Build negative prompt with mandatory anti-placeholder tokens.
    variant: "pack" or "piece" - uses different base negatives
    studio_mode: if True, add strong studio photo negative tokens
    """
    user_tokens = user_negative.strip() if user_negative else ""
    
    # Use variant-specific base negative prompts
    if variant == "pack":
        base_negative = DEFAULT_PACK_NEGATIVE
    else:  # piece
        base_negative = DEFAULT_PIECE_NEGATIVE
    
    if studio_mode:
        # Strong negative tokens for studio photo mode
        studio_negatives = (
            "placeholder, mockup, template, "
            "plate, bowl, napkin, table, garnish, sauce, jam, dessert, cheese, dairy, yogurt, "
            "cartoon, illustration, 3d render, cgi, low quality, blurry"
        )
        
        if user_tokens:
            combined = f"{user_tokens}, {base_negative}, {studio_negatives}"
        else:
            combined = f"{base_negative}, {studio_negatives}"
    else:
        # Standard mode: just use base negative
        if user_tokens:
            combined = f"{user_tokens}, {base_negative}"
        else:
            combined = base_negative
    
    return combined

def clamp_init_strength(value: float, min_val: float = 0.70, max_val: float = 0.98) -> float:
    """Clamp init_strength to safe range [0.70, 0.98]."""
    return max(min_val, min(max_val, value))

def detect_extension_from_url(url: str) -> str:
    # heuristic; we download as .png by default
    lower = url.lower()
    for ext in [".png", ".jpg", ".jpeg", ".webp"]:
        if lower.endswith(ext):
            return ext if ext != ".jpeg" else ".jpg"
    return ".png"

def normalize_product_framing(image_path: Path, target_width: int, target_height: int, padding_ratio: float = 0.10) -> None:
    """
    Postprocess generated image for consistent "camera product shot framing":
    - Detect non-white bounding box
    - Crop to product with padding
    - Paste centered on white canvas of target size
    - Save over original file
    
    This standardizes composition and avoids weird extra whitespace.
    Does NOT attempt to remove watermarks/overlays.
    """
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Convert to grayscale for threshold detection
            gray = img.convert('L')
            width, height = img.size
            
            # Find non-white bounding box (threshold: >240 is near-white)
            # Scan from edges inward
            left = 0
            right = width - 1
            top = 0
            bottom = height - 1
            
            # Find left edge
            for x in range(width):
                col = [gray.getpixel((x, y)) for y in range(height)]
                if any(p < 240 for p in col):
                    left = x
                    break
            
            # Find right edge
            for x in range(width - 1, -1, -1):
                col = [gray.getpixel((x, y)) for y in range(height)]
                if any(p < 240 for p in col):
                    right = x
                    break
            
            # Find top edge
            for y in range(height):
                row = [gray.getpixel((x, y)) for x in range(width)]
                if any(p < 240 for p in row):
                    top = y
                    break
            
            # Find bottom edge
            for y in range(height - 1, -1, -1):
                row = [gray.getpixel((x, y)) for x in range(width)]
                if any(p < 240 for p in row):
                    bottom = y
                    break
            
            # If no non-white pixels found, skip processing
            if left >= right or top >= bottom:
                return
            
            # Calculate padding
            bbox_width = right - left + 1
            bbox_height = bottom - top + 1
            pad_x = int(bbox_width * padding_ratio)
            pad_y = int(bbox_height * padding_ratio)
            
            # Crop with padding (clamp to image bounds)
            crop_left = max(0, left - pad_x)
            crop_top = max(0, top - pad_y)
            crop_right = min(width, right + 1 + pad_x)
            crop_bottom = min(height, bottom + 1 + pad_y)
            
            cropped = img.crop((crop_left, crop_top, crop_right, crop_bottom))
            
            # Create white canvas of target size
            canvas = Image.new('RGB', (target_width, target_height), (255, 255, 255))
            
            # Calculate centering position
            paste_x = (target_width - cropped.width) // 2
            paste_y = (target_height - cropped.height) // 2
            
            # Paste centered
            canvas.paste(cropped, (paste_x, paste_y))
            
            # Save over original
            canvas.save(image_path, 'PNG')
    except Exception as e:
        # If processing fails, keep original image
        pass

def ensure_manifest(manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if not manifest_path.exists():
        with manifest_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "sku", "name", "variant", "generation_id", "api_credit_cost", "file_path",
                "profile", "width", "height", "steps", "alchemy", "init_strength", "modelId"
            ])

def append_manifest(manifest_path: Path, row: List[Any]) -> None:
    with manifest_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(row)

def read_skus(csv_path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "sku" not in [h.lower() for h in reader.fieldnames]:
            raise ValueError("CSV must have a 'sku' column.")
        # normalize key lookup
        field_map = {h.lower(): h for h in reader.fieldnames}
        sku_key = field_map["sku"]
        name_key = field_map.get("name")

        for r in reader:
            sku = (r.get(sku_key) or "").strip()
            if not sku:
                continue
            name = (r.get(name_key) or "").strip() if name_key else ""
            rows.append({"sku": sku, "name": name})
    return rows

# ---------------- GUI ----------------

class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Laurapets Leonardo Tool")
        self.root.geometry("980x680")

        self.log_q: Queue[str] = Queue()
        self.stop_event = threading.Event()
        self.worker: Optional[threading.Thread] = None

        # Defaults
        self.csv_var = tk.StringVar(value=str(Path("skus.csv").resolve()))
        self.pack_dir_var = tk.StringVar(value=str(Path("input/pack").resolve()))
        self.piece_dir_var = tk.StringVar(value=str(Path("input/piece").resolve()))
        self.output_dir_var = tk.StringVar(value=str(Path("output").resolve()))

        self.width_var = tk.IntVar(value=1024)
        self.height_var = tk.IntVar(value=1024)
        self.pack_strength_var = tk.DoubleVar(value=0.90)
        self.piece_strength_var = tk.DoubleVar(value=0.90)
        self.alchemy_var = tk.BooleanVar(value=True)
        self.skip_existing_var = tk.BooleanVar(value=True)
        self.steps_var = tk.IntVar(value=12)
        self.hq_var = tk.BooleanVar(value=False)
        self.profile_var = tk.StringVar(value="CHEAP")

        self.gen_pack_var = tk.BooleanVar(value=True)
        self.gen_piece_var = tk.BooleanVar(value=True)

        self.progress_var = tk.DoubleVar(value=0.0)
        self.status_var = tk.StringVar(value="Idle.")

        self.pack_prompt = tk.StringVar(value=DEFAULT_PACK_PROMPT)
        self.piece_prompt = tk.StringVar(value=DEFAULT_PIECE_PROMPT)
        self.negative_prompt = tk.StringVar(value=DEFAULT_NEGATIVE)
        
        self.reject_watermarks_var = tk.BooleanVar(value=True)
        self.studio_mode_var = tk.BooleanVar(value=True)
        self.normalize_framing_var = tk.BooleanVar(value=False)
        self.enhance_refs_var = tk.BooleanVar(value=True)

        self._build_ui()
        self._poll_log_queue()
        # Apply initial profile to sync UI
        self.apply_profile(self.profile_var.get())
        # Apply initial profile to sync UI
        self.apply_profile(self.profile_var.get())

    def _build_ui(self):
        pad = 12

        header = ttk.Frame(self.root, padding=pad)
        header.pack(fill="x")

        ttk.Label(header, text="Laurapets – Leonardo Batch Generator", font=("Segoe UI", 16, "bold")).pack(anchor="w")
        ttk.Label(
            header,
            text="Goal: generate 2 images per SKU (pack + piece) and save to disk.",
            font=("Segoe UI", 10),
        ).pack(anchor="w", pady=(4, 0))

        # Actions
        actions = ttk.Frame(self.root, padding=pad)
        actions.pack(fill="x")

        ttk.Button(actions, text="Test API Key (GET /me)", command=self.on_test_api).pack(side="left")
        ttk.Button(actions, text="Validate Batch (files + CSV)", command=self.on_validate).pack(side="left", padx=(8, 0))
        ttk.Button(actions, text="Start Generation", command=self.on_start).pack(side="left", padx=(8, 0))
        ttk.Button(actions, text="Stop", command=self.on_stop).pack(side="left", padx=(8, 0))
        ttk.Button(actions, text="Open Output Folder", command=self.on_open_output).pack(side="left", padx=(8, 0))

        # Paths
        paths = ttk.LabelFrame(self.root, text="Paths", padding=pad)
        paths.pack(fill="x", padx=pad, pady=(0, pad))

        self._row_path(paths, "CSV (skus.csv):", self.csv_var, self.browse_csv)
        self._row_path(paths, "Pack refs dir:", self.pack_dir_var, lambda: self.browse_dir(self.pack_dir_var))
        self._row_path(paths, "Piece refs dir:", self.piece_dir_var, lambda: self.browse_dir(self.piece_dir_var))
        self._row_path(paths, "Output dir:", self.output_dir_var, lambda: self.browse_dir(self.output_dir_var))

        # Settings
        settings = ttk.LabelFrame(self.root, text="Generation Settings", padding=pad)
        settings.pack(fill="x", padx=pad, pady=(0, pad))

        row1 = ttk.Frame(settings)
        row1.pack(fill="x")
        ttk.Label(row1, text="Width").pack(side="left")
        ttk.Entry(row1, textvariable=self.width_var, width=8).pack(side="left", padx=(6, 16))
        ttk.Label(row1, text="Height").pack(side="left")
        ttk.Entry(row1, textvariable=self.height_var, width=8).pack(side="left", padx=(6, 16))
        ttk.Checkbutton(row1, text="Generate pack", variable=self.gen_pack_var).pack(side="left", padx=(0, 12))
        ttk.Checkbutton(row1, text="Generate piece", variable=self.gen_piece_var).pack(side="left", padx=(0, 16))
        ttk.Checkbutton(row1, text="Alchemy", variable=self.alchemy_var).pack(side="left", padx=(0, 16))
        ttk.Checkbutton(row1, text="Skip existing outputs", variable=self.skip_existing_var).pack(side="left")

        ttk.Label(row1, text="Profile").pack(side="left", padx=(16, 0))
        profile_combo = ttk.Combobox(row1, textvariable=self.profile_var, values=["CHEAP", "HQ"], width=6, state="readonly")
        profile_combo.pack(side="left", padx=(6, 16))
        profile_combo.bind("<<ComboboxSelected>>", lambda e: self.apply_profile(self.profile_var.get()))
        ttk.Label(row1, text="Steps").pack(side="left", padx=(0, 0))
        ttk.Entry(row1, textvariable=self.steps_var, width=6).pack(side="left", padx=(6, 16))

        ttk.Checkbutton(row1, text="HQ model", variable=self.hq_var).pack(side="left", padx=(0, 12))
        ttk.Button(row1, text="Preset: CHEAP", command=self.apply_preset_cheap).pack(side="left", padx=(0, 8))
        ttk.Button(row1, text="Preset: HQ", command=self.apply_preset_hq).pack(side="left")

        row2 = ttk.Frame(settings)
        row2.pack(fill="x", pady=(8, 0))
        ttk.Label(row2, text="Pack init_strength").pack(side="left")
        ttk.Entry(row2, textvariable=self.pack_strength_var, width=8).pack(side="left", padx=(6, 16))
        ttk.Label(row2, text="Piece init_strength").pack(side="left")
        ttk.Entry(row2, textvariable=self.piece_strength_var, width=8).pack(side="left", padx=(6, 16))
        
        row3 = ttk.Frame(settings)
        row3.pack(fill="x", pady=(8, 0))
        ttk.Checkbutton(row3, text="Reject refs with text/watermark overlays (recommended)", variable=self.reject_watermarks_var).pack(side="left", padx=(0, 16))
        ttk.Checkbutton(row3, text="Studio Photo Mode", variable=self.studio_mode_var).pack(side="left", padx=(0, 16))
        ttk.Checkbutton(row3, text="Normalize framing (postprocess)", variable=self.normalize_framing_var).pack(side="left", padx=(0, 16))
        ttk.Checkbutton(row3, text="Enhance blurry refs (denoise+upscale+sharpen)", variable=self.enhance_refs_var).pack(side="left")

        # Prompts
        prompts = ttk.LabelFrame(self.root, text="Prompts", padding=pad)
        prompts.pack(fill="both", expand=False, padx=pad, pady=(0, pad))

        ttk.Label(prompts, text="Pack prompt:").pack(anchor="w")
        ttk.Entry(prompts, textvariable=self.pack_prompt).pack(fill="x", pady=(2, 8))

        ttk.Label(prompts, text="Piece prompt:").pack(anchor="w")
        ttk.Entry(prompts, textvariable=self.piece_prompt).pack(fill="x", pady=(2, 8))

        ttk.Label(prompts, text="Negative prompt:").pack(anchor="w")
        ttk.Entry(prompts, textvariable=self.negative_prompt).pack(fill="x", pady=(2, 0))

        # Progress + status
        prog = ttk.Frame(self.root, padding=(pad, 0, pad, pad))
        prog.pack(fill="x")

        ttk.Label(prog, textvariable=self.status_var).pack(anchor="w")
        ttk.Progressbar(prog, variable=self.progress_var, maximum=100.0).pack(fill="x", pady=(6, 0))

        # Log
        log_frame = ttk.LabelFrame(self.root, text="Log", padding=pad)
        log_frame.pack(fill="both", expand=True, padx=pad, pady=(0, pad))

        self.log_box = tk.Text(log_frame, wrap="word", state="disabled")
        self.log_box.pack(fill="both", expand=True)

    def _row_path(self, parent, label: str, var: tk.StringVar, browse_cmd):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=4)
        ttk.Label(row, text=label, width=16).pack(side="left")
        ttk.Entry(row, textvariable=var).pack(side="left", fill="x", expand=True, padx=(6, 6))
        ttk.Button(row, text="Browse", command=browse_cmd, width=10).pack(side="left")

    def _log(self, msg: str):
        self.log_q.put(msg)

    def _poll_log_queue(self):
        try:
            while True:
                msg = self.log_q.get_nowait()
                self.log_box.configure(state="normal")
                self.log_box.insert("end", msg + "\n")
                self.log_box.see("end")
                self.log_box.configure(state="disabled")
        except Empty:
            pass
        self.root.after(120, self._poll_log_queue)

    def browse_csv(self):
        p = filedialog.askopenfilename(
            title="Select skus.csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if p:
            self.csv_var.set(p)

    def browse_dir(self, var: tk.StringVar):
        p = filedialog.askdirectory(title="Select folder")
        if p:
            var.set(p)

    def on_open_output(self):
        out_dir = Path(self.output_dir_var.get())
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(out_dir))  # Windows
        except Exception as e:
            messagebox.showerror("Error", f"Cannot open folder: {e}")

    def apply_profile(self, profile: str):
        """Apply profile preset to UI (width, height, steps, alchemy, model profile, strengths)."""
        if profile == "CHEAP":
            self.width_var.set(768)
            self.height_var.set(768)
            self.steps_var.set(30)
            self.alchemy_var.set(True)
            self.pack_strength_var.set(0.9)
            self.piece_strength_var.set(0.9)
            if hasattr(self, 'hq_var'):
                self.hq_var.set(False)
            self.profile_var.set("CHEAP")
            self._log(f"Applied profile CHEAP: 768×768, steps=30, alchemy=ON, strength=0.9")
        elif profile == "HQ":
            self.width_var.set(1024)
            self.height_var.set(1024)
            self.steps_var.set(30)
            self.alchemy_var.set(True)
            self.pack_strength_var.set(0.9)
            self.piece_strength_var.set(0.9)
            if hasattr(self, 'hq_var'):
                self.hq_var.set(True)
            self.profile_var.set("HQ")
            self._log(f"Applied profile HQ: 1024×1024, steps=30, alchemy=ON, strength=0.9")

    def apply_preset_cheap(self):
        self.apply_profile("CHEAP")

    def apply_preset_hq(self):
        self.apply_profile("HQ")

    def _get_settings(self) -> Settings:
        return Settings(
            csv_path=Path(self.csv_var.get()),
            pack_dir=Path(self.pack_dir_var.get()),
            piece_dir=Path(self.piece_dir_var.get()),
            output_dir=Path(self.output_dir_var.get()),
            gen_pack=bool(self.gen_pack_var.get()),
            gen_piece=bool(self.gen_piece_var.get()),
            width=int(self.width_var.get()),
            height=int(self.height_var.get()),
            pack_strength=float(self.pack_strength_var.get()),
            piece_strength=float(self.piece_strength_var.get()),
            alchemy=bool(self.alchemy_var.get()),
            skip_existing=bool(self.skip_existing_var.get()),
            inference_steps=int(self.steps_var.get()),
            model_profile=self.profile_var.get(),
            pack_num_images=1,
            piece_num_images=1,
            pack_prompt=self.pack_prompt.get().strip(),
            piece_prompt=self.piece_prompt.get().strip(),
            negative_prompt=self.negative_prompt.get().strip(),
            reject_watermarks=bool(self.reject_watermarks_var.get()),
            studio_mode=bool(self.studio_mode_var.get()),
            normalize_framing=bool(self.normalize_framing_var.get()),
            enhance_refs=bool(self.enhance_refs_var.get()),
        )

    def on_test_api(self):
        try:
            api_key = load_api_key()
            client = LeonardoClient(api_key)
            me = client.get_me()
            self._log(f"Using modelId: {get_model_id()}")
            user = me.get("user") or {}
            username = user.get("username") or user.get("email") or "unknown"
            self._log(f"API OK ✅  Logged in as: {username}")
            # token snapshot (if present)
            tokens = user.get("subscriptionTokens") or user.get("subscription_tokens") or None
            if tokens is not None:
                self._log(f"Token snapshot: {tokens}")
            self.status_var.set("API check OK.")
        except Exception as e:
            messagebox.showerror("API Test Failed", str(e))
            self.status_var.set("API check failed.")

    def on_validate(self):
        try:
            s = self._get_settings()

            if not s.gen_pack and not s.gen_piece:
                messagebox.showerror("Validation", "Select at least one: Generate pack or Generate piece.")
                return

            if not s.csv_path.exists():
                raise FileNotFoundError(f"CSV not found: {s.csv_path}")
            if not s.pack_dir.exists():
                raise FileNotFoundError(f"Pack dir not found: {s.pack_dir}")
            if not s.piece_dir.exists():
                raise FileNotFoundError(f"Piece dir not found: {s.piece_dir}")

            rows = read_skus(s.csv_path)
            self._log(f"Loaded {len(rows)} SKUs from CSV.")

            missing_pack = 0
            missing_piece = 0
            ok_any = 0

            for r in rows:
                sku = r["sku"]
                pack_ref = find_ref_image(s.pack_dir, sku, "pack")
                piece_ref = find_ref_image(s.piece_dir, sku, "piece")

                will_do_any = False

                if s.gen_pack:
                    if not pack_ref:
                        missing_pack += 1
                        self._log(f"[MISSING PACK] {sku}")
                    else:
                        self._log(f"[OK PACK] {sku} ref={pack_ref.name}")
                if s.gen_piece:
                    if not piece_ref:
                        missing_piece += 1
                        self._log(f"[MISSING PIECE] {sku}")
                    else:
                        self._log(f"[OK PIECE] {sku} ref={piece_ref.name}")

                if (s.gen_pack and pack_ref) or (s.gen_piece and piece_ref):
                    will_do_any = True

                if will_do_any:
                    ok_any += 1

            self._log(
                f"Validation summary: "
                f"ready={ok_any}/{len(rows)} | "
                f"missing_pack={missing_pack if s.gen_pack else 'n/a'} | "
                f"missing_piece={missing_piece if s.gen_piece else 'n/a'}"
            )

            if ok_any == 0:
                self.status_var.set("Validation: nothing to generate.")
            else:
                self.status_var.set("Validation OK (partial allowed).")

        except Exception as e:
            messagebox.showerror("Validation Failed", str(e))
            self.status_var.set("Validation failed.")

    def on_start(self):
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("Busy", "Generation is already running.")
            return

        self.stop_event.clear()
        self.progress_var.set(0.0)
        self.status_var.set("Starting...")

        self.worker = threading.Thread(target=self._run_batch, daemon=True)
        self.worker.start()

    def on_stop(self):
        self.stop_event.set()
        self._log("Stop requested. Finishing current step, then aborting…")
        self.status_var.set("Stopping...")

    def _run_batch(self):
        try:
            api_key = load_api_key()
            client = LeonardoClient(api_key)
            s = self._get_settings()

            if not s.gen_pack and not s.gen_piece:
                messagebox.showerror("Start", "Select at least one: Generate pack or Generate piece.")
                self.status_var.set("Idle.")
                return

            rows = read_skus(s.csv_path)

            # Prepare output directory
            s.output_dir.mkdir(parents=True, exist_ok=True)

            # Pre-compute total work units (for accurate progress)
            planned = 0
            for r in rows:
                sku = r["sku"]
                name = r.get("name", "")
                folder_name = safe_folder_name(sku, name)
                pack_ref = find_ref_image(s.pack_dir, sku, "pack")
                piece_ref = find_ref_image(s.piece_dir, sku, "piece")

                out_sku_dir = s.output_dir / folder_name
                out_pack = out_sku_dir / f"{sku}__pack.png"
                out_piece = out_sku_dir / f"{sku}__piece.png"

                if s.gen_pack and pack_ref:
                    if not (s.skip_existing and out_pack.exists()):
                        planned += 1
                if s.gen_piece and piece_ref:
                    if not (s.skip_existing and out_piece.exists()):
                        planned += 1

            if planned == 0:
                self._log("Nothing to generate (either missing refs or outputs already exist).")
                self.status_var.set("Nothing to do.")
                self.progress_var.set(0.0)
                return

            done = 0
            self._log(f"Batch start: {len(rows)} SKUs -> {planned} images planned.")
            current_model = MODEL_HQ if s.model_profile == "HQ" else MODEL_CHEAP
            self._log(f"Profile={s.model_profile} modelId={current_model} | {s.width}x{s.height} | steps={s.inference_steps} | alchemy={s.alchemy}")

            for r in rows:
                if self.stop_event.is_set():
                    self._log("Stopped by user.")
                    self.status_var.set("Stopped.")
                    return

                sku = r["sku"]
                name = r.get("name", "")
                folder_name = safe_folder_name(sku, name)

                pack_ref = find_ref_image(s.pack_dir, sku, "pack")
                piece_ref = find_ref_image(s.piece_dir, sku, "piece")

                out_sku_dir = s.output_dir / folder_name
                out_sku_dir.mkdir(parents=True, exist_ok=True)

                # Per-SKU manifest
                manifest_path = out_sku_dir / "manifest.csv"
                ensure_manifest(manifest_path)

                out_pack = out_sku_dir / f"{sku}__pack.png"
                out_piece = out_sku_dir / f"{sku}__piece.png"

                # Log before generating each SKU
                current_model = MODEL_HQ if s.model_profile == "HQ" else MODEL_CHEAP
                self._log(f"Profile={s.model_profile} modelId={current_model} | {s.width}x{s.height} | steps={s.inference_steps} | alchemy={s.alchemy}")

                # --- PACK (optional) ---
                if s.gen_pack:
                    if not pack_ref:
                        self._log(f"[SKIP] {sku} Missing PACK ref")
                        continue
                    
                    # Watermark detection
                    if detect_watermark_overlay(pack_ref):
                        if s.reject_watermarks:
                            self._log(f"[SKIP] {sku} pack ref looks like it contains text/watermark overlay. Please use a clean photo/reference.")
                            continue
                        else:
                            self._log(f"[WARNING] {sku} pack ref may contain text/watermark overlay, but continuing (reject_watermarks is OFF).")
                    
                    if not (s.skip_existing and out_pack.exists()):
                        # Preprocess reference image
                        prep_dir = out_sku_dir / "_prep"
                        prep_pack = prep_dir / f"{sku}__pack_prep.png"
                        prep_path, sharpness_score = preprocess_ref_image(
                            pack_ref, prep_pack, s.width, s.height, enhance=s.enhance_refs
                        )
                        self._log(f"[{sku}] Preprocessed pack ref: sharpness={sharpness_score:.6f}")
                        
                        # Auto-tune init_strength based on sharpness
                        base_strength = s.pack_strength
                        if sharpness_score < BLUR_THRESHOLD:
                            effective_strength = max(0.70, min(base_strength, 0.78))
                        else:
                            effective_strength = base_strength
                        effective_strength = clamp_init_strength(effective_strength)
                        
                        self.status_var.set(f"{sku}: uploading pack ref…")
                        self._log(f"[{sku}] Using pack ref: {pack_ref.name} (preprocessed: {prep_path.name})")
                        pack_init_id = client.upload_init_image(prep_path)
                        
                        # Build strict prompt with studio mode
                        pack_prompt_text = build_strict_prompt("pack", s.pack_prompt, name, studio_mode=s.studio_mode)
                        
                        current_model = MODEL_HQ if s.model_profile == "HQ" else MODEL_CHEAP
                        
                        # Debug payload log
                        self._log(f"[PAYLOAD] {sku} PACK | {s.width}x{s.height} | steps={s.inference_steps} | alchemy={s.alchemy} | modelId={current_model} | sharpness={sharpness_score:.6f} base_strength={base_strength:.2f} effective_strength={effective_strength:.2f} | ref={pack_ref.name}")

                        self.status_var.set(f"{sku}: generating PACK…")
                        gen_id, cost = client.create_generation(
                            prompt=pack_prompt_text,
                            negative_prompt=s.negative_prompt,
                            width=s.width,
                            height=s.height,
                            init_image_id=pack_init_id,
                            init_strength=effective_strength,
                            alchemy=s.alchemy,
                            num_images=s.pack_num_images,
                            inference_steps=s.inference_steps,
                            model_profile=s.model_profile,
                            studio_mode=s.studio_mode,
                            variant="pack",
                        )
                        
                        self._log(f"[{sku}] PACK gen_id={gen_id} cost={cost}")

                        urls = client.wait_for_urls(gen_id, poll_s=s.poll_s, timeout_s=s.timeout_s)
                        client.download(urls[0], out_pack)
                        
                        # Optional postprocess for framing
                        if s.normalize_framing:
                            normalize_product_framing(out_pack, s.width, s.height)
                            self._log(f"[{sku}] Applied framing normalization to PACK")
                        
                        # Save settings.json
                        settings_json = {
                            "sku": sku,
                            "name": name,
                            "variant": "pack",
                            "profile": s.model_profile,
                            "width": s.width,
                            "height": s.height,
                            "steps": s.inference_steps,
                            "alchemy": s.alchemy,
                            "init_strength": effective_strength,
                            "modelId": MODEL_HQ if s.model_profile == "HQ" else MODEL_CHEAP,
                            "prompt": pack_prompt_text,
                            "negative_prompt": s.negative_prompt,
                        }
                        settings_path = out_sku_dir / "pack_settings.json"
                        with settings_path.open("w", encoding="utf-8") as f:
                            json.dump(settings_json, f, indent=2)
                        
                        append_manifest(manifest_path, [
                            sku, name, "pack", gen_id, cost, str(out_pack),
                            s.model_profile, s.width, s.height, s.inference_steps,
                            s.alchemy, s.pack_strength,
                            MODEL_HQ if s.model_profile == "HQ" else MODEL_CHEAP
                        ])
                        self._log(f"[{sku}] Saved PACK -> {out_pack.name}")

                        done += 1
                        self.progress_var.set(done / planned * 100.0)
                    else:
                        self._log(f"[{sku}] PACK exists, skipping.")

                if self.stop_event.is_set():
                    self._log("Stopped by user.")
                    self.status_var.set("Stopped.")
                    return

                # --- PIECE (optional) ---
                if s.gen_piece:
                    if not piece_ref:
                        self._log(f"[SKIP] {sku} Missing PIECE ref")
                        continue
                    
                    # Watermark detection
                    if detect_watermark_overlay(piece_ref):
                        if s.reject_watermarks:
                            self._log(f"[SKIP] {sku} piece ref looks like it contains text/watermark overlay. Please use a clean photo/reference.")
                            continue
                        else:
                            self._log(f"[WARNING] {sku} piece ref may contain text/watermark overlay, but continuing (reject_watermarks is OFF).")
                    
                    if not (s.skip_existing and out_piece.exists()):
                        # Preprocess reference image
                        prep_dir = out_sku_dir / "_prep"
                        prep_piece = prep_dir / f"{sku}__piece_prep.png"
                        prep_path, sharpness_score = preprocess_ref_image(
                            piece_ref, prep_piece, s.width, s.height, enhance=s.enhance_refs
                        )
                        self._log(f"[{sku}] Preprocessed piece ref: sharpness={sharpness_score:.6f}")
                        
                        # Auto-tune init_strength based on sharpness
                        base_strength = s.piece_strength
                        if sharpness_score < BLUR_THRESHOLD:
                            effective_strength = max(0.70, min(base_strength, 0.78))
                        else:
                            effective_strength = base_strength
                        effective_strength = clamp_init_strength(effective_strength)
                        
                        self.status_var.set(f"{sku}: uploading piece ref…")
                        self._log(f"[{sku}] Using piece ref: {piece_ref.name} (preprocessed: {prep_path.name})")
                        piece_init_id = client.upload_init_image(prep_path)
                        
                        # Build strict prompt with studio mode
                        piece_prompt_text = build_strict_prompt("piece", s.piece_prompt, name, studio_mode=s.studio_mode)
                        
                        current_model = MODEL_HQ if s.model_profile == "HQ" else MODEL_CHEAP
                        
                        # Debug payload log
                        self._log(f"[PAYLOAD] {sku} PIECE | {s.width}x{s.height} | steps={s.inference_steps} | alchemy={s.alchemy} | modelId={current_model} | sharpness={sharpness_score:.6f} base_strength={base_strength:.2f} effective_strength={effective_strength:.2f} | ref={piece_ref.name}")

                        self.status_var.set(f"{sku}: generating PIECE…")
                        gen_id2, cost2 = client.create_generation(
                            prompt=piece_prompt_text,
                            negative_prompt=s.negative_prompt,
                            width=s.width,
                            height=s.height,
                            init_image_id=piece_init_id,
                            init_strength=effective_strength,
                            alchemy=s.alchemy,
                            num_images=s.piece_num_images,
                            inference_steps=s.inference_steps,
                            model_profile=s.model_profile,
                            studio_mode=s.studio_mode,
                            variant="piece",
                        )
                        
                        self._log(f"[{sku}] PIECE gen_id={gen_id2} cost={cost2}")

                        urls2 = client.wait_for_urls(gen_id2, poll_s=s.poll_s, timeout_s=s.timeout_s)
                        client.download(urls2[0], out_piece)
                        
                        # Optional postprocess for framing
                        if s.normalize_framing:
                            normalize_product_framing(out_piece, s.width, s.height)
                            self._log(f"[{sku}] Applied framing normalization to PIECE")
                        
                        # Save settings.json
                        settings_json = {
                            "sku": sku,
                            "name": name,
                            "variant": "piece",
                            "profile": s.model_profile,
                            "width": s.width,
                            "height": s.height,
                            "steps": s.inference_steps,
                            "alchemy": s.alchemy,
                            "init_strength": effective_strength,
                            "modelId": MODEL_HQ if s.model_profile == "HQ" else MODEL_CHEAP,
                            "prompt": piece_prompt_text,
                            "negative_prompt": s.negative_prompt,
                        }
                        settings_path = out_sku_dir / "piece_settings.json"
                        with settings_path.open("w", encoding="utf-8") as f:
                            json.dump(settings_json, f, indent=2)
                        
                        append_manifest(manifest_path, [
                            sku, name, "piece", gen_id2, cost2, str(out_piece),
                            s.model_profile, s.width, s.height, s.inference_steps,
                            s.alchemy, effective_strength,
                            MODEL_HQ if s.model_profile == "HQ" else MODEL_CHEAP
                        ])
                        self._log(f"[{sku}] Saved PIECE -> {out_piece.name}")

                        done += 1
                        self.progress_var.set(done / planned * 100.0)
                    else:
                        self._log(f"[{sku}] PIECE exists, skipping.")

            self.status_var.set("Done ✅")
            self._log("Batch complete.")

        except Exception as e:
            self.status_var.set("Error.")
            self._log(f"[ERROR] {e}")
            messagebox.showerror("Generation Error", str(e))

def main():
    root = tk.Tk()
    ttk.Style().theme_use("default")
    App(root)
    root.mainloop()

if __name__ == "__main__":
    main()
