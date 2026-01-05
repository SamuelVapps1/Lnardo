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
from PIL import Image, ImageFilter

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
    "High-end ecommerce product photo. "
    "Keep the exact same packaging design, text, logo, colors, and shape as the reference image. "
    "Centered front-facing packshot on pure white seamless background, soft natural shadow under product, "
    "neutral studio lighting, sharp focus, ultra realistic. "
    "No extra objects, no added stickers, no redesign."
)

DEFAULT_PIECE_PROMPT = (
    "High-end macro ecommerce photo of a single piece of the treat. "
    "Keep the exact shape, texture and color as the reference image. "
    "Pure white seamless background, soft shadow, realistic details, sharp focus. "
    "No extra pieces, no garnish, no stylization."
)

DEFAULT_NEGATIVE = (
    "cartoon, illustration, anime, 3d render, cgi, low quality, blurry, "
    "extra text, watermark, logo overlay, fake label, distorted packaging, "
    "cheese, dairy, yogurt, dessert, jam, sauce, fruit, garnish, plate, bowl, "
    "napkin, table setting, food styling, props, background objects"
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
    pack_strength: float = 0.22
    piece_strength: float = 0.35
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
        image_prompt_ids: Optional[List[str]] = None,
    ) -> Optional[Tuple[str, Optional[int]]]:
        model_id = MODEL_HQ if model_profile == "HQ" else MODEL_CHEAP
        payload: Dict[str, Any] = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "modelId": model_id,
            "width": int(width),
            "height": int(height),
            "num_images": int(num_images),
            "alchemy": bool(alchemy),
            "num_inference_steps": int(inference_steps),
        }

        # iba ak ideš image-to-image (máš init image)
        if init_image_id:
            if model_id in I2I_UNSUPPORTED_MODEL_IDS:
                mname = I2I_UNSUPPORTED_MODEL_IDS[model_id]
                raise ValueError(
                    f"Model '{mname}' (modelId={model_id}) nepodporuje image-to-image cez init_image_id. "
                    f"Nastav LEONARDO_MODEL_ID na SDXL model, napr. Leonardo Vision XL "
                    f"(5c232a9e-9061-4777-980a-ddc8e65647c6) alebo Leonardo Diffusion XL "
                    f"(1e60896f-3c26-4296-8ecc-53e2afecc132)."
                )
            payload["init_image_id"] = init_image_id
            payload["init_strength"] = float(init_strength)

        # Add imagePrompts if provided (for multi-reference guidance)
        if image_prompt_ids and len(image_prompt_ids) >= 1:
            payload["imagePrompts"] = image_prompt_ids

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
            # If imagePrompts causes 400, return None to signal retry without it
            if r.status_code == 400 and image_prompt_ids and ("imagePrompts" in txt.lower() or "image prompt" in txt.lower()):
                return None
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

def find_ref_images(directory: Path, sku: str, suffix: str) -> List[Path]:
    """
    Find all reference images for a SKU+suffix.
    Returns list where exact "{sku}_{suffix}.{ext}" is first if it exists,
    then includes "{sku}_{suffix}_*.{ext}" sorted by name.
    """
    results: List[Path] = []
    
    # First, find the primary ref (exact match)
    primary = find_ref_image(directory, sku, suffix)
    if primary:
        results.append(primary)
    
    # Then find all additional refs matching pattern {sku}_{suffix}_*.{ext}
    if directory.exists():
        pattern_base = f"{sku}_{suffix}_"
        additional: List[Path] = []
        
        for file_path in directory.iterdir():
            if not file_path.is_file():
                continue
            
            name = file_path.name
            name_lower = name.lower()
            
            # Check if it matches pattern and has allowed extension
            if name.startswith(pattern_base) and any(name_lower.endswith(ext) for ext in ALLOWED_EXTS):
                # Make sure it's not the primary (shouldn't happen, but be safe)
                if file_path != primary:
                    additional.append(file_path)
        
        # Sort additional refs by name
        additional.sort(key=lambda p: p.name)
        results.extend(additional)
    
    return results

def select_primary_ref_image(paths: List[Path]) -> Path:
    """
    Select the best primary reference image from a list.
    Prefers exact base filename if present, otherwise uses scoring (resolution + sharpness).
    """
    if not paths:
        raise ValueError("Empty paths list")
    
    if len(paths) == 1:
        return paths[0]
    
    # Check if first path is exact base filename (without _* suffix)
    first_path = paths[0]
    first_name = first_path.name.lower()
    # Check if it matches pattern {sku}_{suffix}.{ext} (no underscore after suffix)
    if '_' not in first_name.split('.')[0].split('_')[-1] or first_name.count('_') == 1:
        # Likely exact match, prefer it
        return first_path
    
    # Score all images
    scored: List[Tuple[Path, float]] = []
    
    for path in paths:
        try:
            with Image.open(path) as img:
                # Resolution score (w * h)
                resolution_score = img.width * img.height
                
                # Sharpness score (edge energy via FIND_EDGES filter)
                gray = img.convert('L')
                edges = gray.filter(ImageFilter.FIND_EDGES)
                # Mean pixel value as edge energy proxy
                pixels = list(edges.getdata())
                sharpness_score = statistics.mean(pixels) if pixels else 0
                
                # Combined score (normalize resolution to 0-1 range, then combine)
                # Assuming max reasonable resolution is 4K (3840*2160 = 8294400)
                norm_res = min(resolution_score / 8294400.0, 1.0)
                # Sharpness is already 0-255 range, normalize to 0-1
                norm_sharp = sharpness_score / 255.0
                
                # Weighted combination (favor resolution slightly)
                combined_score = (norm_res * 0.6) + (norm_sharp * 0.4)
                scored.append((path, combined_score))
        except Exception as e:
            # If image can't be opened, give it low score
            scored.append((path, 0.0))
    
    # Sort by score descending, return best
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[0][0]

def detect_extension_from_url(url: str) -> str:
    # heuristic; we download as .png by default
    lower = url.lower()
    for ext in [".png", ".jpg", ".jpeg", ".webp"]:
        if lower.endswith(ext):
            return ext if ext != ".jpeg" else ".jpg"
    return ".png"

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
        self.pack_strength_var = tk.DoubleVar(value=0.22)
        self.piece_strength_var = tk.DoubleVar(value=0.35)
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
                pack_refs = find_ref_images(s.pack_dir, sku, "pack")
                piece_refs = find_ref_images(s.piece_dir, sku, "piece")

                will_do_any = False

                if s.gen_pack:
                    if not pack_refs:
                        missing_pack += 1
                        self._log(f"[MISSING PACK] {sku}")
                    else:
                        self._log(f"[OK PACK] {sku} refs={len(pack_refs)}")
                if s.gen_piece:
                    if not piece_refs:
                        missing_piece += 1
                        self._log(f"[MISSING PIECE] {sku}")
                    else:
                        self._log(f"[OK PIECE] {sku} refs={len(piece_refs)}")

                if (s.gen_pack and pack_refs) or (s.gen_piece and piece_refs):
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
                pack_refs = find_ref_images(s.pack_dir, sku, "pack")
                piece_refs = find_ref_images(s.piece_dir, sku, "piece")

                out_sku_dir = s.output_dir / folder_name
                out_pack = out_sku_dir / f"{sku}__pack.png"
                out_piece = out_sku_dir / f"{sku}__piece.png"

                if s.gen_pack and pack_refs:
                    if not (s.skip_existing and out_pack.exists()):
                        planned += 1
                if s.gen_piece and piece_refs:
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

                pack_refs = find_ref_images(s.pack_dir, sku, "pack")
                piece_refs = find_ref_images(s.piece_dir, sku, "piece")

                out_sku_dir = s.output_dir / folder_name
                out_sku_dir.mkdir(parents=True, exist_ok=True)

                # Per-SKU manifest
                manifest_path = out_sku_dir / "manifest.csv"
                ensure_manifest(manifest_path)

                out_pack = out_sku_dir / f"{sku}__pack.png"
                out_piece = out_sku_dir / f"{sku}__piece.png"

                # If neither selected ref exists, skip SKU
                has_any = (s.gen_pack and pack_refs) or (s.gen_piece and piece_refs)
                if not has_any:
                    self._log(f"[SKIP] {sku} no usable refs for selected modes.")
                    continue

                # Log before generating each SKU
                current_model = MODEL_HQ if s.model_profile == "HQ" else MODEL_CHEAP
                self._log(f"Profile={s.model_profile} modelId={current_model} | {s.width}x{s.height} | steps={s.inference_steps} | alchemy={s.alchemy}")

                # --- PACK (optional) ---
                if s.gen_pack and pack_refs:
                    if not (s.skip_existing and out_pack.exists()):
                        # Select primary ref image
                        primary_pack_ref = select_primary_ref_image(pack_refs)
                        self.status_var.set(f"{sku}: uploading pack ref…")
                        self._log(f"[{sku}] Selected primary pack ref: {primary_pack_ref.name}")
                        pack_init_id = client.upload_init_image(primary_pack_ref)
                        
                        # Build prompt with product context
                        pack_prompt_text = f"{name}. {s.pack_prompt}" if name else s.pack_prompt

                        self.status_var.set(f"{sku}: generating PACK…")
                        gen_id, cost = client.create_generation(
                            prompt=pack_prompt_text,
                            negative_prompt=s.negative_prompt,
                            width=s.width,
                            height=s.height,
                            init_image_id=pack_init_id,
                            init_strength=s.pack_strength,
                            alchemy=s.alchemy,
                            num_images=s.pack_num_images,
                            inference_steps=s.inference_steps,
                            model_profile=s.model_profile,
                            image_prompt_ids=None,
                        )
                        
                        self._log(f"[{sku}] PACK gen_id={gen_id} cost={cost}")

                        urls = client.wait_for_urls(gen_id, poll_s=s.poll_s, timeout_s=s.timeout_s)
                        client.download(urls[0], out_pack)
                        
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
                            "init_strength": s.pack_strength,
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
                if s.gen_piece and piece_refs:
                    if not (s.skip_existing and out_piece.exists()):
                        # Select primary ref image
                        primary_piece_ref = select_primary_ref_image(piece_refs)
                        self.status_var.set(f"{sku}: uploading piece ref…")
                        self._log(f"[{sku}] Selected primary piece ref: {primary_piece_ref.name}")
                        piece_init_id = client.upload_init_image(primary_piece_ref)
                        
                        # Build prompt with product context
                        piece_prompt_text = f"{name}. High-end macro ecommerce photo of a single piece of this dog treat. {s.piece_prompt}" if name else s.piece_prompt

                        self.status_var.set(f"{sku}: generating PIECE…")
                        gen_id2, cost2 = client.create_generation(
                            prompt=piece_prompt_text,
                            negative_prompt=s.negative_prompt,
                            width=s.width,
                            height=s.height,
                            init_image_id=piece_init_id,
                            init_strength=s.piece_strength,
                            alchemy=s.alchemy,
                            num_images=s.piece_num_images,
                            inference_steps=s.inference_steps,
                            model_profile=s.model_profile,
                            image_prompt_ids=None,
                        )
                        
                        self._log(f"[{sku}] PIECE gen_id={gen_id2} cost={cost2}")

                        urls2 = client.wait_for_urls(gen_id2, poll_s=s.poll_s, timeout_s=s.timeout_s)
                        client.download(urls2[0], out_piece)
                        
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
                            "init_strength": s.piece_strength,
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
                            s.alchemy, s.piece_strength,
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
