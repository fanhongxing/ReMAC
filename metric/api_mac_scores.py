"""
Usage Instructions:

1. Set Environment Variables:
   # For SiliconFlow
   export SILICONFLOW_API_KEY="your_api_key"
   export SILICONFLOW_BASE_URL="https://api.siliconflow.cn/v1"
   
   # For OpenRouter
   export OPENROUTER_API_KEY="your_api_key"

2. Run the script:
   # SiliconFlow
   python metric/api_mac_scores.py \\
       --provider siliconflow \\
       --model "Qwen/Qwen3-VL-32B-Instruct" \\
       --root-dir data_cocoa \\
       --out-dir results_siliconflow

   # OpenRouter
   python metric/api_mac_scores.py \\
       --provider openrouter \\
       --model "Qwen/Qwen3-VL-32B-Instruct" \\
       --root-dir data_cocoa \\
       --out-dir results_openrouter

   Arguments:
   --provider: 'siliconflow' or 'openrouter' (default: siliconflow)
   --root-dir: Root directory containing 'image' subdir (originals) and model output subdirs.
   --out-dir: Directory to save results.
   --model: Model name to use for evaluation.
   --metrics: Metrics to evaluate (completeness, consistency).
"""
import os
import json
import base64
import io
import argparse
import concurrent.futures
import threading
from typing import Dict, List, Optional, Tuple

from openai import OpenAI
from tqdm import tqdm
from PIL import Image, ImageFile

# Allow loading truncated images to be more tolerant
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Global variables for client and provider
client: Optional[OpenAI] = None
PROVIDER = "siliconflow"

# Global lock for writing error logs
error_log_lock = threading.Lock()

# Debug config: whether to save images sent to API
DEBUG_SAVE_IMAGES = False
DEBUG_IMAGE_DIR = "debug_images"

# ================= Prompt Templates =================
prompt_template_dict = {
    # 1. Completeness determination
    "completeness": """You are an expert in visual perception and object recognition.

You will be given **two images**:
- The **first image** is the original image containing the scene.
- The **second image** is the **segmented result** of the main object, obtained from an **Amodal Completion** task.

Your task is to determine whether the segmented object (the second image) represents a **complete and intact** version of the object seen in the original image.

Definitions:

- **"Complete"** means that the object is entirely visible within the image frame, not partially cut off, hidden, or distorted. The segmented result should contain the object in its natural, full form as it appears in the real world.
- **"Incomplete"** means the object is missing parts, truncated at the edges, occluded, or not consistent with the full object that should exist in the original image.

**Important:**
- Focus on comparing the segmented result (second image) with the original image (first image).  
  The segmented object should correspond to the same object visible in the original image and should not miss essential parts.
- If the segmented object appears cut off, has missing limbs or edges, or is inconsistent with the object’s full structure in the original image, it should be classified as **Incomplete**.

Instructions:

1. Carefully compare the segmented object (second image) with the original image (first image).
2. Determine if the segmented object is **Complete** or **Incomplete**.
3. Provide your decision in this strict JSON format:

{
  "object_status": "Complete" | "Incomplete",
  "explanation": "A short sentence explaining why you made this decision, focusing on missing parts, truncation, or mismatch with the original image."
}

Note:
- Only use "Complete" or "Incomplete" as the categories.
- Focus on whether the segmented result accurately represents the complete form of the object seen in the original image.
""",
    # 2. Consistency scoring
    "consistency": """You are an evaluator comparing two images:

The original image, which contains the visible part of the object (partially occluded).
The completed image, which shows the object after amodal completion.

Amodal completion is the process of inferring and representing an object’s occluded parts so the object is understood as a complete, closed whole.

Your task is to rate how consistent and realistic the completed object itself is.

Critical Definition of "Incomplete":
If the completed object still looks cut off, truncated, or has a straight "image border" edge where it should be round or continuous, it is considered Incomplete. This is a structural failure.

Evaluation Dimensions:

1. Structural Continuity (0–4 points)
Focus on the closure and logical continuation of the shape.

0: The object boundary is abruptly cut off, forming a straight line or unnatural truncation (looks like the original occluded input). The shape is NOT closed.
1: The object attempts to close the shape but the boundary is severely distorted, jagged, or structurally impossible.
2: Generally continuous but with noticeable misalignment or irregularities in the completed region.
3: Contours flow seamlessly and align well between completed and visible regions.
4: Boundaries are perfectly closed, continuous, and fully consistent with the visible parts.
2. Semantic Consistency (0–4 points)

0: The completed region introduces incorrect or unrelated elements.
1: Roughly matches the object but contains major semantic errors (e.g., wrong parts or unrealistic details).
2: Generally consistent but with notable smaller semantic inaccuracies.
3: Mostly consistent, with only very minor or negligible semantic differences.
4: Perfect match to the original object’s type, structure, and expected real-world form.
3. Object Realism (0–2 points)

0: The completed object does not resemble a plausible real-world version of the object (e.g., a half-object is not realistic).
1: Somewhat realistic but with small inconsistencies.
2: Perfectly realistic and faithful to how this object should appear in reality.
Scoring:
Add up the points from all categories.
score = Structural Continuity + Semantic Consistency + Object Realism

Output Format:
{
"score": X,
"dimension_scores": {
"structural_continuity": Y,
"semantic_consistency": Z,
"object_realism": A
},
"explanation": "One or two sentences summarizing why you gave this score."
}
"""
}

# ================== Utility Functions ==================
def setup_client(provider: str):
    global client, PROVIDER
    PROVIDER = provider.lower()
    
    if PROVIDER == "siliconflow":
        api_key = os.getenv("SILICONFLOW_API_KEY", "sk-...")
        base_url = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
    elif PROVIDER == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        base_url = "https://openrouter.ai/api/v1"
        if not api_key:
            print("Warning: OPENROUTER_API_KEY environment variable not set.")
    else:
        raise ValueError(f"Unknown provider: {provider}")

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

def encode_image_as_png(image_path: str) -> str:
    try:
        with Image.open(image_path) as im:
            im.load()
            
            # If RGBA, set background to white, keep foreground only
            if im.mode == "RGBA":
                background = Image.new("RGB", im.size, (255, 255, 255))
                background.paste(im, mask=im.split()[3])
                im = background
            elif im.mode != "RGB":
                im = im.convert("RGB")

            # Debug: Save processed image
            if DEBUG_SAVE_IMAGES:
                try:
                    os.makedirs(DEBUG_IMAGE_DIR, exist_ok=True)
                    # Construct unique filename: parent_dirname_filename
                    parent = os.path.basename(os.path.dirname(image_path))
                    name = os.path.basename(image_path)
                    save_path = os.path.join(DEBUG_IMAGE_DIR, f"{parent}_{name}")
                    im.save(save_path)
                except Exception as e:
                    print(f"Warning: Failed to save debug image {image_path}: {e}")

            buf = io.BytesIO()
            im.save(buf, format="PNG", optimize=True)
            return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")


def api_response(prompt: str, image_paths: list, model="Qwen/Qwen3-VL-32B-Instruct", temperature: float = 0.0) -> str:
    if client is None:
        raise RuntimeError("Client not initialized. Call setup_client() first.")

    content = [{"type": "text", "text": prompt}]
    for path in image_paths:
        b64_png = encode_image_as_png(path)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64_png}"}
        })
    
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 1000,
        "temperature": temperature,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "seed": 42
    }

    # Provider-specific configurations
    if PROVIDER == "openrouter":
        kwargs["top_p"] = 1.0
        kwargs["extra_headers"] = {
            "HTTP-Referer": "https://github.com/fanhongxing/Multi-Agent",
            "X-Title": "Multi-Agent Evaluation"
        }
    else:
        # SiliconFlow default
        kwargs["top_p"] = 0

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


def try_parse_json(text: str) -> Optional[dict]:
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    stripped = text.strip()
    if stripped.startswith("```"):
        parts = stripped.split("```")
        if len(parts) >= 3:
            candidate = parts[1] if len(parts) == 3 else parts[2]
            if candidate.lstrip().lower().startswith("json"):
                candidate = candidate.lstrip()[4:].lstrip("\n")
            try:
                return json.loads(candidate)
            except Exception:
                pass
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            return json.loads(text[start:end + 1])
    except Exception:
        pass
    return None


def append_jsonl(path: str, obj: dict) -> None:
    with error_log_lock:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def list_images_map(dir_path: str, exts: Tuple[str, ...]) -> Dict[str, str]:
    result = {}
    if not os.path.isdir(dir_path):
        return result
    for name in os.listdir(dir_path):
        if name.lower().endswith(exts):
            result[os.path.splitext(name)[0]] = os.path.join(dir_path, name)
    return result


def extract_object_name(filename: str) -> str:
    base = os.path.splitext(os.path.basename(filename))[0]
    if "_" in base:
        return base.split("_")[-1].strip().lower()
    return ""


def process_single_stem(
    stem: str,
    original_img: str,
    model_names: Tuple[str, ...],
    model_maps: Dict[str, Dict[str, str]],
    out_dir: str,
    errors_log: str,
    model_api_name: str,
    metrics: List[str]
) -> None:
    object_name = extract_object_name(original_img)

    # Aggregate available candidates for current sample (files exist)
    candidates: List[Tuple[str, str]] = []
    for mn in model_names:
        mapping = model_maps.get(mn, {})
        if stem in mapping:
            candidates.append((mn, mapping[stem]))

    out_path = os.path.join(out_dir, f"{stem}.json")

    # Try loading existing results
    per_image = None
    if os.path.exists(out_path):
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                per_image = json.load(f)
        except Exception:
            pass

    # If no existing results, initialize a new one
    if per_image is None:
        per_image = {
            "stem": stem,
            "object_name": object_name,
            "original": original_img,
            "candidates": {},
            "evaluations": {},
        }

    # Ensure basic structure exists
    if "candidates" not in per_image: per_image["candidates"] = {}
    if "evaluations" not in per_image: per_image["evaluations"] = {}
    
    # Update candidates path info
    for n, p in candidates:
        per_image["candidates"][n] = p

    updated = False

    # === Dynamically construct Prompt with object_name ===
    prompt_prefix = f"Target object is {object_name}.\n\n" if object_name else ""

    for metric in metrics:
        if metric not in prompt_template_dict:
            continue
            
        if metric not in per_image["evaluations"]:
            per_image["evaluations"][metric] = {}
            
        final_prompt = prompt_prefix + prompt_template_dict[metric]

        for model_name, cand_path in candidates:
            # Check if already evaluated
            has_metric = model_name in per_image["evaluations"][metric]

            if has_metric:
                continue

            updated = True
            
            try:
                resp_text = api_response(final_prompt, [original_img, cand_path], model=model_api_name)
                resp_json = try_parse_json(resp_text)
                per_image["evaluations"][metric][model_name] = {"raw": resp_text, "parsed": resp_json}
            except Exception as e:
                append_jsonl(errors_log, {"stem": stem, "stage": metric, "error": str(e)})

    if updated or not os.path.exists(out_path):
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(per_image, f, ensure_ascii=False, indent=2)


# ================== Main Function ==================
def run_batch(
    root_dir: str = "data",
    out_dir: str = "results",
    model_names: Optional[Tuple[str, ...]] = None,
    image_subdir: str = "image",
    exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png"),
    stems_filter: Optional[List[str]] = None,
    limit: int = 0,
    max_workers: int = 4,
    model_api_name: str = "Qwen/Qwen3-VL-32B-Instruct",
    metrics: List[str] = ["completeness", "consistency"],
    provider: str = "siliconflow"
) -> None:
    
    # Initialize client based on provider
    setup_client(provider)

    def _discover_models(root: str) -> List[str]:
        names: List[str] = []
        try:
            for n in sorted(os.listdir(root)):
                p = os.path.join(root, n)
                if n == image_subdir:
                    continue
                if os.path.isdir(p):
                    # Consider as candidate directory only if it contains at least 1 image
                    m = list_images_map(p, exts)
                    if len(m) > 0:
                        names.append(n)
        except Exception:
            pass
        return names

    image_dir = os.path.join(root_dir, image_subdir)
    originals = list_images_map(image_dir, exts)

    # Candidate model directories: use provided model_names if available; otherwise auto-discover
    if model_names is None or len(model_names) == 0:
        model_names = tuple(_discover_models(root_dir))  # type: ignore[assignment]

    # Build image mapping for each candidate model
    model_maps: Dict[str, Dict[str, str]] = {}
    for mn in model_names:
        model_maps[mn] = list_images_map(os.path.join(root_dir, mn), exts)

    stems = sorted(originals.keys())
    # Optional: Filter samples by name
    if stems_filter:
        choose = set(s.strip() for s in stems_filter if isinstance(s, str) and s.strip())
        stems = [s for s in stems if s in choose]
    # Optional: Limit number of samples
    if isinstance(limit, int) and limit > 0:
        stems = stems[:limit]
    print(f"🔎 Found {len(stems)} original images.")
    if not stems:
        print(f"⚠️ No original images found in '{image_dir}'. Please check root_dir and subdirectory name (current image_subdir='{image_subdir}').")
    os.makedirs(out_dir, exist_ok=True)
    errors_log = os.path.join(out_dir, "errors.jsonl")

    # Use ThreadPoolExecutor for multi-threaded processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for stem in stems:
            original_img = originals[stem]
            futures.append(
                executor.submit(
                    process_single_stem,
                    stem,
                    original_img,
                    model_names,
                    model_maps,
                    out_dir,
                    errors_log,
                    model_api_name,
                    metrics
                )
            )
        
        # Use tqdm to show progress
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing"):
            pass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch evaluate amodal completion results with SiliconFlow or OpenRouter")
    parser.add_argument("--root-dir", type=str, default="data_cocoa", help="Data root directory, must contain original image subdirectory (default 'image') and candidate model output subdirectories")
    parser.add_argument("--out-dir", type=str, default=None, help="Evaluation output directory; defaults to 'result_overall' in the script directory")
    parser.add_argument("--image-subdir", type=str, default="image", help="Subdirectory name for original images (default 'image')")
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated list of candidate model subdirectory names; auto-discovered if not provided (excluding original image directory)",
    )
    parser.add_argument(
        "--stems",
        type=str,
        default="",
        help="Optional: Comma-separated sample filenames (without extension); if provided, only these samples are evaluated",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional: Maximum number of samples to evaluate (>0 to enable)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=6,
        help="Maximum number of threads (default 6)",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Specify model name to use (e.g., Qwen/Qwen3-VL-32B-Instruct)",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="completeness,consistency",
        help="Evaluation metrics to run, comma-separated (default: completeness,consistency)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="siliconflow",
        choices=["siliconflow", "openrouter"],
        help="API Provider to use: 'siliconflow' or 'openrouter' (default: siliconflow)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    # If --out-dir is not explicitly specified, use 'result_overall' in the script directory
    if args.out_dir is None or not args.out_dir.strip():
        args.out_dir = os.path.join(SCRIPT_DIR, "result_overall")
    models_tuple: Optional[Tuple[str, ...]] = None
    if args.models:
        models_tuple = tuple([s.strip() for s in args.models.split(",") if s.strip()])
    stems_list: Optional[List[str]] = None
    if args.stems:
        stems_list = [s.strip() for s in args.stems.split(",") if s.strip()]
    
    metrics_list = [m.strip() for m in args.metrics.split(",") if m.strip()]

    run_batch(
        root_dir=args.root_dir,
        out_dir=args.out_dir,
        model_names=models_tuple,
        image_subdir=args.image_subdir,
        stems_filter=stems_list,
        limit=args.limit,
        max_workers=args.max_workers,
        model_api_name=args.model,
        metrics=metrics_list,
        provider=args.provider
    )

    # After evaluation, output final result summary
    try:
        def analyze_results(results_dir: str, output_path: str) -> dict:
            accum: Dict[str, dict] = {}

            def ensure_method(method: str) -> None:
                if method not in accum:
                    accum[method] = {
                        "completeness": {"complete": 0, "total": 0},
                        "consistency": {"sum": 0.0, "count": 0},
                    }

            def normalize_status(value: str) -> Optional[str]:
                if not isinstance(value, str):
                    return None
                v = value.strip().lower()
                if v in {"complete", "completed"}:
                    return "Complete"
                if v in {"incomplete", "not complete", "in-complete"}:
                    return "Incomplete"
                return None

            def extract_completeness_status(entry: dict) -> Optional[str]:
                parsed = entry.get("parsed")
                raw = entry.get("raw")
                data = parsed if isinstance(parsed, dict) else try_parse_json(raw) if isinstance(raw, str) else None
                if not isinstance(data, dict):
                    return None
                for key in ("object_status", "status", "result"):
                    if key in data:
                        norm = normalize_status(data[key])
                        if norm:
                            return norm
                return None

            def extract_consistency_score(entry: dict) -> Optional[float]:
                parsed = entry.get("parsed")
                raw = entry.get("raw")
                data = parsed if isinstance(parsed, dict) else try_parse_json(raw) if isinstance(raw, str) else None
                if not isinstance(data, dict):
                    return None
                score = data.get("score")
                try:
                    if score is not None:
                        return float(score)
                except Exception:
                    return None
                return None

            if not os.path.isdir(results_dir):
                raise FileNotFoundError(f"Results directory not found: {results_dir}")

            files = [f for f in os.listdir(results_dir) if f.lower().endswith(".json")]
            files.sort()

            for name in files:
                path = os.path.join(results_dir, name)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        item = json.load(f)
                except Exception:
                    continue

                evaluations = item.get("evaluations") or {}
                
                # Process Completeness
                comp_all = evaluations.get("completeness") or {}
                for method, entry in comp_all.items():
                    ensure_method(method)
                    status = extract_completeness_status(entry if isinstance(entry, dict) else {})
                    if status is not None:
                        accum[method]["completeness"]["total"] += 1
                        if status == "Complete":
                            accum[method]["completeness"]["complete"] += 1
                
                # Process Consistency
                cons_all = evaluations.get("consistency") or {}
                for method, entry in cons_all.items():
                    ensure_method(method)
                    score = extract_consistency_score(entry if isinstance(entry, dict) else {})
                    if isinstance(score, (int, float)):
                        accum[method]["consistency"]["sum"] += float(score)
                        accum[method]["consistency"]["count"] += 1

            summary: Dict[str, dict] = {}
            for method, stats in accum.items():
                # Completeness stats
                c_complete = stats["completeness"]["complete"]
                c_total = stats["completeness"]["total"]
                c_rate = (c_complete / c_total) if c_total > 0 else None
                
                # Consistency stats
                s_sum = stats["consistency"]["sum"]
                s_cnt = stats["consistency"]["count"]
                s_avg = (s_sum / s_cnt) if s_cnt > 0 else None

                summary[method] = {
                    "completeness": {
                        "complete": c_complete,
                        "total": c_total,
                        "rate": c_rate,
                    },
                    "consistency": {
                        "sum": s_sum,
                        "count": s_cnt,
                        "avg": s_avg,
                    },
                }

            result = {
                "results_dir": results_dir,
                "methods": sorted(summary.keys()),
                "summary": summary,
            }

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            print("✅ Analysis completed, results saved to:", output_path)
            for method in sorted(summary.keys()):
                s = summary[method]
                c_rate = s["completeness"]["rate"]
                s_avg = s["consistency"]["avg"]
                print(
                    f"- {method}: Completeness={c_rate if c_rate is not None else 'N/A'}, Consistency={s_avg if s_avg is not None else 'N/A'}"
                )
            return result

        # Summary file placed inside results directory, named summary.json
        analyze_results(args.out_dir, os.path.join(args.out_dir, "summary.json"))
    except Exception as e:
        print(f"⚠️ Failed to analyze results: {e}")
