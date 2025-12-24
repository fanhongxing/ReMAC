import os
import json
import base64
import tempfile
from typing import Optional, List
import requests

# Optional Gemini SDK import is deferred in methods to avoid hard dependency at import time


def _encode_image_to_data_url(image_path: str) -> str:
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    # default use jpeg data url
    return f"data:image/jpeg;base64,{b64}"


# Inline prompt templates from the provided file (at least the one we use)
PROMPT_TEMPLATES = {
    "prompt": (
        "Given an image of {} with partial occlusion, generate a concise one-sentence inpainting prompt "
        "that directly describes the {} overall shape, color, and texture for restoration, without mentioning "
        "any occlusion details or background information."
    ),
    "target_segmentation_check": (
        "You will be given TWO images: (1) the original scene, and (2) a white-background cutout that is supposed to be the currently segmented VISIBLE PART of the target object: '{target_object}'.\n\n"
        "Task: Decide if the target object has been successfully segmented OUT at all.\n\n"
        "Success criteria (binary):\n"
        "- The cutout clearly contains the target object's visible region in the correct location/shape (not blank or totally unrelated).\n"
        "- It's okay if the cutout is incomplete (occluded) or has rough edges.\n\n"
        "Output STRICTLY as JSON with a single boolean field (no extra text):\n"
        "{\"segmented\": true}  or  {\"segmented\": false}"
    ),
    # other templates from the file can be added if needed:
    "occluding_object": (
        'Given an image, perform the following analysis for the instance of {}: Identify Occlusions: Examine {} and determine if any portions of it are occluded by other objects in the image. If occlusions exist, list the names of the objects that are directly blocking {}. Fixed Format Output: Present the results using the following fixed format (for each {} analyzed): [<comma-separated list of names of objects occluding {}>] Ensure that your output strictly adheres to the fixed format provided above.'
    ),
    "boundary": (
    'Review the image to determine whether the entire {} is visible within its boundaries. If any part of the {} is cut off, specify the affected edge(s) (top, bottom, left, or right). For each impacted edge, provide an estimate of the missing portion as a relative fraction of the overall dimension (for example, "extends upward by 0.25 of the image height"), and indicate the necessary expansion. Output your analysis as a JSON object following the structure below. Ensure that the "extension_amount" values are expressed as numeric relative proportions (e.g., 0.25 for one quarter), not as strings: {{"is_occluded_object_outside": <boolean>, "extension_direction":  [""], "extension_amount": number}}'
    ),
    "boundary_bbox": (
    "Review the image to determine whether the entire {} is visible within its boundaries.\n\n"
    "Preliminary Analysis:\n"
    "-------------------------------------------------\n"
    "{{\n"
    "  \"exceeded_edges\": [<list of edges that have been detected as exceeded>]\n"
    "}}\n"
    "-------------------------------------------------\n\n"
    "Note: The preliminary analysis above indicates which edge(s) the object's bounding box has exceeded. "
    "For any edge listed in \"exceeded_edges\", it is confirmed that the object extends beyond the image boundary on that side. "
    "For other cases—such as when the object is partially occluded by overlapping elements—rely solely on your visual analysis.\n\n"
    "If any part of the {} is cut off, specify the affected edge(s) (top, bottom, left, or right).\n"
    "Do NOT try to compute an exact distance. Instead, use coarse, visual judgment to place the required expansion for the object into one of three levels:\n"
    "- \"slight\" expansion: the object only slightly exceeds the boundary; extending by about 20% of the image size in that direction is sufficient.\n"
    "- \"moderate\" expansion: a noticeable portion is missing; extending by about 50% of the image size in that direction is appropriate.\n"
    "- \"large\" expansion: a large portion is missing; extending by about 70% of the image size in that direction is required.\n\n"
    "Map these three levels to numeric extension values as follows:\n"
    "- slight  -> 0.2\n"
    "- moderate -> 0.5\n"
    "- large   -> 0.7\n\n"
    "If the object is fully contained and no expansion is needed, set extension_direction to [] and extension_amount to 0.\n\n"
    "When multiple edges are affected, include all of them in \"extension_direction\". "
    "Set \"extension_amount\" to the single numeric value corresponding to the largest required expansion among the affected edges (or 0 if none).\n\n"
    "Output your analysis as a JSON object following the structure below. Ensure that the \"extension_amount\" value is expressed as a numeric "
    "relative proportion (0, 0.2, 0.5, or 0.7), not as a string:\n\n"
    "{{\"is_occluded_object_outside\": <boolean>, \"extension_direction\": [list of affected edges], \"extension_amount\": <number: 0 | 0.2 | 0.5 | 0.7>}}"
    ),
    # New: probabilistic reasoning to produce multiple hypotheses with probabilities summing to 1.0
    "probabilistic_hypotheses": (
        "You are an expert visual reasoning agent. Your task is to analyze the partially occluded '{object_name}' in an image and generate a list of plausible hypotheses for its complete appearance in a structured JSON format. Without mentioning any occlusion details or background information."
        "Each hypothesis should include a detailed description suitable for an inpainting model and a corresponding probability score. The probabilities must sum to 1.0."

        "For example, if the input is an image of a cat partially hidden behind a sofa, your output should look like this:"

        "```json"
        "{"
        '  "hypotheses": ['
        '    {'
        '      "description": "A full ginger tabby cat with green eyes and white paws, curled up comfortably and sleeping.",'
        '      "probability": 0.70'
        '    },'
        '    {'
        '      "description": "A ginger tabby cat sitting upright and looking towards the viewer, with its tail wrapped neatly around its front paws.",'
        '      "probability": 0.25'
        '    },'
        '    {'
        '      "description": "A long-bodied ginger tabby cat stretching forward with its front paws extended.",'
        '      "probability": 0.05'
        '    }'
        '  ]'
        "}"
        "```"

        "Now, analyze the new image provided and generate the JSON output for the '{object_name}'."
    ),
    # New: Missed occlusion checking using two images (original + mask/overlay)
    "missed_occlusion_check": (
        """You are a meticulous visual analyst specializing in spatial reasoning and occlusion detection. Your task is to act as a final verifier.

You will be given A SINGLE processed image in which:
1. The visible region of the target object '{target_object}' is shown (possibly brightened slightly).
2. All currently identified occluders have been replaced with PURE WHITE (#FFFFFF) so those areas no longer reveal their texture.

Your goal is to identify any **missed occluders** by strictly adhering to the following definition and procedure.

---
### **Definition of a "Missed Occluder"**

A "Missed Occluder" is an object that meets **ALL THREE** of the following criteria:
1.  **Spatial Position:** The object must be unambiguously **in front of** the '{target_object}' based on visual cues (e.g., overlap, perspective).
2.  **Visual Obstruction:** The object must physically **cover pixels** that should logically be part of the '{target_object}', causing an unnatural or incomplete boundary.
3.  **Novelty:** The object must **NOT** be one of the occluders already marked by a colored area in the segmentation mask.

---
### **CRITICAL EXCLUSION RULES**

You must **EXCLUDE** the following from your list, as they are common errors:
* **Background Objects:** Any object, no matter how close it appears, that is located behind the '{target_object}'.
* **Adjacent Objects:** Any object that is merely touching or next to the '{target_object}' without clearly overlapping it from the front.
* **Surface Phenomena:** Any shadows, reflections, water, water spots, or markings that are on the surface of the '{target_object}'.
* **Self-Occlusion:** Any part of the '{target_object}' itself (e.g., a person's own arm crossing their body).
* **Environmental Substrates & Particulates:** Ground/background elements (snow, sand, grass, water, sky, walls) and any kicked-up/falling material (spray, flakes, splashes, droplets, dust) are **not occluders**—**unless** they form a separate, well-bounded object (e.g., a snowball).
* **Insignificant Occluders:** Exclude occluders that are extremely small, thin, or trivial relative to the overall size of the '{target_object}'. If the occlusion is so minor that it does not significantly alter the target's perceived shape, it should be ignored.
---
### **Procedure: Step-by-Step Reasoning (Mandatory)**

Before providing your final output, you must first articulate your reasoning process. Follow these steps:
1.  **Candidate Identification:** Briefly list all potential candidate objects near the '{target_object}' that are not already marked as occluders.
2.  **Filtering Analysis:** For each candidate, methodically check it against the 3 criteria in the definition and the 6 critical exclusion rules. State explicitly whether each candidate **PASSES** or **FAILS** and provide a one-sentence justification.
3.  **Final List Compilation:** Only the candidates that **PASS** all checks should be included in your final list.

---
### **Fixed Format Output**

Present your full analysis in the following format. If your analysis finds no valid missed occluders, the final list must be empty `[]`.

**Reasoning:**
* **Candidate 1: [Name of candidate object]**
    * Analysis: [Your analysis of the candidate against the rules].
    * Result: PASS/FAIL
* **Candidate 2: [Name of candidate object]**
    * Analysis: [Your analysis of the candidate against the rules].
    * Result: PASS/FAIL
* ...

**Final Answer:**
[<comma-separated list of names of missed objects that PASSED the analysis>]"""
    ),
}


class GPTAdapter:
    """Self-contained adapter that builds prompts and (optionally) calls OpenRouter API without importing external files."""

    def __init__(
        self,
        backend: Optional[str] = None,
        azure_api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        azure_api_version: Optional[str] = None,
        azure_deployment: Optional[str] = None,
    ):
        # backends: 'openrouter' (default) | 'azure'
        self.backend = (backend or os.getenv("GPT_BACKEND", "openrouter")).lower()
        # OpenRouter settings
        self.or_base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        self.or_api_key = os.getenv("OPENROUTER_API_KEY", "")
        
        # SiliconFlow settings (for Qwen models)
        self.sf_base_url = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
        self.sf_api_key = os.getenv("SILICONFLOW_API_KEY", "")

        # Default models (can be overridden via env):
        # General agents: OPENROUTER_MODEL_GENERAL or OPENROUTER_MODEL or fallback to GPT-4o
        self.model_name_general = os.getenv(
            "OPENROUTER_MODEL_GENERAL",
            os.getenv("OPENROUTER_MODEL", "openai/gpt-4o"),
        )
        # Checker agent: OPENROUTER_MODEL_CHECK or fallback to google/gemini-2.5-pro
        self.model_name_check = os.getenv("OPENROUTER_MODEL_CHECK", "google/gemini-2.5-pro")

        # Azure OpenAI settings
        self.azure_api_key = azure_api_key or os.getenv("AZURE_OPENAI_API_KEY", "")
        self.azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT", "")
        self.azure_api_version = azure_api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")
        # For Azure, you must pass the deployment name (not the model id). Allow override via env.
        # Fallback to a generic name if not provided.
        self.azure_deployment = azure_deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT", os.getenv("AZURE_OPENAI_MODEL", "gpt-4o"))

        # Gemini settings (official Google Generative AI SDK)
        # If model_name_* indicates gemini-2.5-pro, prefer Gemini SDK by default when GEMINI_API_KEY is present
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
        # Optional explicit overrides
        self.gemini_model_general = os.getenv("GEMINI_MODEL_GENERAL", "")
        self.gemini_model_check = os.getenv("GEMINI_MODEL_CHECK", "")

    def _get_provider_config(self, model_name: str):
        """Return (base_url, api_key) based on model name."""
        # Always use OpenRouter as requested
        return self.or_base_url, self.or_api_key

    # --- helpers for boundary_bbox ---
    @staticmethod
    def get_exceeded_edges(bbox_json_path: Optional[str], occluded_object: str, bbox_data: Optional[dict] = None) -> List[str]:
        """Read detections json (or use provided dict) and check which edges are exceeded for `<occluded_object>_0`."""
        data = {}
        if bbox_data:
            data = bbox_data
        elif bbox_json_path:
            try:
                with open(bbox_json_path, "r") as f:
                    data = json.load(f)
            except Exception:
                return []
        else:
            return []

        key = f"{occluded_object}_0"
        if key not in data:
            return []
        try:
            bbox = data[key]["bounding_box_xyxy"]
            image_width = data[key]["image_width"]
            image_height = data[key]["image_height"]
        except Exception:
            return []
        thr = 10
        exceeded = []
        if bbox[0] <= thr:
            exceeded.append("left")
        if bbox[1] <= thr:
            exceeded.append("top")
        if bbox[2] >= (image_width - thr):
            exceeded.append("right")
        if bbox[3] >= (image_height - thr):
            exceeded.append("bottom")
        return exceeded

    # --- helpers for Gemini routing ---
    def _is_gemini_model(self, name: Optional[str]) -> bool:
        if not name:
            return False
        return "gemini" in str(name).lower()

    def _normalize_gemini_model(self, name: Optional[str]) -> str:
        """Map various identifiers (e.g., 'google/gemini-2.5-pro') to official Gemini model ids.

        Preference order for check-model override:
        - Explicit env overrides (GEMINI_MODEL_CHECK / GEMINI_MODEL_GENERAL)
        - Parsed from provided `name`
        - Default to 'gemini-2.5-pro' for best reasoning, else fallback 'gemini-1.5-pro'
        """
        # Env explicit overrides
        if name and "check" in str(name).lower() and self.gemini_model_check:
            return self.gemini_model_check
        if self.gemini_model_general and (not name or not self._is_gemini_model(name)):
            return self.gemini_model_general

        raw = (name or "").strip()
        # Common forms: 'google/gemini-2.5-pro', 'gemini-2.5-pro'
        if "/" in raw:
            raw = raw.split("/")[-1]
        # Minimal validation
        if raw.startswith("gemini-"):
            return raw
        # Sensible default
        return "gemini-2.5-pro"

    def _gemini_load_image_parts(self, image_paths: List[str]) -> List[object]:
        parts: List[object] = []
        # Try PIL first for best compatibility
        try:
            from PIL import Image  # type: ignore
            for p in image_paths:
                try:
                    img = Image.open(p)
                    parts.append(img)
                except Exception:
                    # Fallback to raw bytes with mime hint
                    try:
                        with open(p, "rb") as f:
                            data = f.read()
                        parts.append({"mime_type": "image/jpeg", "data": data})
                    except Exception:
                        continue
        except Exception:
            # No PIL: use raw bytes
            for p in image_paths:
                try:
                    with open(p, "rb") as f:
                        data = f.read()
                    parts.append({"mime_type": "image/jpeg", "data": data})
                except Exception:
                    continue
        return parts

    def _gemini_chat_with_images(self, image_paths: List[str], system_prompt: str, model_override: Optional[str] = None) -> Optional[str]:
        if not self.gemini_api_key:
            return None
        try:
            import google.generativeai as genai  # type: ignore
        except Exception:
            return None
        try:
            genai.configure(api_key=self.gemini_api_key)
            model_id = self._normalize_gemini_model(model_override)
            model = genai.GenerativeModel(model_id)
            parts = [system_prompt]
            parts.extend(self._gemini_load_image_parts(image_paths))
            resp = model.generate_content(
                parts,
                generation_config={"temperature": 0.2, "top_p": 1},
                safety_settings=None,
                request_options={"timeout": 120},
            )
            try:
                return getattr(resp, "text", None) or (resp.candidates[0].content.parts[0].text if resp.candidates else None)  # type: ignore
            except Exception:
                return None
        except Exception:
            return None

    def _gemini_chat_with_image(self, image_path: str, system_prompt: str, model_override: Optional[str] = None) -> Optional[str]:
        return self._gemini_chat_with_images([image_path], system_prompt, model_override=model_override)

    def _build_prompt(self, prompt_type: str, seg_text: str, bbox_json_path: Optional[str] = None, occluded_object: Optional[str] = None, bbox_data: Optional[dict] = None) -> str:
        template = PROMPT_TEMPLATES.get(prompt_type) or PROMPT_TEMPLATES["prompt"]
        # Support both positional {} and named {object_name} placeholders (do not break literal braces in examples)
        prompt_text = template
        if "{object_name}" in prompt_text:
            prompt_text = prompt_text.replace("{object_name}", seg_text)
        else:
            # substitute {} with seg_text across occurrences
            count = prompt_text.count("{}")
            if count:
                prompt_text = prompt_text.format(*([seg_text] * count))
        # boundary_bbox needs exceeded_edges injection
        if prompt_type == "boundary_bbox":
            edges: List[str] = []
            if (bbox_json_path or bbox_data) and occluded_object:
                edges = self.get_exceeded_edges(bbox_json_path, occluded_object, bbox_data=bbox_data)
            prompt_text = prompt_text.replace("[<list of edges that have been detected as exceeded>]", json.dumps(edges))
        return prompt_text

    def _openrouter_chat_with_image(self, image_path: str, system_prompt: str, model_override: Optional[str] = None) -> Optional[str]:
        # Always use OpenRouter backend
        model = model_override or self.model_name_general
        base_url, api_key = self._get_provider_config(model)

        if not api_key:
            return None
        url = f"{base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        try:
            data_url = _encode_image_to_data_url(image_path)
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": system_prompt},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    },
                ],
            }
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            if resp.status_code != 200:
                return None
            obj = resp.json()
            return obj.get("choices", [{}])[0].get("message", {}).get("content")
        except Exception:
            return None

    # --- Azure backend ---
    def _azure_chat_with_image(self, image_path: str, system_prompt: str) -> Optional[str]:
        """Call Azure OpenAI Chat Completions with vision input.

        Priority: try official SDK (openai.AzureOpenAI). If unavailable, fallback to REST.
        Expects envs AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION, AZURE_OPENAI_DEPLOYMENT.
        """
        key = self.azure_api_key
        ep = (self.azure_endpoint or "").rstrip("/")
        ver = self.azure_api_version
        dep = self.azure_deployment
        if not key or not ep or not dep:
            return None

        # Build messages with data URL image
        try:
            data_url = _encode_image_to_data_url(image_path)
        except Exception:
            data_url = None
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    [
                        {"type": "text", "text": system_prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ]
                    if data_url
                    else system_prompt
                ),
            },
        ]

        # Try SDK first
        try:
            try:
                from openai import AzureOpenAI  # type: ignore
            except Exception:
                AzureOpenAI = None  # type: ignore
            if AzureOpenAI is not None:
                client = AzureOpenAI(api_key=key, azure_endpoint=ep, api_version=ver)
                resp = client.chat.completions.create(
                    model=dep,
                    messages=messages,
                    temperature=0.2,
                    top_p=1,
                )
                # SDK object -> text
                try:
                    return resp.choices[0].message.content  # type: ignore
                except Exception:
                    pass
        except Exception:
            # fall through to REST
            pass

        # REST fallback
        try:
            url = f"{ep}/openai/deployments/{dep}/chat/completions?api-version={ver}"
            headers = {"api-key": key, "Content-Type": "application/json"}
            payload = {
                "messages": messages,
                "temperature": 0.2,
                "top_p": 1,
            }
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            if resp.status_code != 200:
                return None
            data = resp.json()
            return data.get("choices", [{}])[0].get("message", {}).get("content")
        except Exception:
            return None

    def inpaint_prompt(self, seg_text: str) -> str:
        """Build prompt text strictly from the provided file if it exposes templates; otherwise minimal fallback."""
        template = PROMPT_TEMPLATES.get("prompt")
        count = template.count("{}")
        return template.format(*([seg_text] * count))

    def gen_inpaint_prompt_from_image(self, image, seg_text: str, prompt_type: str = "prompt", bbox_json_path: Optional[str] = None, occluded_object: Optional[str] = None, bbox_data: Optional[dict] = None) -> str:
        """Use the provided module's api_response with its exact prompt template, passing the image file path."""
        # Build prompt by template (purely from inline templates)
        prompt_text = self._build_prompt(prompt_type, seg_text, bbox_json_path=bbox_json_path, occluded_object=occluded_object, bbox_data=bbox_data)
        # Mock backend: return deterministic outputs without calling any external APIs
        if self.backend == "mock":
            if prompt_type == "probabilistic_hypotheses":
                obj = seg_text
                return (
                    "{\n"
                    "  \"hypotheses\": [\n"
                    "    {\n"
                    f"      \"description\": \"A complete {obj} with plausible continuation consistent with visible parts.\",\n"
                    "      \"probability\": 0.7\n"
                    "    },\n"
                    "    {\n"
                    f"      \"description\": \"An alternative {obj} pose consistent with the scene context.\",\n"
                    "      \"probability\": 0.3\n"
                    "    }\n"
                    "  ]\n"
                    "}"
                )
            # default mock: just return the single-sentence description
            return self.inpaint_prompt(seg_text)
        # Save image to temp file and call api_response
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tf:
            # Try saving using the object API; if not available, assume it's already bytes
            try:
                image.save(tf.name)
            except Exception:
                try:
                    tf.write(image)
                    tf.flush()
                except Exception:
                    # fall back to text-only prompt
                    return prompt_text
            # Dispatch by backend
            if self.backend == "azure":
                resp_text = self._azure_chat_with_image(tf.name, prompt_text)
            else:
                resp_text = self._openrouter_chat_with_image(tf.name, prompt_text, model_override=None)
            if resp_text:
                return resp_text
        # If cannot call api_response, return prompt text
        return prompt_text

    def generate_inpaint_description(self, seg_text: str, **kwargs) -> str:
        """Prefer calling a dedicated function in the external module that handles both prompt building and GPT call."""
        # For compatibility: just return the inline template prompt
        return self.inpaint_prompt(seg_text)

    # -------------- New multi-image helpers --------------
    def _openrouter_chat_with_images(self, image_paths: List[str], system_prompt: str, model_override: Optional[str] = None) -> Optional[str]:
        """Send multiple images to OpenRouter vision model in a single prompt.

        If `model_override` indicates Gemini and a GEMINI_API_KEY is present, route to Gemini SDK by default.
        """
        # Always use OpenRouter backend
        model = model_override or self.model_name_general
        base_url, api_key = self._get_provider_config(model)

        if not api_key:
            return None
        url = f"{base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        try:
            contents = [{"type": "text", "text": system_prompt}]
            for p in image_paths:
                data_url = _encode_image_to_data_url(p)
                contents.append({"type": "image_url", "image_url": {"url": data_url}})
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": contents},
                ],
            }
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            if resp.status_code != 200:
                return None
            obj = resp.json()
            return obj.get("choices", [{}])[0].get("message", {}).get("content")
        except Exception:
            return None

    def _azure_chat_with_images(self, image_paths: List[str], system_prompt: str) -> Optional[str]:
        """Send multiple images to Azure OpenAI Chat Completions with vision input."""
        key = self.azure_api_key
        ep = (self.azure_endpoint or "").rstrip("/")
        ver = self.azure_api_version
        dep = self.azure_deployment
        if not key or not ep or not dep:
            return None
        # Build messages with data URL images
        contents: List[dict] = [{"type": "text", "text": system_prompt}]
        for p in image_paths:
            try:
                data_url = _encode_image_to_data_url(p)
            except Exception:
                data_url = None
            if data_url:
                contents.append({"type": "image_url", "image_url": {"url": data_url}})
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": contents if contents else system_prompt},
        ]
        # Try SDK first
        try:
            try:
                from openai import AzureOpenAI  # type: ignore
            except Exception:
                AzureOpenAI = None  # type: ignore
            if AzureOpenAI is not None:
                client = AzureOpenAI(api_key=key, azure_endpoint=ep, api_version=ver)
                resp = client.chat.completions.create(
                    model=dep,
                    messages=messages,
                    temperature=0.2,
                    top_p=1,
                )
                try:
                    return resp.choices[0].message.content  # type: ignore
                except Exception:
                    pass
        except Exception:
            pass
        # REST fallback
        try:
            url = f"{ep}/openai/deployments/{dep}/chat/completions?api-version={ver}"
            headers = {"api-key": key, "Content-Type": "application/json"}
            payload = {
                "messages": messages,
                "temperature": 0.2,
                "top_p": 1,
            }
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            if resp.status_code != 200:
                return None
            data = resp.json()
            return data.get("choices", [{}])[0].get("message", {}).get("content")
        except Exception:
            return None

    def check_missed_occlusions(self, processed_single_image, target_object: str) -> str:
        """Identify missed occluders using ONE processed image (whitened known occluders, target visible region retained/brightened).

        Returns raw LLM text; fallback to "[]" if backend unavailable or error.
        Writes minimal status json for debugging.
        """
        template = PROMPT_TEMPLATES.get("missed_occlusion_check")
        prompt_text = template.replace("{target_object}", str(target_object)) if template else (
            f"Identify missed occluders still covering {target_object}; output [] if none."
        )
        
        model = self.model_name_check
        _, api_key = self._get_provider_config(model)
        has_key = bool(api_key) or bool(self.azure_api_key)

        status = {"attempted": False, "backend": self.backend, "api_key_present": has_key, "response": None, "error": None}
        # Mock backend shortcut
        if self.backend == "mock":
            status["attempted"] = True
            status["response"] = "[]"
            return "[]"
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as t1:
                try:
                    processed_single_image.save(t1.name)
                except Exception:
                    t1.write(processed_single_image)
                    t1.flush()
                status["attempted"] = True
                resp_text = None
                if api_key:
                    resp_text = self._openrouter_chat_with_images([t1.name], prompt_text, model_override=model)
                if not resp_text:
                    resp_text = self._azure_chat_with_images([t1.name], prompt_text)
                if resp_text:
                    status["response"] = resp_text[:300]
                    return resp_text
        except Exception as e:
            status["error"] = str(e)
        return "[]"

    def check_target_segmentation_success(self, orig_image, target_cut_image, target_object: str) -> bool:
        """Return True if agent thinks the target object has been segmented out at all.

        Inputs: original image and a white-background cutout of the currently segmented visible part.
        Output: boolean parsed from JSON {"segmented": bool}. Fallback to True if backend unavailable.
        """
        template = PROMPT_TEMPLATES.get("target_segmentation_check")
        prompt_text = template.replace("{target_object}", str(target_object)) if template else (
            f"Decide if the white-background cutout contains the visible part of {target_object}. Reply JSON {{\"segmented\": true/false}}"
        )
        # Mock backend shortcut
        if self.backend == "mock":
            return True
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as t1, tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as t2:
                try:
                    orig_image.save(t1.name)
                except Exception:
                    t1.write(orig_image)
                    t1.flush()
                try:
                    target_cut_image.save(t2.name)
                except Exception:
                    t2.write(target_cut_image)
                    t2.flush()
                resp_text = None
                
                model = self.model_name_check
                _, api_key = self._get_provider_config(model)

                if api_key:
                    resp_text = self._openrouter_chat_with_images([t1.name, t2.name], prompt_text, model_override=model)
                if not resp_text:
                    resp_text = self._azure_chat_with_images([t1.name, t2.name], prompt_text)
                if resp_text:
                    # Extract first JSON object containing a boolean field "segmented"
                    txt = resp_text.strip()
                    # Try to find a JSON object substring
                    start_idxs = [i for i, ch in enumerate(txt) if ch == '{']
                    for st in start_idxs:
                        depth = 0
                        for i in range(st, len(txt)):
                            if txt[i] == '{':
                                depth += 1
                            elif txt[i] == '}':
                                depth -= 1
                                if depth == 0:
                                    frag = txt[st:i+1]
                                    try:
                                        obj = json.loads(frag)
                                        val = obj.get("segmented")
                                        if isinstance(val, bool):
                                            return bool(val)
                                    except Exception:
                                        break
                    # Fallback: simple heuristics
                    low = txt.lower()
                    if "true" in low and "false" not in low:
                        return True
                    if "false" in low and "true" not in low:
                        return False
        except Exception:
            pass
        # Default non-blocking behavior
        return True
