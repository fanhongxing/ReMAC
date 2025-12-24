import os, json, re, base64
from typing import List, Dict, Tuple, Optional
from PIL import Image
import requests

RANKING_PROMPT_TEMPLATE = os.getenv("RANKING_PROMPT_OVERRIDE", """You are an impartial evaluator comparing multiple amodal completion candidates for the SAME partially occluded object.

INPUT IMAGES:
1. Original image (contains only the visible part of the object; some regions are occluded / truncated).
2..N. Candidate completions of that object (Candidate A, B, C, ... in the exact order provided below).

YOUR GOAL:
Rank candidates from best to worst based on QUALITY OF COMPLETION of the SINGLE OBJECT, strictly ignoring scene background, global lighting, and minor color shifts. Focus on structural fidelity and plausible extrapolation of occluded parts.

WEIGHTED EVALUATION CRITERIA (apply in this order; do NOT skip):
1. Visible-Part Fidelity (CRITICAL):
    - The completed object's visible region MUST align pixel-wise (shape, contour, orientation, fine edges) with the visible portion in the original image.
    - Penalize any drift, inflated outlines, shrunken shapes, smoothing that erases distinctive protrusions, or warping of the visible part.
2. Plausible Completion of Occluded Regions:
    - Added (inferred) parts should be anatomically / structurally reasonable for this object category.
    - Penalize exaggerated bulk, invented appendages, mirrored duplicates, or implausible symmetry that contradicts the visible geometry.
3. Detail Preservation:
    - Fine structures (thin legs, handles, antennae, edges, slots, spokes, keyboard keys, racket strings, etc.) in the VISIBLE region should remain crisp—not blobbed, melted, or over-smoothed.
    - Penalize candidates that replace detailed areas with uniform color patches.
4. Boundary Cleanliness:
    - Object mask / silhouette should have coherent, continuous edges without jagged noise, holes, halos, floating fragments, translucent bleeding, or background leakage.
5. Semantic Correctness:
    - The candidate must still depict the SAME object category (e.g., an elephant must not become an undefined blob or a different animal). Penalize hallucinated secondary objects fused into the shape.
6. Conservative Realism:
    - Prefer a candidate that completes ONLY what is reasonably missing over one that hallucinates excessive, speculative structure.

TIE-BREAKERS (in order):
    a. Fewer artifacts (holes, duplicates, floating fragments).
    b. Sharper retention of original visible fine details.
    c. More anatomically/structurally plausible occluded completion.

WHAT TO IGNORE ENTIRELY:
- Background accuracy, global lighting differences, minor color or texture mismatches.
- Absolute scale or placement within the original image frame.
- Slight resolution differences if structural fidelity is retained.

ASSESSMENT FORMAT PER CANDIDATE (keep terse):
- Mention 1 strongest positive trait (e.g., "sharp visible contour", "plausible limb completion") and 1 key weakness if any (e.g., "loss of fine spokes", "inflated torso").
- If a candidate is clearly superior on criteria 1–3, say so explicitly (e.g., "best visible-part fidelity").

OUTPUT JSON (STRICT SCHEMA):
{
  "ranking": "A > B > C",  // Use '>' to denote strict ordering; use '=' only if indistinguishable on ALL weighted criteria
  "candidate_assessments": {
     "A": "One or two concise sentences",
     "B": "...",
     "C": "..."
  },
  "reasoning": "Single short paragraph (3–5 sentences) summarizing decisive factors referencing criteria numbers (e.g., 'A ranks first due to superior (1) and (3), B second with minor smoothing artifacts, C last for contour drift')."
}

RULES:
- Do NOT invent criteria. Use only those defined above.
- If one candidate is clearly best in (1) Visible-Part Fidelity plus at least one of (2) or (3), it MUST be ranked first.
- Avoid generic language; be specific about shape fidelity, detail retention, or artifact presence.
""")

def _encode_image_to_data_url(image_path: str) -> str:
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


class GeminiRankingAdapter:
    def __init__(self, model_name: str = "gemini-2.5-flash", dry_run: Optional[bool] = None):
        if dry_run is None:
            dry_run = bool(int(os.getenv("RANK_DRY_RUN", "0")))
        self.model_name = model_name
        # backend selection: always openrouter as requested
        self.backend = "openrouter"
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.or_base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        self.or_api_key = os.getenv("OPENROUTER_API_KEY", "")
        self._model = None
        if self.backend == "gemini":
            self.dry_run = dry_run or not self.api_key
            if not self.dry_run:
                try:
                    import google.generativeai as genai
                    genai.configure(api_key=self.api_key)
                    self._model = genai.GenerativeModel(self.model_name)
                except Exception:
                    self.dry_run = True
        else:
            self.dry_run = dry_run or not self.or_api_key

    def _load_images(self, paths: List[str]) -> List[Image.Image]:
        ims = []
        for p in paths:
            try:
                ims.append(Image.open(p).convert("RGB"))
            except Exception:
                pass
        return ims

    def _simulate_ranking(self, count: int) -> Dict[str, any]:
        # Deterministic pseudo-ranking: alphabetical order
        letters = [chr(ord('A') + i) for i in range(count)]
        ranking = ' > '.join(letters)
        assessments = {L: f"Simulated: placeholder evaluation for candidate {L}." for L in letters}
        reasoning = "Dry-run mode: returning deterministic alphabetical ranking without visual analysis."
        return {"ranking": ranking, "candidate_assessments": assessments, "reasoning": reasoning}

    def _extract_json(self, text: str) -> Dict[str, any]:
        # Find first JSON object containing keys ranking & candidate_assessments
        candidates = []
        for m in re.finditer(r"\{", text):
            start = m.start()
            depth = 0
            for i in range(start, len(text)):
                if text[i] == '{': depth += 1
                elif text[i] == '}': depth -= 1
                if depth == 0:
                    frag = text[start:i+1]
                    try:
                        obj = json.loads(frag)
                        if isinstance(obj, dict) and 'ranking' in obj and 'candidate_assessments' in obj:
                            return obj
                    except Exception:
                        pass
                    break
        # Fallback: try full text
        try:
            obj = json.loads(text)
            if isinstance(obj, dict) and 'ranking' in obj and 'candidate_assessments' in obj:
                return obj
        except Exception:
            pass
        return {}

    def _openrouter_rank(self, original_image_path: str, candidate_paths: List[str]) -> Dict[str, any]:
        if not self.or_api_key:
            return {}
        url = f"{self.or_base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.or_api_key}",
            "Content-Type": "application/json",
        }
        contents = [{"type": "text", "text": RANKING_PROMPT_TEMPLATE}]
        try:
            for p in [original_image_path] + candidate_paths:
                contents.append({"type": "image_url", "image_url": {"url": _encode_image_to_data_url(p)}})
        except Exception:
            return {}
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": RANKING_PROMPT_TEMPLATE},
                {"role": "user", "content": contents},
            ],
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        if resp.status_code != 200:
            return {}
        try:
            data = resp.json()
        except Exception:
            return {}
        txt = data.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
        obj = self._extract_json(txt)
        if obj:
            obj["raw_text"] = txt[:2000]
        return obj

    def rank(self, original_image_path: str, candidate_paths: List[str]) -> Dict[str, any]:
        if len(candidate_paths) == 0:
            raise ValueError("candidate_paths is empty")
        letters = [chr(ord('A') + i) for i in range(len(candidate_paths))]
        mapping = {L: p for L, p in zip(letters, candidate_paths)}
        if self.dry_run:
            out = self._simulate_ranking(len(candidate_paths))
            out['mapping'] = mapping
            return out
        # Real inference
        try:
            if self.backend == "openrouter":
                obj = self._openrouter_rank(original_image_path, candidate_paths)
            else:
                imgs = self._load_images([original_image_path] + candidate_paths)
                parts: List = [RANKING_PROMPT_TEMPLATE]
                parts.extend(imgs)
                resp = self._model.generate_content(parts, safety_settings={})
                txt = "".join([c.text for c in resp.candidates[0].content.parts if hasattr(c, 'text')]) if resp and resp.candidates else ""
                obj = self._extract_json(txt)
                if obj:
                    obj['raw_text'] = txt[:2000]
            if not obj:
                obj = self._simulate_ranking(len(candidate_paths))
            obj['mapping'] = mapping
            return obj
        except Exception as e:
            sim = self._simulate_ranking(len(candidate_paths))
            sim['error'] = str(e)
            sim['mapping'] = mapping
            return sim
