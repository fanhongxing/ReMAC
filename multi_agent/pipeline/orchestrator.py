from typing import Optional, Dict, Any, List
import os
import json
from PIL import Image, ImageChops
from ..adapters.seg_adapter import SegmentationAdapter
from ..adapters.inpaint_adapter import InpaintAdapter
from ..adapters.gpt_adapter import GPTAdapter
from ..adapters.gemini_ranking_adapter import GeminiRankingAdapter
from ..utils.mask_adapter import binary, invert, dilate, erode, combine_with, cutout_with_white, pad_by_edges, pad_by_edges_with_mask, safe_save
from ..utils.mask_adapter import pad_to_square
from ..utils.mask_adapter import directional_extend_mask
from ..utils.mask_adapter import keep_components_connected_to

class Orchestrator:
    """
    Contract:
    - Input: image (PIL), seg-text prompt, inpaint prompt (optionally from GPT), and options
    - Steps:
      1) Use GPT to refine the seg-text or generate an inpaint prompt (optional)
      2) Call segmentation to get mask
      3) Post-process mask
      4) Call inpainting with mask and prompt
    - Output: inpainted PIL image and intermediate artifacts
    """

    def __init__(self, seg: Optional[SegmentationAdapter] = None, ip: Optional[InpaintAdapter] = None, gpt: Optional[GPTAdapter] = None):
        self.seg = seg or SegmentationAdapter()
        self.ip = ip or InpaintAdapter()
        self.gpt = gpt or GPTAdapter()

    def run(
        self,
        image: Image.Image,
        seg_text: str,
        prompt: Optional[str] = None,
        # description generation mode: 'simple' (existing single prompt) or 'probabilistic' (LLM JSON hypotheses)
        generation_mode: str = "simple",
        # number of images to request from inpainting in simple mode
        num_inpaint_images: int = 1,
        mask_thr: int = 128,
        invert_mask: bool = False,
        dilate_k: int = 0,
        dilate_iters: int = 1,
        extended_pad_dilate_px: int = 0,
        # boundary prompt selector: either "boundary" or "boundary_bbox"
        boundary_mode: str = "boundary_bbox",
        bbox_json_path: Optional[str] = None,
        occluded_object: Optional[str] = None,
        debug_dir: str = None,
        enable_ranking: bool = True,
        enable_boundary_analysis: bool = True,
        restore_square_crop: bool = False,
        save_intermediate: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        # portable default debug dir (env > arg > ./debug)
        if debug_dir is None:
            debug_dir = os.getenv("DEBUG_DIR", "debug")
        os.makedirs(debug_dir, exist_ok=True)
        initial_dir = os.path.join(debug_dir, "initial")
        check_dir = os.path.join(debug_dir, "check")
        if save_intermediate:
            os.makedirs(initial_dir, exist_ok=True)
            os.makedirs(check_dir, exist_ok=True)
        seg_dir = os.path.join(debug_dir, "seg")
        meta_dir = os.path.join(debug_dir, "meta")
        variants_dir = os.path.join(debug_dir, "variants")
        if save_intermediate:
            os.makedirs(seg_dir, exist_ok=True)
            os.makedirs(meta_dir, exist_ok=True)
        # Deprecated: variants duplicated images; keep path but do not write into it any more
        inpaint_dir = os.path.join(debug_dir, "inpaint")
        if save_intermediate:
            os.makedirs(inpaint_dir, exist_ok=True)
        simple_dir = os.path.join(inpaint_dir, "simple")
        if save_intermediate:
            os.makedirs(simple_dir, exist_ok=True)
        ranking_dir = os.path.join(debug_dir, "ranking")
        if save_intermediate:
            os.makedirs(ranking_dir, exist_ok=True)
        # Ensure we have a saved original input for ranking reference
        try:
            if save_intermediate or enable_ranking:
                safe_save(image, os.path.join(debug_dir, "original_input.png"))
        except Exception:
            pass
        
        # Helper: ensure prompt contains single-object + white background constraints
        def _ensure_constraints(text: str, target: str) -> str:
            s = (text or "").strip()
            if not s:
                return f"There is only one complete {target}. Pure white background."
            if s[-1] not in ".!?":
                s = s + "."
            lower = s.lower()
            if "there is only one complete" not in lower:
                s = s + f" There is only one complete {target}."
                lower = s.lower()
            if "pure white background" not in lower:
                s = s + " Pure white background."
            return s
        # Save run parameters (under meta/)
        if save_intermediate:
            with open(os.path.join(meta_dir, "run_params.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "seg_text": seg_text,
                    "mask_thr": mask_thr,
                    "invert_mask": invert_mask,
                    "dilate_k": dilate_k,
                    "dilate_iters": dilate_iters,
                    "extended_pad_dilate_px": extended_pad_dilate_px,
                    "occluded_object": occluded_object or seg_text,
                    "boundary_mode": boundary_mode,
                    "num_inpaint_images": int(num_inpaint_images or 1),
                }, f, ensure_ascii=False, indent=2)

        # 1) Use occluding_object template to identify occluder list
        occluding_names = self.gpt.gen_inpaint_prompt_from_image(image, seg_text, prompt_type="occluding_object") or "[]"
        if save_intermediate:
            with open(os.path.join(initial_dir, "occluding_objects.txt"), "w", encoding="utf-8") as f:
                f.write(str(occluding_names))

        # 2) Describe the occluded object (prompt template) or generate probabilistic hypotheses
        desc = None
        hypo_json_raw = None
        if generation_mode == "probabilistic":
            # Use the probabilistic template
            hypo_json_raw = self.gpt.gen_inpaint_prompt_from_image(image, seg_text, prompt_type="probabilistic_hypotheses") or ""
            if save_intermediate:
                with open(os.path.join(initial_dir, "object_hypotheses_raw.txt"), "w", encoding="utf-8") as f:
                    f.write(str(hypo_json_raw))
            # Also build a base single-sentence description (simple result is required in probabilistic mode)
            desc = prompt or self.gpt.gen_inpaint_prompt_from_image(image, seg_text, prompt_type="prompt") or ""
            desc = _ensure_constraints(desc, (occluded_object or seg_text))
            try:
                if save_intermediate:
                    with open(os.path.join(initial_dir, "object_description.txt"), "w", encoding="utf-8") as f:
                        f.write(str(desc))
            except Exception:
                pass
        else:
            # Keep the original single-sentence description path
            desc = prompt or self.gpt.gen_inpaint_prompt_from_image(image, seg_text, prompt_type="prompt") or ""
            desc = _ensure_constraints(desc, (occluded_object or seg_text))
            if save_intermediate:
                with open(os.path.join(initial_dir, "object_description.txt"), "w", encoding="utf-8") as f:
                    f.write(str(desc))

        # Parse occluder list to array (robust)
        class_names: List[str] = [seg_text]
        raw_txt = str(occluding_names)
        def extract_bracket_list(s: str) -> str:
            # Return first balanced [ ... ] segment; fallback to original text if not found
            start_idxs = [i for i, ch in enumerate(s) if ch == '[']
            for start in start_idxs:
                depth = 0
                for i in range(start, len(s)):
                    if s[i] == '[':
                        depth += 1
                    elif s[i] == ']':
                        depth -= 1
                        if depth == 0:
                            return s[start:i+1]
            return s
        candidate = extract_bracket_list(raw_txt)
        occluders_raw_list: List[str] = []
        # Try JSON parsing first
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, list):
                occluders_raw_list = [str(x) for x in parsed]
        except Exception:
            # Fallback: split by multiple delimiters
            tmp = candidate.strip()
            tmp = tmp.strip('[]').strip()
            if tmp:
                # Support comma, Chinese comma, list separators, semicolons, new lines, bullet/ dash, etc.
                import re
                parts = re.split(r"[,，、;；\n\r\t•·•\-–—]+", tmp)
                occluders_raw_list = [p.strip().strip('\"\'') for p in parts if p.strip()]

        # Filter: remove same as target, negative words, and deduplicate (case-insensitive)
        target_lower = (occluded_object or seg_text).lower()
        NEG = {"none", "no", "no occlusion", "no occlusions", "n/a", "na", "null", "nothing"}
        seen = set()
        occluders_final: List[str] = []
        for name in occluders_raw_list:
            n = name.strip().strip('.').strip()
            if not n:
                continue
            nl = n.lower()
            if nl in NEG:
                continue
            if nl == target_lower:
                continue
            if nl not in seen:
                seen.add(nl)
                occluders_final.append(n)

        # Persist parsed and post-filtered lists (after heuristics)
        if save_intermediate:
            # with open(os.path.join(initial_dir, "occluding_objects_parsed.json"), "w", encoding="utf-8") as f:
            #     json.dump(occluders_raw_list, f, ensure_ascii=False, indent=2)
            with open(os.path.join(initial_dir, "occluding_objects_postfilter.json"), "w", encoding="utf-8") as f:
                json.dump(occluders_final, f, ensure_ascii=False, indent=2)

        # Classes for segmentation: target + filtered occluders
        class_names += occluders_final
        # Unified target key (lowercase) for bbox recording and boundary usage
        target_key = (occluded_object or seg_text).lower()

        # 3) Segmentation strategy: try joint (target+occluders), fallback to target-only, then occluders-only
        segmentation_attempt_logs: List[Dict[str, Any]] = []
        target_mask = None
        occluding_mask = None
        bbox_target = None
        from ..utils.mask_adapter import merge_masks

        def _collect_masks(payload: Dict[str, Any], class_names_all: List[str]):
            dets = payload.get("detections") or []
            masks_target: List[Image.Image] = []
            masks_occluding: List[Image.Image] = []
            occluder_named: List[tuple] = []  # (mask_img, class_name)
            target_key_local = (occluded_object or seg_text).lower()
            occluder_keys_local = [k.lower() for k in class_names_all if k != seg_text]
            nonlocal bbox_target
            for d in dets:
                cls = str(d.get("class_name", "")).lower()
                m = d.get("mask") or d.get("mask_base64") or d.get("mask_png_base64") or d.get("mask_png")
                mask_img = None
                if isinstance(m, str) and m:
                    import base64, io
                    if m.startswith("data:"):
                        m = m.split(",", 1)[1]
                    try:
                        from PIL import Image as PILImage
                        mask_img = PILImage.open(io.BytesIO(base64.b64decode(m))).convert("L")
                    except Exception:
                        mask_img = None
                if cls == target_key_local and bbox_target is None:
                    bbox_target = d.get("bounding_box_xyxy")
                if mask_img is not None:
                    if cls == target_key_local:
                        masks_target.append(mask_img)
                    elif cls in occluder_keys_local:
                        masks_occluding.append(mask_img)
                        occluder_named.append((mask_img, cls))
            return masks_target, masks_occluding, dets, occluder_named

        def _filter_masks_by_overlap(occluder_masks: List[Image.Image], target_visible: Image.Image, thr: float = 0.8):
            """Keep occluder masks whose overlap with target_visible is <= thr (measured against occluder area)."""
            if not occluder_masks or target_visible is None:
                return occluder_masks, []
            kept: List[Image.Image] = []
            dropped_indices: List[int] = []
            import numpy as np
            tv = binary(target_visible, thr=1)
            Tw, Th = tv.size
            arr_tv = (np.array(tv) > 0)
            for i, m in enumerate(occluder_masks):
                mm = m
                if mm.size != (Tw, Th):
                    try:
                        mm = mm.resize((Tw, Th), resample=Image.NEAREST)
                    except Exception:
                        pass
                arr_m = (np.array(binary(mm, thr=1)) > 0)
                denom = arr_m.sum()
                if denom == 0:
                    # empty mask, drop it
                    dropped_indices.append(i)
                    continue
                inter = (arr_m & arr_tv).sum()
                ratio = inter / float(denom)
                if ratio > thr:
                    dropped_indices.append(i)
                else:
                    kept.append(m)
            return kept, dropped_indices

        # 3.1) Joint segmentation (target + occluders)
        w, h = image.size
        seg_out_joint = self.seg.segment(image, text=seg_text, extra={"class_names": class_names, "return_masks": True})
        payload_joint = seg_out_joint.get("payload") or {}
        # if save_intermediate:
        #     with open(os.path.join(initial_dir, "detections.json"), "w", encoding="utf-8") as f:
        #         json.dump(payload_joint, f, ensure_ascii=False, indent=2)
        masks_target, masks_occluding, dets_joint, occluder_named = _collect_masks(payload_joint, class_names)
        segmentation_attempt_logs.append({
            "attempt": 1,
            "mode": "joint",
            "detections_count": len(dets_joint),
            "target_masks_count": len(masks_target),
            "occluder_masks_count": len(masks_occluding),
            "success": len(masks_target) > 0,
        })

        if masks_target:
            # Merge initial target masks
            target_mask = merge_masks(masks_target)
            # Save pre-rescue target artifacts
            try:
                if save_intermediate:
                    # safe_save(target_mask, os.path.join(initial_dir, "target_mask_before_rescue.png"))
                    target_cut_pre = cutout_with_white(image, target_mask)
                    # safe_save(target_cut_pre, os.path.join(initial_dir, "target_cutout_white_before_rescue.png"))
                else:
                    target_cut_pre = cutout_with_white(image, target_mask)
            except Exception:
                target_cut_pre = None
            # Early segmentation quality check & rescue (moved before occluder filtering & boundary expansion)
            seg_ok = True
            try:
                if target_cut_pre is not None:
                    seg_ok = self.gpt.check_target_segmentation_success(image, target_cut_pre, target_object=(occluded_object or seg_text))
            except Exception:
                seg_ok = True
            if not seg_ok:
                try:
                    alt_backend = "xsam" if getattr(self.seg, "backend", "sam") != "xsam" else "sam"
                    from ..adapters.seg_adapter import SegmentationAdapter as _Seg
                    seg_alt = _Seg(backend=alt_backend)
                    seg_out_alt = seg_alt.segment(image, text=seg_text, extra={"class_names": [seg_text], "return_masks": True})
                    payload_alt = (seg_out_alt or {}).get("payload") or {}
                    # Persist rescue detections
                    # try:
                    #     if save_intermediate:
                    #         with open(os.path.join(initial_dir, "detections_target_rescue.json"), "w", encoding="utf-8") as f:
                    #             json.dump({"backend": alt_backend, "payload": payload_alt}, f, ensure_ascii=False, indent=2)
                    # except Exception:
                    #     pass
                    masks_t_alt, _, dets_alt, _ = _collect_masks(payload_alt, [seg_text])
                    if masks_t_alt:
                        from ..utils.mask_adapter import merge_masks as _merge
                        target_mask = _merge(masks_t_alt)
                        # Update bbox_target if rescue provided one
                        for d in dets_alt:
                            cls = str(d.get("class_name", "")).lower()
                            if cls == (occluded_object or seg_text).lower():
                                bbox_target = d.get("bounding_box_xyxy") or bbox_target
                                break
                        # try:
                        #     if save_intermediate:
                        #         safe_save(target_mask, os.path.join(initial_dir, "target_mask_after_rescue.png"))
                        #         target_cut_post = cutout_with_white(image, target_mask)
                        #         safe_save(target_cut_post, os.path.join(initial_dir, "target_cutout_white_after_rescue.png"))
                        # except Exception:
                        #     pass
                        # mark run params
                        try:
                            if save_intermediate:
                                run_params_path = os.path.join(meta_dir, "run_params.json")
                                rp = {}
                                if os.path.exists(run_params_path):
                                    with open(run_params_path, "r", encoding="utf-8") as fr:
                                        rp = json.load(fr)
                                rp["target_seg_rescued"] = True
                                rp["target_seg_rescue_backend"] = alt_backend
                                with open(run_params_path, "w", encoding="utf-8") as fw:
                                    json.dump(rp, fw, ensure_ascii=False, indent=2)
                        except Exception:
                            pass
                except Exception:
                    pass
            # Save raw per-occluder masks (pre-filter)
            # try:
            #     if save_intermediate:
            #         raw_dir = os.path.join(initial_dir, "occluders_raw")
            #         os.makedirs(raw_dir, exist_ok=True)
            #         for i, (m, cls) in enumerate(occluder_named):
            #             fname = f"{i:02d}_{cls or 'unknown'}.png"
            #             safe_save(m, os.path.join(raw_dir, fname))
            # except Exception:
            #     pass
            # Occluder overlap filtering uses (possibly rescued) target_mask
            thr_overlap = kwargs.get("occluder_overlap_thr", 0.8)
            try:
                thr_overlap = float(thr_overlap)
            except Exception:
                thr_overlap = 0.8
            filtered_masks, dropped = _filter_masks_by_overlap(masks_occluding, target_mask, thr=thr_overlap)
            try:
                kept_idx = [i for i in range(len(masks_occluding)) if i not in set(dropped)]
                if save_intermediate:
                    with open(os.path.join(initial_dir, "occluder_overlap_filter.json"), "w", encoding="utf-8") as f:
                        json.dump({
                            "input_count": len(masks_occluding),
                            "kept": len(filtered_masks),
                            "dropped": len(dropped),
                            "kept_indices": kept_idx,
                            "dropped_indices": dropped,
                            "classes": [c for (_, c) in occluder_named]
                        }, f, ensure_ascii=False, indent=2)
                    # filt_dir = os.path.join(initial_dir, "occluders_filtered")
                    # os.makedirs(filt_dir, exist_ok=True)
                    # for i in kept_idx:
                    #     m, cls = occluder_named[i]
                    #     fname = f"{i:02d}_{cls or 'unknown'}.png"
                    #     safe_save(m, os.path.join(filt_dir, fname))
            except Exception:
                pass
            occluding_mask = merge_masks(filtered_masks) if filtered_masks else Image.new("L", image.size, 0)
        else:
            # 3.2) Fallback A: target-only segmentation
            seg_out_t = self.seg.segment(image, text=seg_text, extra={"class_names": [seg_text], "return_masks": True})
            payload_t = seg_out_t.get("payload") or {}
            # if save_intermediate:
            #     with open(os.path.join(initial_dir, "detections_fallback_target.json"), "w", encoding="utf-8") as f:
            #         json.dump(payload_t, f, ensure_ascii=False, indent=2)
            masks_t, _, dets_t, _ = _collect_masks(payload_t, [seg_text])
            segmentation_attempt_logs.append({
                "attempt": 2,
                "mode": "target_only",
                "detections_count": len(dets_t),
                "target_masks_count": len(masks_t),
                "occluder_masks_count": 0,
                "success": len(masks_t) > 0,
            })
            if masks_t:
                target_mask = merge_masks(masks_t)
                # Check if joint segmentation already produced occluder masks (from xsam)
                # If so, reuse them instead of re-segmenting occluders
                if masks_occluding:
                    # Reuse occluder masks from joint segmentation (preserve xsam results)
                    segmentation_attempt_logs.append({
                        "attempt": 2.5,
                        "mode": "occluders_reuse_from_joint",
                        "detections_count": len(occluder_named),
                        "target_masks_count": 0,
                        "occluder_masks_count": len(masks_occluding),
                        "success": len(masks_occluding) > 0,
                    })
                    # Apply overlap filtering on occluders using the (rescue) target mask
                    thr_overlap = kwargs.get("occluder_overlap_thr", 0.8)
                    try:
                        thr_overlap = float(thr_overlap)
                    except Exception:
                        thr_overlap = 0.8
                    filtered_masks_reuse, dropped_reuse = _filter_masks_by_overlap(masks_occluding, target_mask, thr=thr_overlap)
                    occluding_mask = merge_masks(filtered_masks_reuse) if filtered_masks_reuse else Image.new("L", image.size, 0)
                    # Save filtered occluder masks
                    # try:
                    #     if save_intermediate:
                    #         filt_dir_reuse = os.path.join(initial_dir, "occluders_filtered_reuse")
                    #         os.makedirs(filt_dir_reuse, exist_ok=True)
                    #         kept_idx_reuse = [i for i in range(len(masks_occluding)) if i not in set(dropped_reuse)]
                    #         for i in kept_idx_reuse:
                    #             m_reuse, cls_reuse = occluder_named[i]
                    #             fname_reuse = f"{i:02d}_{cls_reuse or 'unknown'}.png"
                    #             safe_save(m_reuse, os.path.join(filt_dir_reuse, fname_reuse))
                    # except Exception:
                    #     pass
                else:
                    # No occluder masks from joint segmentation, try to segment occluders separately
                    occluder_list = [c for c in class_names if c != seg_text]
                    if occluder_list:
                        try:
                            seg_out_occ_fb = self.seg.segment(image, text=seg_text, extra={"class_names": occluder_list, "return_masks": True})
                            payload_occ_fb = seg_out_occ_fb.get("payload") or {}
                            # if save_intermediate:
                            #     with open(os.path.join(initial_dir, "detections_fallback_occluders_after_target.json"), "w", encoding="utf-8") as f:
                            #         json.dump(payload_occ_fb, f, ensure_ascii=False, indent=2)
                            _, masks_occ_fb, dets_occ_fb, occluder_named_fb = _collect_masks(payload_occ_fb, class_names)
                            segmentation_attempt_logs.append({
                                "attempt": 2.5,
                                "mode": "occluders_after_target_only",
                                "detections_count": len(dets_occ_fb),
                                "target_masks_count": 0,
                                "occluder_masks_count": len(masks_occ_fb),
                                "success": len(masks_occ_fb) > 0,
                            })
                            # Apply overlap filtering on occluders using the target mask
                            thr_overlap = kwargs.get("occluder_overlap_thr", 0.8)
                            try:
                                thr_overlap = float(thr_overlap)
                            except Exception:
                                thr_overlap = 0.8
                            filtered_masks_fb, dropped_fb = _filter_masks_by_overlap(masks_occ_fb, target_mask, thr=thr_overlap)
                            occluding_mask = merge_masks(filtered_masks_fb) if filtered_masks_fb else Image.new("L", image.size, 0)
                            # Save filtered occluder masks
                            # try:
                            #     if save_intermediate:
                            #         filt_dir_fb = os.path.join(initial_dir, "occluders_filtered_fallback")
                            #         os.makedirs(filt_dir_fb, exist_ok=True)
                            #         kept_idx_fb = [i for i in range(len(masks_occ_fb)) if i not in set(dropped_fb)]
                            #         for i in kept_idx_fb:
                            #             m_fb, cls_fb = occluder_named_fb[i]
                            #             fname_fb = f"{i:02d}_{cls_fb or 'unknown'}.png"
                            #             safe_save(m_fb, os.path.join(filt_dir_fb, fname_fb))
                            # except Exception:
                            #     pass
                        except Exception:
                            occluding_mask = Image.new("L", image.size, 0)
                    else:
                        occluding_mask = Image.new("L", image.size, 0)
            else:
                # 3.3) Fallback B: occluders-only (collect masks for visualization/check; still raise if target missing)
                occluder_list = [c for c in class_names if c != seg_text]
                if occluder_list:
                    seg_out_o = self.seg.segment(image, text=seg_text, extra={"class_names": occluder_list, "return_masks": True})
                    payload_o = seg_out_o.get("payload") or {}
                    # if save_intermediate:
                    #     with open(os.path.join(initial_dir, "detections_fallback_occluders.json"), "w", encoding="utf-8") as f:
                    #         json.dump(payload_o, f, ensure_ascii=False, indent=2)
                    _, masks_o, dets_o, _ = _collect_masks(payload_o, class_names)
                    segmentation_attempt_logs.append({
                        "attempt": 3,
                        "mode": "occluders_only",
                        "detections_count": len(dets_o),
                        "target_masks_count": 0,
                        "occluder_masks_count": len(masks_o),
                        "success": len(masks_o) > 0,
                    })
                    occluding_mask = merge_masks(masks_o) if masks_o else Image.new("L", image.size, 0)
                # Target mask still missing: raise
                raise RuntimeError("No target mask from segmentation after fallback (target-only and occluders-only)")

        # Record segmentation_attempts.json and attempts count
        try:
            if save_intermediate:
                with open(os.path.join(initial_dir, "segmentation_attempts.json"), "w", encoding="utf-8") as f:
                    json.dump(segmentation_attempt_logs, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        try:
            if save_intermediate:
                run_params_path = os.path.join(meta_dir, "run_params.json")
                if os.path.exists(run_params_path):
                    with open(run_params_path, "r", encoding="utf-8") as fr:
                        rp = json.load(fr)
                else:
                    rp = {}
                rp["segmentation_attempts"] = len(segmentation_attempt_logs)
                with open(run_params_path, "w", encoding="utf-8") as fw:
                    json.dump(rp, fw, ensure_ascii=False, indent=2)
        except Exception:
            pass



        # Save masks and overlays
        if save_intermediate:
            safe_save(target_mask, os.path.join(initial_dir, "target_mask.png"))
            safe_save(occluding_mask, os.path.join(initial_dir, "occluding_mask.png"))
            # safe_save(combine_with(image, target_mask), os.path.join(initial_dir, "target_overlay.png"))
            # safe_save(combine_with(image, occluding_mask), os.path.join(initial_dir, "occluding_overlay.png"))

        # Target cutout on white background, used for boundary expansion and inpainting
        target_cut = cutout_with_white(image, target_mask)
        # if save_intermediate:
        #     safe_save(target_cut, os.path.join(initial_dir, "target_cutout_white.png"))

        # Rescue logic moved earlier; simple check artifact now (reflect final segmentation state)
        try:
            seg_ok_final = True
            target_cut_tmp = cutout_with_white(image, target_mask) if target_mask is not None else None
            if target_cut_tmp is not None:
                seg_ok_final = self.gpt.check_target_segmentation_success(image, target_cut_tmp, target_object=(occluded_object or seg_text))
            if save_intermediate:
                with open(os.path.join(check_dir, "target_seg_check.json"), "w", encoding="utf-8") as f:
                    json.dump({"segmented": bool(seg_ok_final), "early_rescue": True}, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

        # 4) boundary_bbox: out-of-boundary detection
        bbox_json_path_out = os.path.join(initial_dir, "bbox_info.json")
        bbox_info_data = None
        if bbox_target is not None:
            bbox_info_data = {f"{target_key}_0": {"bounding_box_xyxy": bbox_target, "image_width": w, "image_height": h}}
            if save_intermediate:
                with open(bbox_json_path_out, "w", encoding="utf-8") as f:
                    json.dump(bbox_info_data, f, ensure_ascii=False, indent=2)
        # Third LLM: boundary analysis (choose template by boundary_mode, default boundary_bbox)
        pad_dirs: List[str] = []
        pad_amount: float = 0.0

        if enable_boundary_analysis:
            prompt_name = "boundary_bbox" if boundary_mode == "boundary_bbox" else "boundary"
            bb = self.gpt.gen_inpaint_prompt_from_image(
                image,
                seg_text,
                prompt_type=prompt_name,
                bbox_json_path=bbox_json_path_out if (bbox_target is not None and save_intermediate) else None,
                occluded_object=target_key,
                bbox_data=bbox_info_data,
            )
            if save_intermediate:
                with open(os.path.join(initial_dir, "boundary_bbox.json.txt"), "w", encoding="utf-8") as f:
                    f.write(str(bb))
            # Parse boundary output robustly: allow code-fences and prose, extract the first JSON object
            def _extract_json_object(s: str) -> dict:
                s = s.strip()
                # Strip code fences ```...```
                if s.startswith("```"):
                    # Get the first fenced block content
                    try:
                        first = s.find("```")
                        if first != -1:
                            rest = s[first+3:]
                            # Skip optional language tag to the next newline
                            nl = rest.find("\n")
                            if nl != -1:
                                rest = rest[nl+1:]
                            end = rest.find("```")
                            if end != -1:
                                s = rest[:end]
                    except Exception:
                        pass
                # Scan for a balanced JSON object
                start_idxs = [i for i, ch in enumerate(s) if ch == '{']
                for start in start_idxs:
                    depth = 0
                    for i in range(start, len(s)):
                        if s[i] == '{':
                            depth += 1
                        elif s[i] == '}':
                            depth -= 1
                            if depth == 0:
                                frag = s[start:i+1]
                                try:
                                    return json.loads(frag)
                                except Exception:
                                    break
                # Try parsing the whole string as JSON
                try:
                    return json.loads(s)
                except Exception:
                    return {}

            obj = _extract_json_object(str(bb))
            if isinstance(obj, dict):
                dirs = obj.get("extension_direction") or obj.get("directions") or []
                amt = obj.get("extension_amount") or obj.get("amount") or 0.0
                # Normalize directions
                if isinstance(dirs, str):
                    dirs = [dirs]
                norm = []
                for d in dirs:
                    dl = str(d).strip().lower()
                    if dl in {"left","right","top","bottom"}:
                        norm.append(dl)
                pad_dirs = norm
                # Parse numeric value (tolerate string/percentage)
                if isinstance(amt, (int, float)):
                    pad_amount = float(amt)
                elif isinstance(amt, str):
                    t = amt.strip().lower()
                    # Map coarse levels to numeric per new prompt
                    level_map = {"slight": 0.2, "moderate": 0.5, "large": 0.7}
                    if t in level_map:
                        pad_amount = level_map[t]
                    elif t.endswith('%'):
                        try:
                            pad_amount = float(t[:-1]) / 100.0
                        except Exception:
                            pad_amount = 0.0
                    else:
                        try:
                            pad_amount = float(t)
                        except Exception:
                            pad_amount = 0.0
                # Clamp to [0,1] and, if close to tiers, snap to nearest tier for stability
                try:
                    if pad_amount < 0:
                        pad_amount = 0.0
                    if pad_amount > 1:
                        pad_amount = 1.0
                    tiers = [0.2, 0.5, 0.7]
                    # snap if within 0.05
                    for v in tiers:
                        if abs(pad_amount - v) <= 0.05:
                            pad_amount = v
                            break
                except Exception:
                    pass
            # Persist parsed boundary result for debugging
            try:
                if save_intermediate:
                    with open(os.path.join(initial_dir, "boundary_bbox_parsed.json"), "w", encoding="utf-8") as f:
                        json.dump({"extension_direction": pad_dirs, "extension_amount": pad_amount}, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

        # If edge_grow_ratio is provided while extended_pad_dilate_px is not, compute pixels from extension amount
        edge_grow_ratio = kwargs.get("edge_grow_ratio")
        if edge_grow_ratio is not None and (extended_pad_dilate_px is None or extended_pad_dilate_px == 0):
            if pad_dirs and pad_amount > 0:
                w0, h0 = image.size
                # Select base extent per direction (horizontal: w0*pad_amount, vertical: h0*pad_amount); take the max as base
                horiz = int(w0 * pad_amount) if any(d in pad_dirs for d in ["left","right"]) else 0
                vert = int(h0 * pad_amount) if any(d in pad_dirs for d in ["top","bottom"]) else 0
                base_extent = max(horiz, vert, 0)
                px = int(round(base_extent * float(edge_grow_ratio)))
                extended_pad_dilate_px = max(0, px)
                # Update run_params.json with computed values (append)
                try:
                    if save_intermediate:
                        run_params_path = os.path.join(meta_dir, "run_params.json")
                        if os.path.exists(run_params_path):
                            with open(run_params_path, "r", encoding="utf-8") as fr:
                                rp = json.load(fr)
                        else:
                            rp = {}
                        rp["edge_grow_ratio"] = edge_grow_ratio
                        rp["extended_pad_dilate_px"] = extended_pad_dilate_px
                        with open(run_params_path, "w", encoding="utf-8") as fw:
                            json.dump(rp, fw, ensure_ascii=False, indent=2)
                except Exception:
                    pass

        # Visible target erosion control: removed. We no longer erode the visible region before subtraction.

        # 5) Pad target cutout according to boundary analysis
        pad_mask = None
        pad_mask_raw = None  # keep raw pad region before inward-grow
        pad_offsets = (0,0,0,0)
        if pad_dirs and pad_amount > 0:
            target_cut, pad_mask, pad_offsets = pad_by_edges_with_mask(target_cut, pad_dirs, pad_amount, fill=(255, 255, 255))
            pad_mask_raw = pad_mask.copy() if pad_mask is not None else None
            # Directional inward-grow only applies to pad_mask; avoid saving pre/post versions to reduce redundancy
            if extended_pad_dilate_px and pad_mask is not None:
                pad_mask = directional_extend_mask(pad_mask, pad_dirs, target_mask, grow_px=extended_pad_dilate_px, offsets=pad_offsets)
            # if save_intermediate:
            #     safe_save(target_cut, os.path.join(initial_dir, "target_cutout_white_padded.png"))
            #     if pad_mask is not None:
            #         safe_save(pad_mask, os.path.join(initial_dir, "boundary_pad_mask.png"))  # final merged version

        # 6) Occluder mask processing (new order): subtract visible target now, pad if needed; delay dilation to final step
        # Save pre/post for occluder mask processing
        occ_pad_mask = None  # default, set when padding occurs
        occ_mask_pre_raw = binary(occluding_mask, thr=mask_thr)
        # Subtract visible target region immediately (no erosion)
        try:
            vis_bin0 = binary(target_mask, thr=mask_thr) if target_mask is not None else Image.new("L", image.size, 0)
            occ_mask_sub = ImageChops.subtract(occ_mask_pre_raw, vis_bin0)
        except Exception:
            occ_mask_sub = occ_mask_pre_raw.copy()
        # if save_intermediate:
        #     safe_save(occ_mask_sub, os.path.join(initial_dir, "occluding_mask_bin.png"))
        occ_mask_post = occ_mask_sub.copy()  # placeholder for overlay pre-dilate
        # Synchronized padding on the subtracted mask (no dilation here)
        occ_mask_bin = occ_mask_sub
        if pad_dirs and pad_amount > 0:
            occ_padded, occ_pad_mask, _ = pad_by_edges_with_mask(occ_mask_bin, pad_dirs, pad_amount, fill=0, mask_fill=255)
            # if save_intermediate:
            #     safe_save(occ_pad_mask, os.path.join(initial_dir, "occ_pad_mask.png"))
            occ_mask_expanded = ImageChops.lighter(occ_padded, occ_pad_mask)
            if pad_mask is not None:
                pm_bin = binary(pad_mask, thr=1)
                occ_mask_expanded = ImageChops.lighter(occ_mask_expanded, pm_bin)
            occ_mask_bin = occ_mask_expanded
        # if save_intermediate:
        #     safe_save(occ_mask_bin, os.path.join(initial_dir, "occluding_mask_padded.png"))  # pre-dilate, post-subtract
        # Remove extra overlays/mask backups to reduce redundancy

        # (Removed) Do not save inpaint_input_overlay.png (per request)

        # 6.6) Generate multi-region visualization final_regions_overlay.png
        try:
            base = target_cut.convert("RGB")
            W, H = base.size
            origW, origH = image.size
            left_off, top_off, _, _ = pad_offsets if 'pad_offsets' in locals() else (0,0,0,0)

            from PIL import Image as PILImage

            def place_mask(original_mask: Image.Image) -> Image.Image:
                """Place the original-size mask into the padded canvas without resizing; return directly if sizes match."""
                if original_mask is None:
                    return PILImage.new("L", (W, H), 0)
                if original_mask.size == (W, H):
                    return original_mask
                # If target_cut is padded larger and we have offsets, paste with offsets
                if (W, H) != (origW, origH) and original_mask.size == (origW, origH):
                    canvas = PILImage.new("L", (W, H), 0)
                    canvas.paste(original_mask, (left_off, top_off))
                    return canvas
                # Fallback: if sizes still don't match and not original size, paste at top-left without resize
                canvas = PILImage.new("L", (W, H), 0)
                canvas.paste(original_mask, (0, 0))
                return canvas

            # Prepare layers (do not resize occluder masks)
            occ_pre_canvas = place_mask(occ_mask_pre if 'occ_mask_pre' in locals() else None)
            occ_post_canvas = place_mask(occ_mask_post if 'occ_mask_post' in locals() else None)
            pad_edge_canvas = place_mask(occ_pad_mask if pad_dirs and pad_amount>0 else None)
            tgt_pad_raw_canvas = place_mask(pad_mask_raw if pad_mask_raw is not None else None)
            tgt_pad_dil_canvas = place_mask(pad_mask if pad_mask is not None else None)

            import numpy as np

            arr_pre = np.array(occ_pre_canvas) > 0
            arr_post = np.array(occ_post_canvas) > 0
            arr_pad_edge = np.array(pad_edge_canvas) > 0
            arr_tgt_pad_raw = np.array(tgt_pad_raw_canvas) > 0
            arr_tgt_pad_dil = np.array(tgt_pad_dil_canvas) > 0

            overlay = Image.new("RGBA", (W, H), (0,0,0,0))
            px = overlay.load()
            for y in range(H):
                for x in range(W):
                    # Priority same as original implementation
                    if arr_post[y,x] and not arr_pre[y,x]:      # occluder dilation added
                        px[x,y] = (255, 165, 0, 140)
                    elif arr_pre[y,x]:                          # original occluder region
                        px[x,y] = (255, 0, 0, 140)
                    elif arr_pad_edge[y,x]:                     # edge extension area (from padding)
                        px[x,y] = (0, 255, 255, 140)
                    elif arr_tgt_pad_dil[y,x] and not arr_tgt_pad_raw[y,x]:  # target pad dilated addition
                        px[x,y] = (255, 0, 255, 140)
                    elif arr_tgt_pad_raw[y,x]:                  # target pad original
                        px[x,y] = (0, 255, 0, 140)

            final_vis = base.convert("RGBA")
            final_vis.alpha_composite(overlay)
            if save_intermediate:
                safe_save(final_vis.convert("RGB"), os.path.join(initial_dir, "final_regions_overlay.png"))

            legend = {
                "red": "Original occluder region (not dilated)",
                "orange": "Occluder dilation added region",
                "cyan": "Edge extension added region",
                "green": "Target edge extension original region",
                "magenta": "Target edge extension dilation added region"
            }
            # if save_intermediate:
            #     with open(os.path.join(initial_dir, "final_regions_overlay_legend.json"), "w", encoding="utf-8") as f:
            #         json.dump(legend, f, ensure_ascii=False, indent=2)
        except Exception as e:
            # Visualization failures should not block the main flow
            pass

        # 6.7) Before inpainting, build single composite image for missed occluder check
        # Composite rules:
        # - Start from original image
        # - Whiten all currently identified occluder pixels (occluding_mask pre-subtraction)
        # - Overlay a translucent color on the visible target region (while keeping original appearance faintly visible)
        try:
            import numpy as np
            base_img = image.convert("RGB")
            arr = np.array(base_img)
            # Use current occluding_mask (prior to subtraction) for whitening
            occ_src = occluding_mask if occluding_mask is not None else Image.new("L", image.size, 0)
            occ_arr = (np.array(binary(occ_src, thr=1)) > 0)
            # Visible target region (target_mask) for optional brightening
            vis_src = target_mask if target_mask is not None else Image.new("L", image.size, 0)
            vis_arr = (np.array(binary(vis_src, thr=1)) > 0)
            # Whiten occluders
            arr[occ_arr] = 255
            # Overlay translucent color on visible target region (skip whitened areas)
            overlay_color = kwargs.get("missed_check_overlay_color", (0, 255, 0))
            if isinstance(overlay_color, (list, tuple)) and len(overlay_color) == 3:
                try:
                    color_rgb = tuple(max(0, min(255, int(c))) for c in overlay_color)
                except Exception:
                    color_rgb = (0, 255, 0)
            else:
                color_rgb = (0, 255, 0)
            alpha = kwargs.get("missed_check_overlay_alpha", 0.35)
            try:
                alpha = float(alpha)
            except Exception:
                alpha = 0.35
            alpha = max(0.0, min(1.0, alpha))
            sel = vis_arr & (~occ_arr)
            if alpha > 0 and sel.any():
                sub = arr[sel].astype(np.float32)
                color_vec = np.array(color_rgb, dtype=np.float32)
                blended = (1 - alpha) * sub + alpha * color_vec
                arr[sel] = np.clip(blended, 0, 255).astype(np.uint8)
            composite = Image.fromarray(arr, mode="RGB")
            # Save composite for debug
            comp_path = os.path.join(check_dir, "missed_occlusion_input.png")
            # if save_intermediate:
            #     safe_save(composite, comp_path)
            missed_raw = self.gpt.check_missed_occlusions(composite, target_object=(occluded_object or seg_text)) or "[]"
            if save_intermediate:
                with open(os.path.join(check_dir, "missed_occluders_raw.txt"), "w", encoding="utf-8") as f:
                    f.write(str(missed_raw))

            # Parse list: prefer the bracketed list after the "Final Answer" section; fallback to the last bracketed list in the text
            def _extract_final_answer_list(s: str) -> str:
                text = s if isinstance(s, str) else str(s)
                lower = text.lower()

                def _first_bracket_list_from(pos: int) -> str:
                    # From a given position, return the first balanced [ ... ] segment
                    for start in range(pos, len(text)):
                        if text[start] == '[':
                            depth = 0
                            for i in range(start, len(text)):
                                if text[i] == '[':
                                    depth += 1
                                elif text[i] == ']':
                                    depth -= 1
                                    if depth == 0:
                                        return text[start:i+1]
                    return ""

                # 1) Prefer the first bracketed list after the "Final Answer" notation
                idx = lower.find("final answer")
                if idx != -1:
                    seg = _first_bracket_list_from(idx)
                    if seg:
                        return seg

                # 2) Fallback: scan entire text and return the last balanced bracketed list
                last = ""
                start_idxs = [i for i, ch in enumerate(text) if ch == '[']
                for start in start_idxs:
                    depth = 0
                    for i in range(start, len(text)):
                        if text[i] == '[':
                            depth += 1
                        elif text[i] == ']':
                            depth -= 1
                            if depth == 0:
                                last = text[start:i+1]
                                break
                return last or text

            candidate = _extract_final_answer_list(str(missed_raw))
            missed_list: List[str] = []
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, list):
                    missed_list = [str(x) for x in parsed]
            except Exception:
                import re
                tmp = candidate.strip().strip('[]').strip()
                if tmp:
                    parts = re.split(r"[,，、;；\n\r\t•·\-–—]+", tmp)
                    missed_list = [p.strip().strip('"\'') for p in parts if p.strip()]

            # Filter: remove same as target, meaningless words, deduplicate
            target_lower = (occluded_object or seg_text).lower()
            NEG = {"none", "no", "no occlusion", "no occlusions", "n/a", "na", "null", "nothing", "[]"}
            seen = set()
            missed_final: List[str] = []
            for name in missed_list:
                n = name.strip().strip('.')
                if not n:
                    continue
                nl = n.lower()
                if nl in NEG:
                    continue
                if nl == target_lower:
                    continue
                if nl not in seen:
                    seen.add(nl)
                    missed_final.append(n)

            if save_intermediate:
                with open(os.path.join(check_dir, "missed_occluders_parsed.json"), "w", encoding="utf-8") as f:
                    json.dump(missed_final, f, ensure_ascii=False, indent=2)

            # If there are missed occluder categories, perform a second segmentation and merge their masks
            if missed_final:
                # Only request these classes to reduce confusion
                    seg_out2 = self.seg.segment(image, text=seg_text, extra={"class_names": missed_final, "return_masks": True})
                    payload2 = (seg_out2 or {}).get("payload") or {}
                    if save_intermediate:
                        with open(os.path.join(check_dir, "detections_postcheck.json"), "w", encoding="utf-8") as f:
                            json.dump(payload2, f, ensure_ascii=False, indent=2)
                    dets2 = payload2.get("detections") or []
                    add_masks: List[Image.Image] = []
                    add_named: List[tuple] = []  # (mask_img, class_name)
                    missed_keys = [k.lower() for k in missed_final]
                    for d in dets2:
                        cls = str(d.get("class_name", "")).lower()
                        if cls not in missed_keys:
                            continue
                        m = d.get("mask") or d.get("mask_base64") or d.get("mask_png_base64") or d.get("mask_png")
                        if isinstance(m, str) and m:
                            import base64, io
                            if m.startswith("data:"):
                                m = m.split(",", 1)[1]
                            try:
                                from PIL import Image as PILImage
                                mask_img = PILImage.open(io.BytesIO(base64.b64decode(m))).convert("L")
                                add_masks.append(mask_img)
                                add_named.append((mask_img, cls))
                            except Exception:
                                pass
                    if add_masks:
                        from ..utils.mask_adapter import merge_masks
                    # Save raw added occluders (pre-filter)
                    try:
                        if save_intermediate:
                            raw_add_dir = os.path.join(check_dir, "occluders_added_raw")
                            os.makedirs(raw_add_dir, exist_ok=True)
                            for i, (m, cls) in enumerate(add_named):
                                fname = f"{i:02d}_{cls or 'unknown'}.png"
                                safe_save(m, os.path.join(raw_add_dir, fname))
                    except Exception:
                        pass
                    # Before merging, drop masks that overlap target-visible region > threshold
                    thr_overlap = kwargs.get("occluder_overlap_thr", 0.8)
                    try:
                        thr_overlap = float(thr_overlap)
                    except Exception:
                        thr_overlap = 0.8
                    filtered_add, dropped_idx = _filter_masks_by_overlap(add_masks, target_mask, thr=thr_overlap)
                    try:
                        kept_add_idx = [i for i in range(len(add_masks)) if i not in set(dropped_idx)]
                        if save_intermediate:
                            with open(os.path.join(check_dir, "missed_occluders_overlap_filter.json"), "w", encoding="utf-8") as f:
                                json.dump({
                                    "input_count": len(add_masks),
                                    "kept": len(filtered_add),
                                    "dropped": len(dropped_idx),
                                    "kept_indices": kept_add_idx,
                                    "dropped_indices": dropped_idx,
                                    "classes": [c for (_, c) in add_named]
                                }, f, ensure_ascii=False, indent=2)
                            # Save filtered added occluders
                            filt_add_dir = os.path.join(check_dir, "occluders_added_filtered")
                            os.makedirs(filt_add_dir, exist_ok=True)
                            for i in kept_add_idx:
                                m, cls = add_named[i]
                                fname = f"{i:02d}_{cls or 'unknown'}.png"
                                safe_save(m, os.path.join(filt_add_dir, fname))
                    except Exception:
                        pass

                    add_mask = merge_masks(filtered_add) if filtered_add else Image.new("L", image.size, 0)

                    # Merge into existing occluder mask
                    occluding_mask = ImageChops.lighter(occluding_mask, add_mask)
                    # Keep occluding_mask raw at this stage; cleaning will be applied right before inpainting
                    if save_intermediate:
                        safe_save(occluding_mask, os.path.join(check_dir, "occluding_mask_postcheck.png"))

                    # Re-run: binarize current occluder mask, subtract visible target, then synchronized padding (no dilation here)
                    occ_mask_pre = binary(occluding_mask, thr=mask_thr)
                    try:
                        vis_bin2 = binary(target_mask, thr=mask_thr) if target_mask is not None else Image.new("L", image.size, 0)
                        occ_mask_sub2 = ImageChops.subtract(occ_mask_pre, vis_bin2)
                    except Exception:
                        occ_mask_sub2 = occ_mask_pre.copy()
                    if save_intermediate:
                        safe_save(occ_mask_sub2, os.path.join(check_dir, "occluding_mask_bin.png"))
                    occ_mask_post = occ_mask_sub2.copy()
                    occ_mask_bin = occ_mask_sub2
                    if pad_dirs and pad_amount > 0:
                        occ_padded, occ_pad_mask, _ = pad_by_edges_with_mask(occ_mask_bin, pad_dirs, pad_amount, fill=0, mask_fill=255)
                        if save_intermediate:
                            safe_save(occ_pad_mask, os.path.join(check_dir, "occ_pad_mask.png"))
                        occ_mask_expanded = ImageChops.lighter(occ_padded, occ_pad_mask)
                        if pad_mask is not None:
                            pm_bin = binary(pad_mask, thr=1)
                            occ_mask_expanded = ImageChops.lighter(occ_mask_expanded, pm_bin)
                        occ_mask_bin = occ_mask_expanded
                    if save_intermediate:
                        safe_save(occ_mask_bin, os.path.join(check_dir, "occluding_mask_padded.png"))

                    # Optional: update overlay (overwrite original)
                    try:
                        base = target_cut.convert("RGB")
                        W, H = base.size
                        origW, origH = image.size
                        left_off, top_off, _, _ = pad_offsets if 'pad_offsets' in locals() else (0,0,0,0)
                        if 'square_offsets' in locals() and isinstance(square_offsets, tuple):
                            try:
                                l2, t2, _, _ = square_offsets
                                left_off += int(l2)
                                top_off += int(t2)
                            except Exception:
                                pass
                        from PIL import Image as PILImage
                        def place_mask(original_mask: Image.Image) -> Image.Image:
                            if original_mask is None:
                                return PILImage.new("L", (W, H), 0)
                            if original_mask.size == (W, H):
                                return original_mask
                            if (W, H) != (origW, origH) and original_mask.size == (origW, origH):
                                canvas = PILImage.new("L", (W, H), 0)
                                canvas.paste(original_mask, (left_off, top_off))
                                return canvas
                            canvas = PILImage.new("L", (W, H), 0)
                            canvas.paste(original_mask, (0, 0))
                            return canvas
                        occ_pre_canvas = place_mask(occ_mask_pre if 'occ_mask_pre' in locals() else None)
                        occ_post_canvas = place_mask(occ_mask_post if 'occ_mask_post' in locals() else None)
                        pad_edge_canvas = place_mask(occ_pad_mask if pad_dirs and pad_amount>0 else None)
                        tgt_pad_raw_canvas = place_mask(pad_mask_raw if pad_mask_raw is not None else None)
                        tgt_pad_dil_canvas = place_mask(pad_mask if pad_mask is not None else None)
                        import numpy as np
                        arr_pre = np.array(occ_pre_canvas) > 0
                        arr_post = np.array(occ_post_canvas) > 0
                        arr_pad_edge = np.array(pad_edge_canvas) > 0
                        arr_tgt_pad_raw = np.array(tgt_pad_raw_canvas) > 0
                        arr_tgt_pad_dil = np.array(tgt_pad_dil_canvas) > 0
                        overlay = Image.new("RGBA", (W, H), (0,0,0,0))
                        px = overlay.load()
                        for y in range(H):
                            for x in range(W):
                                if arr_post[y,x] and not arr_pre[y,x]:
                                    px[x,y] = (255, 165, 0, 140)
                                elif arr_pre[y,x]:
                                    px[x,y] = (255, 0, 0, 140)
                                elif arr_pad_edge[y,x]:
                                    px[x,y] = (0, 255, 255, 140)
                                elif arr_tgt_pad_dil[y,x] and not arr_tgt_pad_raw[y,x]:
                                    px[x,y] = (255, 0, 255, 140)
                                elif arr_tgt_pad_raw[y,x]:
                                    px[x,y] = (0, 255, 0, 140)
                        final_vis2 = base.convert("RGBA")
                        final_vis2.alpha_composite(overlay)
                        if save_intermediate:
                            safe_save(final_vis2.convert("RGB"), os.path.join(check_dir, "final_regions_overlay.png"))
                    except Exception:
                        pass
        except Exception:
            # Checker agent failure should not block the main flow
            pass

        # 6.9) Finalize inpaint mask (new order): after all subtractions and padding, now apply invert + dilation once
        inpaint_mask = binary(occ_mask_bin, thr=1)
        if invert_mask:
            inpaint_mask = invert(inpaint_mask)
        if dilate_k and dilate_k > 0:
            inpaint_mask = dilate(inpaint_mask, k=dilate_k, iters=max(1, int(dilate_iters)))
        # try:
        #     if save_intermediate:
        #         safe_save(inpaint_mask, os.path.join(seg_dir, "inpaint_mask.png"))
        # except Exception:
        #     pass

        # 6.10) Before inpainting, uniformly pad the visible part image and mask to a 1:1 square
        # Fill the visible image with pure white; fill the mask with pure black (do not inpaint in these newly filled areas).
        square_offsets = (0, 0, 0, 0)
        try:
            tw, th = target_cut.size
            mw, mh = inpaint_mask.size if inpaint_mask is not None else target_cut.size
            side = max(tw, th, mw, mh)
            target_cut_sq, square_offsets = pad_to_square(target_cut, fill=(255, 255, 255), side=side)
            inpaint_mask_sq, _ = pad_to_square(inpaint_mask, fill=0, side=side)
            target_cut = target_cut_sq
            inpaint_mask = inpaint_mask_sq
            # try:
            #     if save_intermediate:
            #         safe_save(target_cut, os.path.join(seg_dir, "inpaint_input_image_square_preview.png"))
            #         safe_save(inpaint_mask, os.path.join(seg_dir, "inpaint_input_mask_square_preview.png"))
            # except Exception:
            #     pass
        except Exception:
            # Fallback: If padding to square fails, continue using original size
            square_offsets = (0, 0, 0, 0)

        # 7) Inpainting: choose single or multiple outputs based on mode
        outputs: List[Image.Image] = []
        prompts_used: List[str] = []
        # Persist the exact inputs that will be sent to inpainting
        try:
            if save_intermediate:
                safe_save(target_cut, os.path.join(inpaint_dir, "inpaint_input_image.png"))
                safe_save(inpaint_mask, os.path.join(inpaint_dir, "inpaint_input_mask.png"))
            try:
                # Build light-red overlay visualization for inpainting mask region
                base = target_cut.convert("RGBA")
                mask_bin = binary(inpaint_mask, thr=1).convert("L") if inpaint_mask is not None else None
                if mask_bin is not None:
                    overlay = Image.new("RGBA", base.size, (0,0,0,0))
                    light_red = (255, 102, 102, 140)
                    color_layer = Image.new("RGBA", base.size, light_red)
                    overlay.paste(color_layer, (0, 0), mask_bin)
                    vis = base.copy()
                    vis.alpha_composite(overlay)
                    if save_intermediate:
                        safe_save(vis.convert("RGB"), os.path.join(inpaint_dir, "inpaint_input_overlay.png"))
            except Exception:
                pass
        except Exception:
            pass

        def _restore_crop(img: Image.Image) -> Image.Image:
            if not restore_square_crop:
                return img
            if not (square_offsets and any(square_offsets)):
                return img
            l, t, r, b = square_offsets
            w, h = img.size
            # Crop: (left, top, right, bottom)
            # The padding was added as: left, top, right, bottom
            # So we crop: (l, t, w - r, h - b)
            return img.crop((l, t, w - r, h - b))

        output_meta: List[Dict[str, Any]] = []  # Track saved path and category for ranking
        if generation_mode == "probabilistic" and hypo_json_raw:
            # Parse JSON (robust)
            def _extract_json_object(s: str) -> dict:
                s = s.strip()
                if s.startswith("```"):
                    try:
                        first = s.find("```")
                        rest = s[first+3:]
                        nl = rest.find("\n")
                        if nl != -1:
                            rest = rest[nl+1:]
                        end = rest.find("```")
                        if end != -1:
                            s = rest[:end]
                    except Exception:
                        pass
                start_idxs = [i for i, ch in enumerate(s) if ch == '{']
                for start in start_idxs:
                    depth = 0
                    for i in range(start, len(s)):
                        if s[i] == '{':
                            depth += 1
                        elif s[i] == '}':
                            depth -= 1
                            if depth == 0:
                                frag = s[start:i+1]
                                try:
                                    import json as _json
                                    return _json.loads(frag)
                                except Exception:
                                    break
                try:
                    import json as _json
                    return _json.loads(s)
                except Exception:
                    return {}

            obj = _extract_json_object(str(hypo_json_raw))
            hypos = []
            if isinstance(obj, dict):
                hypos = obj.get("hypotheses") or []
            # Filter and sort
            cleaned = []
            for h in hypos:
                try:
                    dsc = str(h.get("description", "")).strip()
                    prob = float(h.get("probability", 0))
                    if not dsc:
                        continue
                    if prob <= 0:
                        continue
                    # Enforce: There is only one complete <target>. Pure white background.
                    dsc = _ensure_constraints(dsc, (occluded_object or seg_text))
                    cleaned.append({"description": dsc, "probability": prob})
                except Exception:
                    continue
            # Normalize probabilities and sort descending
            total_p = sum([c["probability"] for c in cleaned]) or 0.0
            if total_p > 0:
                for c in cleaned:
                    c["probability"] = c["probability"] / total_p
            cleaned.sort(key=lambda x: x["probability"], reverse=True)
            # Persist parsed hypotheses
            try:
                if save_intermediate:
                    with open(os.path.join(initial_dir, "object_hypotheses_parsed.json"), "w", encoding="utf-8") as f:
                        import json as _json
                        _json.dump(cleaned, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
            # First: generate base simple result(s) according to num_inpaint_images
            outputs = []
            prompts_used = []
            gidx = 0
            try:
                n_simple = 1
                try:
                    n_simple = int(num_inpaint_images or 1)
                except Exception:
                    n_simple = 1
                if n_simple > 1:
                    outs_simple = self.ip.inpaint_many(target_cut, inpaint_mask, desc or "", num_images=n_simple)
                    for sidx, img in enumerate(outs_simple or []):
                        img = _restore_crop(img)
                        outputs.append(img)
                        prompts_used.append(desc or "")
                        try:
                            out_path = os.path.join(simple_dir, f"{sidx:03d}.png")
                            if save_intermediate:
                                safe_save(img, out_path)
                            output_meta.append({"category": "simple", "path": out_path, "index": sidx})
                        except Exception:
                            pass
                        gidx += 1
                else:
                    out_simple = self.ip.inpaint(target_cut, inpaint_mask, desc or "")
                    out_simple = _restore_crop(out_simple)
                    outputs.append(out_simple)
                    prompts_used.append(desc or "")
                    # try:
                    #     if save_intermediate:
                    #         safe_save(out_simple, os.path.join(debug_dir, "final_out.png"))
                    # except Exception:
                    #     pass
                    try:
                        # also store under simple folder as 000.png
                        out_path = os.path.join(simple_dir, "000.png")
                        if save_intermediate:
                            safe_save(out_simple, out_path)
                        output_meta.append({"category": "simple", "path": out_path, "index": 0})
                    except Exception:
                        pass
                    gidx += 1
            except Exception:
                pass

            # Then: Generate inpainted images for each hypothesis (respect num_inpaint_images)
            for hidx, h in enumerate(cleaned):
                dsc = h["description"]
                try:
                    n_imgs = 1
                    try:
                        n_imgs = int(num_inpaint_images or 1)
                    except Exception:
                        n_imgs = 1
                    if n_imgs > 1:
                        outs_h = self.ip.inpaint_many(target_cut, inpaint_mask, dsc, num_images=n_imgs)
                        # per-hypothesis subfolder
                        prob_dir = os.path.join(inpaint_dir, f"probabilistic{hidx}")
                        if save_intermediate:
                            os.makedirs(prob_dir, exist_ok=True)
                        for vidx, img in enumerate(outs_h or []):
                            img = _restore_crop(img)
                            outputs.append(img)
                            prompts_used.append(dsc)
                            try:
                                out_path = os.path.join(prob_dir, f"{vidx:03d}.png")
                                if save_intermediate:
                                    safe_save(img, out_path)
                                output_meta.append({"category": f"probabilistic{hidx}", "path": out_path, "index": vidx})
                            except Exception:
                                pass
                            gidx += 1
                    else:
                        out = self.ip.inpaint(target_cut, inpaint_mask, dsc)
                        out = _restore_crop(out)
                        outputs.append(out)
                        prompts_used.append(dsc)
                        prob_dir = os.path.join(inpaint_dir, f"probabilistic{hidx}")
                        if save_intermediate:
                            os.makedirs(prob_dir, exist_ok=True)
                        try:
                            out_path = os.path.join(prob_dir, "000.png")
                            if save_intermediate:
                                safe_save(out, out_path)
                            output_meta.append({"category": f"probabilistic{hidx}", "path": out_path, "index": 0})
                        except Exception:
                            pass
                        gidx += 1
                except Exception:
                    continue
            # Fallback: if no successful outputs, run original description path once
            if not outputs:
                fallback_desc = prompt or self.gpt.gen_inpaint_prompt_from_image(image, seg_text, prompt_type="prompt") or ""
                # Enforce: There is only one complete <target>. Pure white background.
                fallback_desc = _ensure_constraints(fallback_desc, (occluded_object or seg_text))
                try:
                    n_imgs = 1
                    try:
                        n_imgs = int(num_inpaint_images or 1)
                    except Exception:
                        n_imgs = 1
                    if n_imgs > 1:
                        outs_fb = self.ip.inpaint_many(target_cut, inpaint_mask, fallback_desc, num_images=n_imgs)
                        for img in (outs_fb or []):
                            img = _restore_crop(img)
                            outputs.append(img)
                            prompts_used.append(fallback_desc)
                        if outputs:
                            # try:
                            #     if save_intermediate:
                            #         safe_save(outputs[0], os.path.join(debug_dir, "final_out.png"))
                            # except Exception:
                            #     pass
                            try:
                                if save_intermediate:
                                    safe_save(outputs[0], os.path.join(inpaint_dir, "final_out_fallback.png"))
                            except Exception:
                                pass
                    else:
                        out = self.ip.inpaint(target_cut, inpaint_mask, fallback_desc)
                        out = _restore_crop(out)
                        outputs.append(out)
                        prompts_used.append(fallback_desc)
                        # if save_intermediate:
                        #     safe_save(out, os.path.join(debug_dir, "final_out.png"))
                        try:
                            if save_intermediate:
                                safe_save(out, os.path.join(inpaint_dir, "final_out_fallback.png"))
                        except Exception:
                            pass
                except Exception:
                    pass
        else:
            # simple: single or multiple images
            n_imgs = 1
            try:
                n_imgs = int(num_inpaint_images or 1)
            except Exception:
                n_imgs = 1
            if n_imgs > 1:
                outs = self.ip.inpaint_many(target_cut, inpaint_mask, desc or "", num_images=n_imgs)
                outputs = []
                for img in (outs or []):
                    outputs.append(_restore_crop(img))
                if not outputs:
                    final_out = self.ip.inpaint(target_cut, inpaint_mask, desc or "")
                    outputs = [_restore_crop(final_out)]
                prompts_used = [(desc or "")] * len(outputs)
                # Save primary
                # try:
                #     if save_intermediate:
                #         safe_save(outputs[0], os.path.join(debug_dir, "final_out.png"))
                # except Exception:
                #     pass
                # Save all to simple subfolder
                for idx, img in enumerate(outputs):
                    try:
                        out_path = os.path.join(simple_dir, f"{idx:03d}.png")
                        if save_intermediate:
                            safe_save(img, out_path)
                        output_meta.append({"category": "simple", "path": out_path, "index": idx})
                    except Exception:
                        pass
            else:
                final_out = self.ip.inpaint(target_cut, inpaint_mask, desc or "")
                final_out = _restore_crop(final_out)
                outputs = [final_out]
                prompts_used = [desc or ""]
                # if save_intermediate:
                #     safe_save(final_out, os.path.join(debug_dir, "final_out.png"))
                try:
                    out_path = os.path.join(simple_dir, "000.png")
                    if save_intermediate:
                        safe_save(final_out, out_path)
                    output_meta.append({"category": "simple", "path": out_path, "index": 0})
                except Exception:
                    pass

        # Note: All images are now organized under inpaint/simple and inpaint/probabilistic{h}

        # 8) Post-inpainting segmentation for each output
        final_instance_mask = None
        final_instance_masks: List[Image.Image] = []
        # Track per-output segmented object file path in the same order as outputs
        per_output_segmented_paths: List[Optional[str]] = []
        try:
            target_cls = (occluded_object or seg_text)
            for vidx, out_img in enumerate(outputs):
                if out_img is None:
                    final_instance_masks.append(None)
                    per_output_segmented_paths.append(None)
                    continue

                # Determine sub-directory based on output_meta
                meta = output_meta[vidx] if vidx < len(output_meta) else {}
                category = meta.get("category", "simple")
                cat_idx = meta.get("index", 0)
                sub_seg_dir = os.path.join(seg_dir, category)
                if save_intermediate:
                    os.makedirs(sub_seg_dir, exist_ok=True)

                # Visible region mask aligned to this output
                vis_mask_canvas = None
                if target_mask is not None:
                    tb0 = binary(target_mask, thr=mask_thr)
                    if out_img.size != tb0.size:
                        try:
                            from PIL import Image as PILImage
                            left_off, top_off, _, _ = pad_offsets if 'pad_offsets' in locals() else (0,0,0,0)
                            # Superimpose square padding offsets (if applied and not restored crop)
                            if not restore_square_crop and 'square_offsets' in locals() and isinstance(square_offsets, tuple):
                                try:
                                    l2, t2, _, _ = square_offsets
                                    left_off += int(l2)
                                    top_off += int(t2)
                                except Exception:
                                    pass
                            canvas = PILImage.new("L", out_img.size, 0)
                            canvas.paste(tb0, (left_off, top_off))
                            vis_mask_canvas = canvas
                        except Exception:
                            vis_mask_canvas = tb0.resize(out_img.size, resample=Image.NEAREST)
                    else:
                        vis_mask_canvas = tb0
                seg_post = self.seg.segment(out_img, text=target_cls, extra={"class_names": [target_cls], "return_masks": True})
                payload_post = (seg_post or {}).get("payload") or {}
                dets_fname = f"post_detections_{cat_idx:03d}.json"
                try:
                    if save_intermediate:
                        with open(os.path.join(sub_seg_dir, dets_fname), "w", encoding="utf-8") as f2:
                            json.dump(payload_post, f2, ensure_ascii=False, indent=2)
                except Exception:
                    pass
                dets_post = payload_post.get("detections") or []
                masks_post: List[Image.Image] = []
                tkey = str(target_cls).strip().lower()
                for d in dets_post:
                    cls = str(d.get("class_name", "")).strip().lower()
                    if cls and cls != tkey and tkey not in cls:
                        continue
                    m = d.get("mask") or d.get("mask_base64") or d.get("mask_png_base64") or d.get("mask_png")
                    if isinstance(m, str) and m:
                        import base64, io
                        if m.startswith("data:"):
                            m = m.split(",", 1)[1]
                        try:
                            mask_img = Image.open(io.BytesIO(base64.b64decode(m))).convert("L")
                            masks_post.append(mask_img)
                        except Exception:
                            pass
                post_mask = None
                if masks_post:
                    from ..utils.mask_adapter import merge_masks
                    post_mask = merge_masks(masks_post)
                    # try:
                    #     if save_intermediate:
                    #         safe_save(post_mask, os.path.join(sub_seg_dir, f"post_target_mask_raw_{cat_idx:03d}.png"))
                    # except Exception:
                    #     pass
                    post_mask = binary(post_mask, thr=mask_thr)
                if post_mask is not None and vis_mask_canvas is not None:
                    connected = keep_components_connected_to(post_mask, vis_mask_canvas, connectivity=4)
                    if connected.getbbox() is None:
                        cur_mask = vis_mask_canvas
                    else:
                        try:
                            import numpy as _np
                            vis_arr = (_np.array(vis_mask_canvas.convert("L")) > 0)
                            con_arr = (_np.array(connected.convert("L")) > 0)
                            vis_area = int(vis_arr.sum())
                            inter = int((vis_arr & con_arr).sum())
                            thr = kwargs.get("final_min_vis_overlap", 0.5)
                            try:
                                thr = float(thr)
                            except Exception:
                                thr = 0.5
                            if vis_area > 0 and (inter / float(vis_area)) < max(0.0, min(1.0, thr)):
                                cur_mask = vis_mask_canvas
                            else:
                                cur_mask = connected
                        except Exception:
                            cur_mask = connected
                elif post_mask is not None:
                    cur_mask = post_mask
                else:
                    cur_mask = vis_mask_canvas
                if cur_mask is not None and vis_mask_canvas is not None:
                    try:
                        import numpy as np
                        arr_final = (np.array(cur_mask.convert("L")) > 0)
                        arr_vis = (np.array(vis_mask_canvas.convert("L")) > 0)
                        arr_union = (arr_final | arr_vis).astype("uint8") * 255
                        cur_mask = Image.fromarray(arr_union, mode="L")
                    except Exception:
                        pass
                # Save artifacts
                if cur_mask is not None:
                    # try:
                    #     raw_name = f"final_instance_mask_raw_{cat_idx:03d}.png"
                    #     if save_intermediate:
                    #         safe_save(cur_mask, os.path.join(sub_seg_dir, raw_name))
                    # except Exception:
                    #     pass
                    mask_fname = f"final_instance_mask_{cat_idx:03d}.png"
                    overlay_fname = f"final_instance_overlay_{cat_idx:03d}.png"
                    # try:
                    #     if save_intermediate:
                    #         safe_save(cur_mask, os.path.join(sub_seg_dir, mask_fname))
                    # except Exception:
                    #     pass
                    # try:
                    #     overlay_img = combine_with(out_img, cur_mask)
                    #     if save_intermediate:
                    #         safe_save(overlay_img, os.path.join(sub_seg_dir, overlay_fname))
                    # except Exception:
                    #     pass
                    try:
                        obj_cut = cutout_with_white(out_img, cur_mask)
                        obj_fname = f"final_object_{cat_idx:03d}.png"
                        obj_crop_fname = f"final_object_crop_{cat_idx:03d}.png"
                        try:
                            if save_intermediate:
                                safe_save(obj_cut, os.path.join(sub_seg_dir, obj_fname))
                            if vidx == 0:
                                if save_intermediate:
                                    safe_save(obj_cut, os.path.join(debug_dir, "final_object.png"))
                            per_output_segmented_paths.append(os.path.join(sub_seg_dir, obj_fname))
                        except Exception:
                            per_output_segmented_paths.append(None)
                            pass
                        bb = cur_mask.getbbox()
                        if bb is not None:
                            try:
                                obj_crop = obj_cut.crop(bb)
                                # if save_intermediate:
                                #     safe_save(obj_crop, os.path.join(sub_seg_dir, obj_crop_fname))
                            except Exception:
                                pass
                    except Exception:
                        pass
                final_instance_masks.append(cur_mask)
            final_instance_mask = final_instance_masks[0] if final_instance_masks else None
        except Exception:
            final_instance_mask = None
            final_instance_masks = []
            per_output_segmented_paths = [None] * len(outputs)

        # 9) Integrated ranking: per-category then global
        ranking_result = None
        if enable_ranking and outputs:
            try:
                # Build category -> list of candidate paths (prefer segmented, fallback to original)
                from collections import defaultdict
                cat_map = defaultdict(list)
                for idx, meta in enumerate(output_meta):
                    cat = meta.get("category") or "simple"
                    orig_path = meta.get("path")
                    seg_path = per_output_segmented_paths[idx] if idx < len(per_output_segmented_paths) else None
                    cat_map[cat].append(seg_path if seg_path and os.path.exists(seg_path) else orig_path)

                original_input_path = os.path.join(debug_dir, "original_input.png")
                adapter = GeminiRankingAdapter(model_name=str(kwargs.get("rank_model", os.getenv("GEMINI_RANK_MODEL", "gemini-2.5-flash"))),
                                               dry_run=bool(kwargs.get("rank_dry_run", False)))
                per_category = {}
                winners_order = []
                for cat in sorted(cat_map.keys()):
                    candidates = cat_map[cat]
                    if not candidates:
                        continue
                    if len(candidates) == 1:
                        res = {
                            "ranking": "A",
                            "candidate_assessments": {"A": "Only candidate; auto-selected."},
                            "reasoning": "Single candidate; no comparison needed.",
                            "mapping": {"A": candidates[0]},
                            "winner_image": candidates[0],
                            "winner_letter": "A",
                        }
                    else:
                        res = adapter.rank(original_input_path, candidates)
                        rank_str = (res.get("ranking") or "A").strip()
                        first = rank_str.split('>')[0].strip().split('=')[0].strip() or 'A'
                        winner = res.get("mapping", {}).get(first)
                        res["winner_image"] = winner
                        res["winner_letter"] = first
                    per_category[cat] = res
                    if res.get("winner_image"):
                        winners_order.append((cat, res["winner_image"]))
                        # Save snapshot per category
                        try:
                            import shutil
                            tgt = os.path.join(ranking_dir, f"best_{cat}.png")
                            if os.path.exists(res["winner_image"]):
                                if save_intermediate:
                                    shutil.copy(res["winner_image"], tgt)
                        except Exception:
                            pass

                # Global ranking across category winners
                if len(winners_order) <= 1:
                    if winners_order:
                        ranking_result = {
                            "per_category": per_category,
                            "global": {
                                "final_winner_category": winners_order[0][0],
                                "final_winner_image": winners_order[0][1],
                                "global_ranking_skipped": True,
                            },
                        }
                        try:
                            import shutil
                            if save_intermediate:
                                shutil.copy(winners_order[0][1], os.path.join(ranking_dir, "best_overall.png"))
                        except Exception:
                            pass
                    else:
                        ranking_result = {"per_category": per_category, "global": {"global_ranking_skipped": True}}
                else:
                    paths = [p for (_, p) in winners_order]
                    out = adapter.rank(original_input_path, paths)
                    rank_str = (out.get("ranking") or "A").strip()
                    first = rank_str.split('>')[0].strip().split('=')[0].strip() or 'A'
                    winner_path = out.get("mapping", {}).get(first)
                    # Map back to category
                    cat_by_path = {p: c for (c, p) in winners_order}
                    winner_cat = cat_by_path.get(winner_path)
                    out.update({
                        "final_winner_category": winner_cat,
                        "final_winner_image": winner_path,
                        "global_ranking_skipped": False,
                        "category_order": [c for (c, _) in winners_order],
                    })
                    ranking_result = {"per_category": per_category, "global": out}
                    try:
                        import shutil
                        if winner_path and os.path.exists(winner_path):
                            if save_intermediate:
                                shutil.copy(winner_path, os.path.join(ranking_dir, "best_overall.png"))
                    except Exception:
                        pass
                # Persist JSON
                try:
                    if save_intermediate:
                        with open(os.path.join(ranking_dir, "ranking_results.json"), "w", encoding="utf-8") as f:
                            json.dump(ranking_result, f, ensure_ascii=False, indent=2)
                except Exception:
                    pass
            except Exception:
                ranking_result = None

        # Return
        ret = {
            "prompt": prompts_used[0] if prompts_used else (desc or ""),
            "occluding_objects": occluding_names,
            "bbox": bbox_target,
            "target_mask": target_mask,
            "occluding_mask": occ_mask_bin,
            "output": outputs[0] if outputs else None,
        }
        if final_instance_mask is not None:
            ret["final_mask"] = final_instance_mask
        if generation_mode == "probabilistic" or (outputs and len(outputs) > 1):
            # try to load parsed hypos file if available
            hypos = None
            try:
                pth = os.path.join(initial_dir, "object_hypotheses_parsed.json")
                if os.path.exists(pth):
                    with open(pth, "r", encoding="utf-8") as f:
                        hypos = json.load(f)
            except Exception:
                hypos = None
            ret.update({
                "outputs": outputs,
                "prompts": prompts_used,
                "hypotheses": hypos,
            })
            if final_instance_masks:
                ret["final_masks"] = final_instance_masks
        if ranking_result is not None:
            ret["ranking"] = ranking_result
        return ret
