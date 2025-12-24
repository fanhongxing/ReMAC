import argparse
import os
import sys
from typing import List, Tuple

from PIL import Image

# Allow running as a script from repo root
ROOT = os.path.dirname(__file__)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from multi_agent.pipeline.orchestrator import Orchestrator


def _configure_llm_models(general_choice: str, check_choice: str, rank_choice: str | None = None) -> dict:
    """Configure model ids for general, check, and ranking agents, with optional SiliconFlow Qwen routing."""
    general_map = {
        "gpt4o": "openai/gpt-4o",
        "qwen": "Qwen/Qwen3-VL-32B-Instruct",
    }
    check_map = {
        "gemini": "google/gemini-2.5-pro",
        "qwen": "Qwen/Qwen3-VL-32B-Thinking",
    }
    rank_map = {
        "gemini_flash": "gemini-2.5-flash",
        "qwen": "Qwen/Qwen3-VL-32B-Instruct",
    }
    selected: dict = {}
    if general_choice in general_map:
        selected["general"] = general_map[general_choice]
        os.environ["OPENROUTER_MODEL_GENERAL"] = selected["general"]
    if check_choice in check_map:
        selected["check"] = check_map[check_choice]
        os.environ["OPENROUTER_MODEL_CHECK"] = selected["check"]
    if rank_choice and rank_choice in rank_map:
        selected["rank"] = rank_map[rank_choice]
        os.environ["GEMINI_RANK_MODEL"] = selected["rank"]
    
    # Note: We no longer force OPENROUTER_BASE_URL to SiliconFlow here.
    # The GPTAdapter has been updated to handle SiliconFlow routing for Qwen models independently.
    
    return selected


def is_image_file(name: str) -> bool:
    ext = os.path.splitext(name)[1].lower()
    return ext in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_target_from_filename(filename: str) -> str:
    """
    Parse occluded target class from filename.
    Supports:
      - COCO_test2014_000000004996_elephant.jpg -> elephant
      - activity-8122959_640_spraypaintcan.jpg -> spraypaintcan
      - 000000115_violin.jpg -> violin
      - 107911_truck.jpg -> truck
      - aj-elgammal-VwFiIznMVeA-unsplash_tennisball.jpg -> tennisball
    Strategy: take basename without extension, split by '_' and use the last token.
    """
    stem = os.path.splitext(os.path.basename(filename))[0]
    if "_" in stem:
        return stem.split("_")[-1]
    return stem


def list_images(src_dir: str) -> List[str]:
    items: List[str] = []
    for entry in sorted(os.listdir(src_dir)):
        p = os.path.join(src_dir, entry)
        if os.path.isfile(p) and is_image_file(entry):
            items.append(p)
    return items


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def process_one(
    image_path: str,
    out_root: str,
    seg_text: str,
    *,
    prompt: str | None,
    generation_mode: str,
    num_images: int,
    mask_thr: int,
    invert_mask: bool,
    dilate_k: int,
    dilate_iters: int,
    boundary_mode: str,
    overwrite: bool,
    dry_run: bool,
    edge_grow_px: int | None,
    edge_grow_ratio: float | None,
    seg_backend: str | None,
    seg_url: str | None,
    occluder_overlap_thr: float | None,
    rank_model: str | None,
    enable_boundary_analysis: bool = True,
    restore_square_crop: bool = False,
    save_intermediate: bool = False,
) -> Tuple[str, bool, str | None]:
    """
    Run pipeline on a single image.
    Returns: (debug_dir, skipped, error_message)
    """
    base = os.path.splitext(os.path.basename(image_path))[0]
    debug_dir = os.path.join(out_root, base)
    if not overwrite and os.path.exists(debug_dir):
        return debug_dir, True, None

    ensure_dir(debug_dir)
    final_path = os.path.join(debug_dir, "final_out.png")
    if not overwrite and os.path.exists(final_path):
        return debug_dir, True, None

    try:
        # init orchestrator per run to honor dry_run flag
        from multi_agent.adapters.seg_adapter import SegmentationAdapter
        from multi_agent.adapters.inpaint_adapter import InpaintAdapter
        from multi_agent.adapters.gpt_adapter import GPTAdapter

        seg = SegmentationAdapter(dry_run=dry_run, backend=seg_backend, base_url=seg_url)
        ip = InpaintAdapter(dry_run=dry_run)
        gpt = GPTAdapter()
        orch = Orchestrator(seg=seg, ip=ip, gpt=gpt)

        img = Image.open(image_path).convert("RGB")
        
        # If any side of the image is larger than 1024, compress it to within 1024 while maintaining the aspect ratio
        max_side = 1024
        w, h = img.size
        if w > max_side or h > max_side:
            scale = max_side / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)
            print(f"Image resized from {w}x{h} to {new_w}x{new_h}")
        # Build kwargs similar to main.py
        run_kwargs = {}
        # Normalize edge grow values like main.py
        if edge_grow_ratio is not None and edge_grow_ratio < 0:
            edge_grow_ratio = 0
        if edge_grow_px is not None and edge_grow_px < 0:
            edge_grow_px = 0
        # Pass either absolute px or ratio (px takes precedence)
        if edge_grow_px is not None:
            run_kwargs["extended_pad_dilate_px"] = int(edge_grow_px)
        elif edge_grow_ratio is not None:
            run_kwargs["edge_grow_ratio"] = float(edge_grow_ratio)
        # Only pass occluder overlap threshold when explicitly provided; otherwise use orchestrator default
        if occluder_overlap_thr is not None:
            run_kwargs["occluder_overlap_thr"] = float(occluder_overlap_thr)
        if rank_model is not None:
            run_kwargs["rank_model"] = str(rank_model)
        res = orch.run(
            img,
            seg_text=seg_text,
            prompt=prompt,
            generation_mode=generation_mode,
            num_inpaint_images=num_images,
            mask_thr=mask_thr,
            invert_mask=invert_mask,
            dilate_k=dilate_k,
            dilate_iters=dilate_iters,
            boundary_mode=boundary_mode,
            bbox_json_path=None,
            occluded_object=None,
            debug_dir=debug_dir,
            enable_boundary_analysis=enable_boundary_analysis,
            restore_square_crop=restore_square_crop,
            enable_ranking=bool(rank_model),
            save_intermediate=save_intermediate,
            **run_kwargs,
        )

        # Ensure final output is saved (Orchestrator only saves it if save_intermediate=True)
        if res.get("output"):
            res["output"].save(final_path)

        # Save/ensure final instance mask if provided by orchestrator, and write metadata.json
        try:
            final_mask_img = res.get("final_mask")
            final_mask_path = os.path.join(debug_dir, "seg", "simple", "final_instance_mask_000.png")
            # orchestrator saves mask to seg/simple/; do not duplicate at root
        except Exception:
            final_mask_path = None

        # Ensure final object images exist (full cutout and tight crop). If orchestrator saved them, reuse; else generate here.
        final_object_path = os.path.join(debug_dir, "seg", "simple", "final_object_000.png")
        final_object_crop_path = os.path.join(debug_dir, "seg", "simple", "final_object_crop_000.png")
        if save_intermediate:
            try:
                if not os.path.exists(final_object_path) and final_mask_img is not None:
                    from multi_agent.utils.mask_adapter import cutout_with_white
                    # Build cutout from output image and final mask
                    try:
                        out_img = res.get("output")
                        if out_img is None:
                            out_img = Image.open(os.path.join(debug_dir, "final_out.png")).convert("RGB")
                    except Exception:
                        out_img = None
                    if out_img is not None:
                        ensure_dir(os.path.dirname(final_object_path))
                        obj_cut = cutout_with_white(out_img, final_mask_img)
                        obj_cut.save(final_object_path)
                        bb = final_mask_img.getbbox()
                        if bb is not None and not os.path.exists(final_object_crop_path):
                            try:
                                obj_crop = obj_cut.crop(bb)
                                obj_crop.save(final_object_crop_path)
                            except Exception:
                                pass
            except Exception:
                pass

        # Orchestrator already saves final_out.png (root) and variants under variants/
        # Write metadata.json (under meta/) for both modes; include final mask path when available
        if save_intermediate:
            try:
                if generation_mode == "probabilistic":
                    import json, glob
                    variants_dir = os.path.join(debug_dir, "variants")
                    files = []
                    p0 = os.path.join(debug_dir, "final_out.png")
                    if os.path.exists(p0):
                        files.append(p0)
                    # variants from variants/
                    files.extend(sorted(glob.glob(os.path.join(variants_dir, "final_out_*.png"))))
                    meta = {
                        "files": files,
                        "prompts": res.get("prompts"),
                        "hypotheses": res.get("hypotheses"),
                        "final_mask": final_mask_path if final_mask_path and os.path.exists(final_mask_path) else None,
                        "final_object": final_object_path if os.path.exists(final_object_path) else None,
                        "final_object_crop": final_object_crop_path if os.path.exists(final_object_crop_path) else None,
                    }
                    with open(os.path.join(debug_dir, "meta", "metadata.json"), "w", encoding="utf-8") as f:
                        json.dump(meta, f, ensure_ascii=False, indent=2)
                else:
                    import json
                    # collect variants if present
                    files = []
                    try:
                        import glob
                        variants_dir = os.path.join(debug_dir, "variants")
                        p0 = os.path.join(debug_dir, "final_out.png")
                        if os.path.exists(p0):
                            files.append(p0)
                        files.extend(sorted(glob.glob(os.path.join(variants_dir, "final_out_*.png"))))
                    except Exception:
                        files = [os.path.join(debug_dir, "final_out.png")]
                    meta = {
                        "files": files,
                        "prompt": res.get("prompt"),
                        "bbox": res.get("bbox"),
                        "occluding_objects": res.get("occluding_objects"),
                        "final_mask": final_mask_path if final_mask_path and os.path.exists(final_mask_path) else None,
                        "final_object": final_object_path if os.path.exists(final_object_path) else None,
                        "final_object_crop": final_object_crop_path if os.path.exists(final_object_crop_path) else None,
                    }
                    if res.get("final_masks"):
                        # Collect indexed masks from seg directory
                        try:
                            seg_dir = os.path.join(debug_dir, "seg")
                            mask_files = []
                            for entry in sorted(os.listdir(seg_dir)):
                                if entry.startswith("final_instance_mask") and entry.endswith(".png"):
                                    mask_files.append(os.path.join(seg_dir, entry))
                            meta["final_masks"] = mask_files
                        except Exception:
                            pass
                    with open(os.path.join(debug_dir, "meta", "metadata.json"), "w", encoding="utf-8") as f:
                        json.dump(meta, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

        # Optionally mirror/copy input image for reference
        # Optionally mirror/copy input image for reference
        try:
            img.save(os.path.join(debug_dir, "input.jpg"))
        except Exception:
            pass

        return debug_dir, False, None
    except Exception as e:
        return debug_dir, False, str(e)


def main():
    parser = argparse.ArgumentParser(description="Batch process COCO test images with Multi-Agent pipeline.")
    parser.add_argument(
        "--src-dir",
        default="data/cocoa",
        help="source images directory",
    )
    parser.add_argument(
        "--dst-dir",
        default="results/cocoa",
        help="destination root directory; per-image subfolders will be created under this",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="optional inpaint prompt; if omitted, GPT generates it",
    )
    parser.add_argument(
        "--generation-mode",
        choices=["simple", "probabilistic"],
        default="simple",
        help="choose single description ('simple') or probabilistic multi-hypothesis generation",
    )
    parser.add_argument("--num-images", type=int, default=1, help="number of images to generate in simple mode (1-8)")
    parser.add_argument("--mask-thr", type=int, default=128, help="binarize threshold (0-255)")
    parser.add_argument("--invert-mask", action="store_true", help="invert mask before inpainting")
    parser.add_argument("--dilate-k", type=int, default=7, help="MaxFilter kernel size for dilation (odd >=3), 0 to skip")
    parser.add_argument("--dilate-iters", type=int, default=2, help="dilation iterations")
    parser.add_argument("--edge-grow-px", type=int, default=10, help="inward grow pixels for edge-extended band (0 to disable)")
    parser.add_argument("--edge-grow-ratio", type=float, default=None, help="ratio (0~1) of extension size used to compute inward grow px when --edge-grow-px not set")
    parser.add_argument("--occluder-overlap-thr", type=float, default=None, help="Override occluder overlap drop threshold (measured against occluder area). If omitted, use orchestrator default (0.8).")
    parser.add_argument(
        "--boundary-mode",
        choices=["boundary", "boundary_bbox"],
        default="boundary_bbox",
        help="boundary analysis prompt type",
    )
    parser.add_argument(
        "--disable-boundary-analysis",
        action="store_true",
        help="Disable boundary analysis agent. If disabled, image will not be extended by default.",
    )
    parser.add_argument(
        "--restore-square-crop",
        action="store_true",
        help="Restore inpainted image to original size (before square padding) by cropping.",
    )
    parser.add_argument("--overwrite", action="store_true", help="re-run even if final_out.png exists")
    parser.add_argument("--dry-run", action="store_true", help="use mock adapters when services are missing")
    parser.add_argument("--seg-backend", choices=["sam", "xsam"], default=None, help="choose segmentation backend: 'sam' or 'xsam'")
    parser.add_argument("--seg-url", default=None, help="override segmentation server base url")
    parser.add_argument("--reverse", action="store_true", help="Process images in reverse order.")
    parser.add_argument(
        "--llm-general",
        choices=["gpt4o", "qwen"],
        default="gpt4o",
        help="General vision model: gpt4o (default) or Qwen3-VL-32B-Instruct via SiliconFlow.",
    )
    parser.add_argument(
        "--llm-check",
        choices=["gemini", "qwen"],
        default="gemini",
        help="Check agent model: gemini-2.5-pro (default) or Qwen3-VL-32B-Thinking via SiliconFlow.",
    )
    parser.add_argument(
        "--llm-rank",
        choices=["gemini_flash", "qwen"],
        default=None,
        help="Ranking agent model: gemini-2.5-flash or Qwen3-VL-32B-Instruct via SiliconFlow. If not specified, ranking is disabled.",
    )
    parser.add_argument(
        "--save-intermediate",
        action="store_true",
        help="Save intermediate debug files (masks, json logs, overlays). Default is False.",
    )
    args = parser.parse_args()

    models = _configure_llm_models(args.llm_general, args.llm_check, args.llm_rank)

    src = args.src_dir
    dst = args.dst_dir
    ensure_dir(dst)

    files = list_images(src)
    if args.reverse:
        files.reverse()
    if not files:
        print(f"No images found in: {src}")
        return

    ok = 0
    skipped = 0
    failed = 0

    for idx, p in enumerate(files, 1):
        target = parse_target_from_filename(p)
        dbg, was_skipped, err = process_one(
            p,
            dst,
            seg_text=target,
            prompt=args.prompt,
            generation_mode=args.generation_mode,
            num_images=args.num_images,
            mask_thr=args.mask_thr,
            invert_mask=args.invert_mask,
            dilate_k=args.dilate_k,
            dilate_iters=args.dilate_iters,
            boundary_mode=args.boundary_mode,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            edge_grow_px=args.edge_grow_px if args.edge_grow_px and args.edge_grow_px>0 else None,
            edge_grow_ratio=args.edge_grow_ratio,
            seg_backend=args.seg_backend,
            seg_url=args.seg_url,
            occluder_overlap_thr=args.occluder_overlap_thr,
            rank_model=models.get("rank"),
            enable_boundary_analysis=not args.disable_boundary_analysis,
            restore_square_crop=args.restore_square_crop,
            save_intermediate=args.save_intermediate,
        )
        prefix = f"[{idx}/{len(files)}] {os.path.basename(p)} -> {dbg}"
        if was_skipped:
            skipped += 1
            print(prefix + " | skipped (exists)")
        elif err is None:
            ok += 1
            print(prefix + " | done")
        else:
            failed += 1
            print(prefix + f" | ERROR: {err}")

    print(f"Summary: ok={ok}, skipped={skipped}, failed={failed}, total={len(files)}")


if __name__ == "__main__":
    main()
