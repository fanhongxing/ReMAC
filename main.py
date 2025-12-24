import argparse
import os
from PIL import Image
from multi_agent.pipeline.orchestrator import Orchestrator


def _configure_llm_models(general_choice: str, check_choice: str, rank_choice: str | None = None) -> dict:
    """
    Configure model ids for the general vision agent, check agent, and ranking agent.
    Choices map to SiliconFlow Qwen replacements when requested.
    """
    general_map = {
        "gpt4o": "openai/gpt-4o",
        "qwen": "Qwen/Qwen3-VL-32B-Instruct",
    }
    check_map = {
        "gemini": "google/gemini-2.5-pro",
        "qwen": "Qwen/Qwen3-VL-32B-Thinking",
    }
    rank_map = {
        "gemini_flash": "google/gemini-2.5-flash",
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="input image path")
    parser.add_argument("--seg-text", dest="seg_text", required=True, help="segmentation prompt, e.g., 'red cup'")
    parser.add_argument("--prompt", default=None, help="inpaint prompt; if absent, GPT will help")
    parser.add_argument("--out", default=None, help="output image path")
    parser.add_argument("--dry-run", action="store_true", help="simulate adapters when services are missing")
    # segmentation backend selection
    parser.add_argument("--seg-backend", choices=["sam", "xsam"], default=None, help="choose segmentation backend: 'sam' (default) or 'xsam'")
    parser.add_argument("--seg-url", default=None, help="override segmentation server base url; e.g., http://127.0.0.1:8040 for sam, http://127.0.0.1:8042 for xsam")
    parser.add_argument("--generation-mode", choices=["simple", "probabilistic"], default="simple", help="choose single description ('simple') or probabilistic multi-hypothesis generation")
    parser.add_argument("--num-images", type=int, default=1, help="number of images to generate in simple mode (1-8)")
    parser.add_argument("--multi-out-dir", default=None, help="directory to save multiple outputs in probabilistic mode; defaults next to --out")
    # mask options
    parser.add_argument("--mask-thr", type=int, default=128, help="binarize threshold (0-255)")
    parser.add_argument("--invert-mask", action="store_true", help="invert mask before inpainting")
    parser.add_argument("--dilate-k", type=int, default=0, help="MaxFilter kernel size for dilation (odd >=3), 0 to skip")
    parser.add_argument("--dilate-iters", type=int, default=1, help="dilation iterations")
    # edge extension inward grow controls
    parser.add_argument("--edge-grow-ratio", type=float, default=None, help="Edge extension inward-grow ratio (relative to padding extent). For example, 0.15 means add 15% of the extension width inward.")
    parser.add_argument("--edge-grow-px", type=int, default=None, help="Edge extension inward-grow pixels (absolute). Takes precedence over the ratio if provided.")
    # boundary mode selector
    parser.add_argument("--boundary-mode", choices=["boundary", "boundary_bbox"], default="boundary_bbox", help="boundary analysis prompt type")
    parser.add_argument("--disable-boundary-analysis", action="store_true", help="Disable boundary analysis agent. If disabled, image will not be extended by default.")
    parser.add_argument("--restore-square-crop", action="store_true", help="Restore inpainted image to original size (before square padding) by cropping.")
    parser.add_argument("--bbox-json-path", default=None, help="detections json path for boundary_bbox")
    parser.add_argument("--occluded-object", default=None, help="object name used for boundary_bbox analysis, e.g., 'bird'")
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
    parser.add_argument("--save-intermediate", action="store_true", help="Save intermediate debug files (masks, json logs, etc). Default is False.")
    args = parser.parse_args()

    models = _configure_llm_models(args.llm_general, args.llm_check, args.llm_rank)

    # Allow toggling dry-run via env or arg: we recreate adapters accordingly
    from multi_agent.adapters.seg_adapter import SegmentationAdapter
    from multi_agent.adapters.inpaint_adapter import InpaintAdapter
    from multi_agent.adapters.gpt_adapter import GPTAdapter

    seg = SegmentationAdapter(dry_run=args.dry_run, backend=args.seg_backend, base_url=args.seg_url)
    ip = InpaintAdapter(dry_run=args.dry_run)
    gpt = GPTAdapter()

    orch = Orchestrator(seg=seg, ip=ip, gpt=gpt)

    image = Image.open(args.image).convert("RGB")
    
    # If any side of the image is larger than 1024, compress it to within 1024 while maintaining the aspect ratio
    max_side = 1024
    w, h = image.size
    if w > max_side or h > max_side:
        scale = max_side / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        image = image.resize((new_w, new_h), Image.LANCZOS)
        print(f"Image resized from {w}x{h} to {new_w}x{new_h}")

    # We only know pad_amount after boundary analysis, so we cannot directly compute the ratio here.
    # Strategy: pass-through either the absolute pixels or the ratio into orchestrator via kwargs.
    # If ratio is provided (and px not), orchestrator will compute the final inward-grow pixels after parsing pad_amount.

    extra_kwargs = {}
    if args.edge_grow_ratio is not None and args.edge_grow_ratio < 0:
        args.edge_grow_ratio = 0
    if args.edge_grow_px is not None and args.edge_grow_px < 0:
        args.edge_grow_px = 0
    if args.edge_grow_px is not None:
        extra_kwargs["extended_pad_dilate_px"] = args.edge_grow_px
    elif args.edge_grow_ratio is not None:
        # Record the ratio; the actual pixels will be computed inside orchestrator after boundary parsing
        extra_kwargs["edge_grow_ratio"] = args.edge_grow_ratio


    if models.get("rank"):
        extra_kwargs["rank_model"] = models["rank"]

    result = orch.run(
        image,
        seg_text=args.seg_text,
        prompt=args.prompt,
        generation_mode=args.generation_mode,
        num_inpaint_images=args.num_images,
        mask_thr=args.mask_thr,
        invert_mask=args.invert_mask,
        dilate_k=args.dilate_k,
        dilate_iters=args.dilate_iters,
        boundary_mode=args.boundary_mode,
        bbox_json_path=args.bbox_json_path,
        occluded_object=args.occluded_object,
        enable_boundary_analysis=not args.disable_boundary_analysis,
        restore_square_crop=args.restore_square_crop,
        enable_ranking=bool(models.get("rank")),
        save_intermediate=args.save_intermediate,
        **extra_kwargs,
    )
    # Always save the primary output
    if args.out:
        result["output"].save(args.out)

    # Optional: Only save a copy of all outputs when --multi-out-dir is explicitly provided, to avoid duplication with debug/inpaint
    if args.multi_out_dir and result.get("outputs"):
        import os, json
        outs = result.get("outputs") or []
        prompts = result.get("prompts") or []
        hypos = result.get("hypotheses") or []
        # decide target dir
        out_dir = args.multi_out_dir
        os.makedirs(out_dir, exist_ok=True)
        saved_files = []
        for i, img in enumerate(outs):
            fn = os.path.join(out_dir, f"{i:02d}.png")
            img.save(fn)
            saved_files.append(fn)
        # write metadata
        meta = {
            "files": saved_files,
            "prompts": prompts,
            "hypotheses": hypos,
        }
        # Add final_masks if present
        if result.get("final_masks"):
            try:
                # We don't serialize image objects; presence indicated by mask file paths
                mask_files = []
                for i in range(len(saved_files)):
                    # match naming from orchestrator
                    if i == 0:
                        mf = os.path.join(os.path.dirname(args.out), "debug", "seg", "final_instance_mask.png")  # may not exist
                    # Simpler: rely on variants dir listing
                # Instead, build from seg directory directly
                seg_dir = os.path.join(os.path.dirname(args.out) or ".", "debug", "seg")
                if os.path.isdir(seg_dir):
                    for root, dirs, files in os.walk(seg_dir):
                        for entry in sorted(files):
                            if entry.startswith("final_instance_mask") and entry.endswith(".png"):
                                mask_files.append(os.path.join(root, entry))
                meta["final_masks"] = mask_files
            except Exception:
                pass
        with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
