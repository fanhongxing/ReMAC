import io
import os
import time
import json
import base64
import torch
import uvicorn
import numpy as np

from pathlib import Path
from typing import Optional, Any, List, Dict
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse

# Depends on X-SAM code
# Requires setting PYTHONPATH to point to the X-SAM source directory before running, for example:
# export PYTHONPATH=/data/zhaoshuyu/X-SAM/X-SAM-main/xsam:$PYTHONPATH

from mmengine.config import Config
from xsam.demo.demo import XSamDemo  # type: ignore
from xsam.utils.logging import print_log  # type: ignore

"""
Startup method (example):
conda activate xsam
export PYTHONPATH=/data/zhaoshuyu/X-SAM/X-SAM-main/xsam:$PYTHONPATH
export TRANSFORMERS_OFFLINE=0
python xsam_api_server.py \
  --config /data/zhaoshuyu/X-SAM/X-SAM-main/xsam/xsam/configs/xsam/phi3_mini_4k_instruct_siglip2_so400m_p14_384/s3_mixed_finetune/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_large_m2f_gpu16_mixed_finetune.py \
  --workdir /data/zhaoshuyu/X-SAM/X-SAM-main/inits/X-SAM/s3_mixed_finetune/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_large_m2f_gpu16_mixed_finetune \
  --ckpt latest
Default listens on 0.0.0.0:8062
"""

# ---------------------------
# Config via env (can be overridden by command line)
# ---------------------------
CONFIG_PATH = os.getenv("XSAM_CONFIG_PATH", "")
WORKDIR = os.getenv("XSAM_WORKDIR", "")
CKPT = os.getenv("XSAM_CKPT", "latest")

app = FastAPI(title="X-SAM API", version="1.0")
_demo: Optional[XSamDemo] = None


# ---------------------------
# Utility functions
# ---------------------------

def _b64_png_from_array(arr: np.ndarray) -> str:
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _get_bounding_box_from_mask(mask: np.ndarray) -> List[float]:
    """Get bounding box coordinates [x1, y1, x2, y2] from mask"""
    if len(mask.shape) == 3:
        if mask.shape[2] == 1:
            mask = mask.squeeze(2)
        else:
            mask = mask[0]

    if mask.dtype != np.uint8:
        mask = (mask > 0.5).astype(np.uint8)

    if mask.sum() == 0:
        return [0.0, 0.0, 0.0, 0.0]

    rows = np.any(mask > 0, axis=1)
    cols = np.any(mask > 0, axis=0)

    if not np.any(rows) or not np.any(cols):
        return [0.0, 0.0, 0.0, 0.0]

    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return [float(x1), float(y1), float(x2 + 1), float(y2 + 1)]


def _load_demo(config_path: str, workdir: str, ckpt: str) -> XSamDemo:
    cfg = Config.fromfile(config_path)

    if not os.path.isabs(ckpt):
        ckpt_path = os.path.join(workdir, ckpt)
    else:
        ckpt_path = ckpt

    real_path = os.path.realpath(ckpt_path)
    if not os.path.exists(real_path):
        raise RuntimeError(f"Checkpoint file not found: {ckpt_path} -> {real_path}")

    print_log(f"Loading checkpoint: {ckpt_path} -> {real_path}", logger="current")
    print_log("Initializing X-SAM demo for API...", logger="current")

    demo = XSamDemo(cfg, real_path, output_ids_with_output=False, work_dir=workdir)

    if torch.cuda.is_available():
        demo.model = demo.model.cuda()
        if hasattr(demo.model, 'llm'):
            demo.model.llm = demo.model.llm.cuda()
        if hasattr(demo.model, 'sam'):
            demo.model.sam = demo.model.sam.cuda()
        print_log("Model successfully moved to CUDA device", logger="current")
    else:
        print_log("Warning: No CUDA device detected, running on CPU", logger="current")

    print_log("X-SAM demo (API) initialization complete!", logger="current")
    return demo


# ---------------------------
# FastAPI startup event
# ---------------------------

@app.on_event("startup")
def _on_startup():
    global _demo
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=CONFIG_PATH)
    parser.add_argument("--workdir", type=str, default=WORKDIR)
    parser.add_argument("--ckpt", type=str, default=CKPT)
    args, _ = parser.parse_known_args()

    if not args.config or not os.path.isfile(args.config):
        raise RuntimeError("XSAM config file not found. Set --config or XSAM_CONFIG_PATH")
    if not args.workdir or not os.path.isdir(args.workdir):
        raise RuntimeError("XSAM workdir not found. Set --workdir or XSAM_WORKDIR")

    _demo = _load_demo(args.config, args.workdir, args.ckpt)


@app.get("/healthz")
def healthz():
    return {"status": "ok", "has_demo": _demo is not None}


# ---------------------------
# /predict interface
# ---------------------------

@app.post("/predict")
async def predict(
    image: UploadFile = File(..., description="RGB image file"),
    task_name: str = Form("imgconv"),
    prompt: str = Form(""),
    score_thr: float = Form(0.5),
):
    try:
        if _demo is None:
            raise HTTPException(status_code=503, detail="Model not initialized")

        print_log(f"Received request - task_name: {task_name}", logger="current")
        print_log(f"prompt: {prompt}", logger="current")
        print_log(f"score_thr: {score_thr}", logger="current")

        # Read image
        data = await image.read()
        pil = Image.open(io.BytesIO(data)).convert("RGB")
        print_log(f"Image size: {pil.size}", logger="current")

        # Parse prompt
        parsed_prompt: Any = prompt
        if isinstance(prompt, str):
            stripped = prompt.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                try:
                    candidate = json.loads(stripped)
                    if isinstance(candidate, list):
                        parsed_prompt = candidate
                        print_log("Detected JSON list format prompt, parsed as list", logger="current")
                except Exception:
                    pass

        # refseg mode
        if task_name == "refseg" and isinstance(parsed_prompt, list):
            print_log(f"Processing refseg task, prompt list: {parsed_prompt}", logger="current")

            all_detections = []
            for i, single_prompt in enumerate(parsed_prompt):
                try:
                    print_log(f"Processing object {i + 1}: {single_prompt}", logger="current")
                    
                    # Use tensor mode to get real mask data
                    print_log(f"Using tensor mode to get real mask data...", logger="current")
                    
                    # Temporarily set output_ids_with_output to True to get tensor data
                    original_output_ids = _demo.output_ids_with_output
                    _demo.output_ids_with_output = True
                    
                    try:
                        # Directly call the model's forward method to get raw data
                        data_dict = {"pil_image": pil, "vprompt_masks": None, "task_name": task_name}
                        classes, task_name_postprocess = _demo._get_classes_from_prompt(single_prompt, task_name)
                        _demo.model.postprocess_fn = _demo.postprocess_fns[task_name_postprocess]
                        _demo._set_metadata(task_name, classes)
                        data_dict.update(_demo._process_prompt(single_prompt, task_name, classes))
                        data_dict.update(_demo._process_image(pil))
                        data_dict.update(_demo._process_data_dict(data_dict))
                        data_dict, data_samples = _demo._process_input_dict(data_dict)
                        
                        metadata = _demo.metadata
                        
                        with torch.no_grad():
                            llm_outputs, seg_outputs = _demo.model(
                                data_dict,
                                data_samples,
                                mode="tensor",
                                metadata=metadata,
                                generation_config=_demo.generation_config,
                                stopping_criteria=_demo.stop_criteria,
                                do_postprocess=True,
                                do_loss=False,
                                threshold=score_thr,
                            )
                        
                        print_log(f"LLM output type: {type(llm_outputs)}", logger="current")
                        print_log(f"Segmentation output type: {type(seg_outputs)}", logger="current")
                        
                        if seg_outputs is not None and len(seg_outputs) > 0:
                            print_log(f"seg_outputs length: {len(seg_outputs)}", logger="current")
                            seg_data = seg_outputs[0]
                            print_log(f"seg_data type: {type(seg_data)}", logger="current")
                            print_log(f"seg_data keys: {list(seg_data.keys()) if isinstance(seg_data, dict) else 'N/A'}", logger="current")
                            
                            if isinstance(seg_data, dict) and "segmentation" in seg_data:
                                segmentation = seg_data["segmentation"]
                                segments_info = seg_data.get("segments_info", {})
                                
                                print_log(f"segmentation type: {type(segmentation)}", logger="current")
                                print_log(f"segmentation shape: {segmentation.shape if hasattr(segmentation, 'shape') else 'N/A'}", logger="current")
                                
                                if isinstance(segmentation, torch.Tensor):
                                    # Convert tensor to numpy array
                                    mask_np = segmentation.cpu().numpy()
                                    print_log(f"Converted mask shape: {mask_np.shape}, dtype: {mask_np.dtype}", logger="current")
                                    print_log(f"Mask unique values: {np.unique(mask_np)}", logger="current")
                                    
                                    # According to refer_seg_postprocess_fn logic, 1 means object, 255 means ignore
                                    # We create a binary mask: 1 for object, 0 for background
                                    binary_mask = (mask_np == 1).astype(np.uint8) * 255
                                    
                                    print_log(f"Binary mask shape: {binary_mask.shape}, unique values: {np.unique(binary_mask)}", logger="current")
                                    print_log(f"Non-zero pixel count: {np.sum(binary_mask > 0)}", logger="current")
                                    
                                    if binary_mask.sum() > 0:
                                        bbox = _get_bounding_box_from_mask(binary_mask)
                                        mask_b64 = _b64_png_from_array(binary_mask)
                                        
                                        detection = {
                                            "index": i,
                                            "basename": f"{single_prompt.replace(' ', '_')}_{i}",
                                            "class_name": single_prompt,
                                            "bounding_box_xyxy": bbox,
                                            "mask_png_base64": mask_b64
                                        }
                                        all_detections.append(detection)
                                        print_log(f"Successfully processed object {i + 1}: {single_prompt}", logger="current")
                                    else:
                                        print_log(f"Mask for object {i + 1} ({single_prompt}) is empty", logger="current")
                                else:
                                    print_log(f"Segmentation for object {i + 1} ({single_prompt}) is not a tensor: {type(segmentation)}", logger="current")
                            else:
                                print_log(f"seg_data format for object {i + 1} ({single_prompt}) is incorrect", logger="current")
                        else:
                            print_log(f"seg_outputs for object {i + 1} ({single_prompt}) is empty", logger="current")
                            
                    except Exception as e:
                        print_log(f"Error in tensor mode processing: {e}", logger="current")
                        import traceback
                        print_log(traceback.format_exc(), logger="current")
                    finally:
                        # Restore original settings
                        _demo.output_ids_with_output = original_output_ids
                        
                except Exception as e:
                    import traceback
                    print_log(f"Error processing object {i + 1}: {e}", logger="current")
                    print_log(traceback.format_exc(), logger="current")
                    continue

            refseg_result = {
                "image_width": pil.width,
                "image_height": pil.height,
                "detections": all_detections
            }

            return JSONResponse(content={
                "status": "ok",
                "task_name": task_name,
                "llm_input": f"Processed {len(parsed_prompt)} objects",
                "llm_output": f"Successfully segmented {len(all_detections)} objects",
                "refseg_result": refseg_result
            })

        # Non-refseg
        try:
            llm_input, llm_output, seg_output = _demo.run_on_image(
                pil, parsed_prompt, task_name, vprompt_masks=None, threshold=score_thr
            )
        except Exception as e:
            if isinstance(parsed_prompt, list):
                print_log(f"List prompt failed, retrying with concatenation: {e}", logger="current")
                joined_prompt = ", ".join(map(str, parsed_prompt))
                llm_input, llm_output, seg_output = _demo.run_on_image(
                    pil, joined_prompt, task_name, vprompt_masks=None, threshold=score_thr
                )
            else:
                raise

        seg_b64 = None
        if seg_output is not None:
            seg_b64 = _b64_png_from_array(seg_output)

        # Save to local
        output_dir = Path("output_images")
        output_dir.mkdir(exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"segmentation_{timestamp}.png"

        img_data = base64.b64decode(seg_b64)
        with open(output_path, "wb") as f:
            f.write(img_data)
        print_log(f"Segmentation result saved to: {output_path}", logger="current")

        return JSONResponse(content={
            "status": "ok",
            "task_name": task_name,
            "llm_input": llm_input or "",
            "llm_output": llm_output or "",
            "seg_output_png_base64": seg_b64,
        })

    except HTTPException:
        raise
    except Exception as e:
        print_log(f"Error processing request: {str(e)}", logger="current")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------
# Main function
# ---------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8042)
    parser.add_argument("--config", type=str, default=CONFIG_PATH)
    parser.add_argument("--workdir", type=str, default=WORKDIR)
    parser.add_argument("--ckpt", type=str, default=CKPT)
    args = parser.parse_args()

    import sys
    sys.argv = [
        sys.argv[0],
        f"--config={args.config}",
        f"--workdir={args.workdir}",
        f"--ckpt={args.ckpt}"
    ]

    uvicorn.run("xsam_api_server:app", host=args.host, port=args.port, reload=False)
