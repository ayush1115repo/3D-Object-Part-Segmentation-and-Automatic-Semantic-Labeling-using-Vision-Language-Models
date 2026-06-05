"""
PartField Segmentation Studio — Modal Deployment
python -m modal run    modal_app.py::download_weights   # run once
python -m modal deploy modal_app.py                     # go live
python -m modal serve  modal_app.py                     # dev mode
"""

import modal
from pathlib import Path

app    = modal.App("partfield-studio")
volume = modal.Volume.from_name("partfield-data", create_if_missing=True)
VOL    = Path("/data")

# ── Image: download weights ───────────────────────────────────────────────────
dl_image = modal.Image.debian_slim().pip_install("huggingface_hub")

# ── Image: GPU / PartField — exact deps from official README ─────────────────
gpu_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04", add_python="3.10"
    )
    .apt_install(
        "git", "libx11-6", "libgl1", "libxrender1",
        "libglib2.0-0", "libsm6", "libxext6",
        "build-essential", "cmake", "ninja-build",
        "gcc", "g++", "clang",
    )
    # Step 1: PyTorch first (required before torch-scatter + mesh2sdf)
    .pip_install(
        "psutil",
        "torch==2.4.0", "torchvision==0.19.0", "torchaudio==2.4.0",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    # Step 2: All PartField deps in exact order from README
    .pip_install(
        "lightning==2.2", "h5py", "yacs", "trimesh[all]",
        "scikit-image", "loguru", "boto3",
    )
    .pip_install(
        "mesh2sdf", "tetgen", "pymeshlab", "plyfile", "einops",
        "libigl", "polyscope", "potpourri3d", "simple_parsing", "arrgh", "open3d",
    )
    # Step 3: torch-scatter needs torch already installed
    .run_commands(
        "pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.0+cu124.html",
    )
    # Step 4: vtk separately (large package)
    .pip_install("vtk")
    # Step 5: sklearn + matplotlib for our pipeline
    .pip_install(
        "scikit-learn", "scipy", "networkx", "matplotlib", "numpy>=1.24",
    )
    # Step 6: Clone PartField
    .run_commands(
        "git clone https://github.com/nv-tlabs/PartField.git /partfield",
        "cd /partfield && pip install -e . || true",
    )
    # Step 7: Web + Gemini
    .pip_install("flask>=3.0", "flask-cors", "google-generativeai>=0.8", "werkzeug")
)

# ── Image: web server + CPU tasks ─────────────────────────────────────────────
web_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "flask>=3.0", "flask-cors", "requests",
        "werkzeug", "numpy", "trimesh[all]", "scikit-learn",
        "matplotlib", "scipy",
        # v3
    )
    .add_local_dir("./templates", remote_path="/app/templates")
    .add_local_dir("./static",    remote_path="/app/static")
)

# ── Image: Gemma 3 4B vision — local inference on Modal GPU ──────────────────
gemma_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04", add_python="3.11"
    )
    .pip_install(
        "torch==2.4.0", "torchvision==0.19.0",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "transformers==4.51.3",   # force rebuild — has Gemma3ForConditionalGeneration
        "accelerate>=0.35.0",
        "sentencepiece",
        "Pillow>=10.0",
        "numpy",
        "huggingface_hub",
    )
)


# ── Download weights (run once) ───────────────────────────────────────────────
@app.function(image=dl_image, volumes={VOL: volume}, timeout=600)
def download_weights():
    from huggingface_hub import hf_hub_download
    import shutil

    # Reload volume so we see current disk state (important on new accounts)
    volume.reload()

    ckpt_dir  = VOL / "model"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "model_objaverse.ckpt"

    if ckpt_path.exists() and ckpt_path.stat().st_size > 100_000_000:
        sz = ckpt_path.stat().st_size // 1_000_000
        print(f"Checkpoint already exists: {ckpt_path} ({sz} MB)")
        return str(ckpt_path)

    print("Downloading model_objaverse.ckpt (~500 MB)...")
    local = hf_hub_download(
        repo_id="mikaelaangel/partfield-ckpt",
        filename="model_objaverse.ckpt",
        local_dir="/tmp/hf_cache",
    )
    shutil.copy(local, ckpt_path)
    volume.commit()
    sz = ckpt_path.stat().st_size // 1_000_000
    print(f"Saved to {ckpt_path} ({sz} MB)")
    return str(ckpt_path)


# ── Mesh decimation helper ────────────────────────────────────────────────────
# A10G has 22 GB VRAM but PartField loads a full batch of 16 meshes simultaneously.
# Each mesh's sample_points allocates ~28 bytes/face * n_points_per_face.
# Safe per-mesh limit: ~150k faces. Above that we decimate before inference,
# then map cluster labels back to the original mesh for accurate colouring.
MAX_FACES = 150_000

def _maybe_decimate(model_path: Path, job_id: str) -> tuple:
    """
    Returns (path_to_use, original_mesh_path, was_decimated).
    If the mesh is under MAX_FACES, returns the original path unchanged.
    If over, writes a decimated copy to /tmp and returns that path.
    The original_mesh_path is always the original full-res file.
    """
    import trimesh, numpy as np

    try:
        mesh = trimesh.load(str(model_path), force="mesh")
        n_faces = len(mesh.faces)
        print(f"Mesh has {n_faces:,} faces, {len(mesh.vertices):,} vertices")
    except Exception as e:
        print(f"Could not load mesh for face count: {e}")
        return model_path, model_path, False

    if n_faces <= MAX_FACES:
        print(f"Mesh within limit ({n_faces:,} <= {MAX_FACES:,}), no decimation needed")
        return model_path, model_path, False

    target = MAX_FACES
    ratio  = target / n_faces
    print(f"Decimating mesh: {n_faces:,} → ~{target:,} faces (ratio {ratio:.3f})...")

    try:
        # Use open3d for better quality decimation
        import open3d as o3d
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices  = o3d.utility.Vector3dVector(np.array(mesh.vertices))
        o3d_mesh.triangles = o3d.utility.Vector3iVector(np.array(mesh.faces))
        o3d_mesh.compute_vertex_normals()
        decimated = o3d_mesh.simplify_quadric_decimation(target_number_of_triangles=target)
        dec_verts = np.asarray(decimated.vertices)
        dec_faces = np.asarray(decimated.triangles)
        dec_mesh  = trimesh.Trimesh(vertices=dec_verts, faces=dec_faces, process=False)
        print(f"Decimated to {len(dec_mesh.faces):,} faces via open3d")
    except Exception as e:
        print(f"open3d decimation failed ({e}), trying trimesh simplify...")
        try:
            # trimesh simplify_quadric_decimation
            dec_mesh = mesh.simplify_quadric_decimation(target)
            print(f"Decimated to {len(dec_mesh.faces):,} faces via trimesh")
        except Exception as e2:
            print(f"trimesh decimation also failed ({e2}), using vertex subsample...")
            # Last resort: random face subsample (loses quality but never OOMs)
            keep = np.random.choice(len(mesh.faces), target, replace=False)
            keep_faces = mesh.faces[keep]
            used_verts = np.unique(keep_faces)
            v_remap    = {old: new for new, old in enumerate(used_verts)}
            new_faces  = np.vectorize(v_remap.get)(keep_faces)
            dec_mesh   = trimesh.Trimesh(
                vertices=mesh.vertices[used_verts],
                faces=new_faces, process=False,
            )
            print(f"Subsampled to {len(dec_mesh.faces):,} faces")

    # Write decimated mesh as OBJ (PartField handles OBJ well)
    dec_path = Path(f"/tmp/{job_id}_dec.obj")
    dec_mesh.export(str(dec_path))
    print(f"Decimated mesh written to {dec_path}")
    return dec_path, model_path, True


# ── Download Gemma 3 4B weights (run once) ───────────────────────────────────
@app.function(image=gemma_image, volumes={VOL: volume}, timeout=1800)
def download_gemma():
    """Download Gemma 3 4B instruction-tuned weights to Modal volume."""
    from huggingface_hub import snapshot_download
    import os

    volume.reload()
    gemma_dir = VOL / "gemma3-4b"

    # Check if already downloaded
    if (gemma_dir / "config.json").exists():
        print(f"Gemma weights already at {gemma_dir}")
        return str(gemma_dir)

    print("Downloading google/gemma-3-4b-it (~8 GB)...")
    gemma_dir.mkdir(parents=True, exist_ok=True)

    HF_TOKEN = "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"
    snapshot_download(
        repo_id="google/gemma-3-4b-it",
        local_dir=str(gemma_dir),
        token=HF_TOKEN,
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*", "gguf*"],
    )
    volume.commit()
    print(f"Gemma downloaded to {gemma_dir}")
    return str(gemma_dir)


# ── GPU: Gemma 3 4B local vision labelling ────────────────────────────────────
@app.function(
    image=gemma_image,
    gpu="A100",   # 40GB VRAM — handles Gemma 3 4B + 12 images comfortably
    timeout=300,
    volumes={VOL: volume},
)
def run_gemma_labelling(
    clusters: list,
    images_b64: list,
    category: str,
) -> dict:
    """Run Gemma 3 4B locally on Modal GPU to label segmented 3D parts."""
    import torch, json, re, base64, io, traceback
    from pathlib import Path
    from PIL import Image
    from transformers import AutoProcessor

    import os
    volume.reload()
    gemma_dir = str(VOL / "gemma3-4b")
    print(f"Gemma dir: {gemma_dir}, exists: {os.path.exists(gemma_dir)}")
    config_path = os.path.join(gemma_dir, "config.json")

    # Auto-download if weights missing (new account or volume reset)
    if not os.path.exists(config_path):
        print("Gemma weights not found — downloading automatically...")
        from huggingface_hub import snapshot_download
        import shutil as _shu
        Path(gemma_dir).mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id="google/gemma-3-4b-it",
            local_dir=gemma_dir,
            token="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",
            ignore_patterns=["*.msgpack","*.h5","flax_model*","tf_model*","gguf*"],
        )
        volume.commit()
        print("Auto-download complete!")
    else:
        print(f"Gemma weights found: {config_path}")

    PALETTE = [
        "#E63946","#2A9D8F","#E9C46A","#457B9D","#8338EC",
        "#06D6A0","#FF6B6B","#118AB2","#F72585","#3A86FF",
        "#FB5607","#8AC926","#FFD166","#4CC9F0","#7B2D8B",
        "#EF476F","#00B4D8","#90123F","#50C878","#FF8C00",
        "#6432C8","#149650","#C85014","#3264C8",
    ]

    EXAMPLES = {
        "chair":      "seat, backrest, left_armrest, right_armrest, front_left_leg, front_right_leg, rear_left_leg, rear_right_leg",
        "car":        "hood, roof, trunk_lid, front_bumper, rear_bumper, left_front_door, right_front_door, windshield, front_left_wheel, rear_right_wheel",
        "man":        "head, hair, torso_shirt, left_upper_arm, left_forearm, left_hand, right_upper_arm, right_forearm, right_hand, pants_pelvis, left_thigh, left_shin, left_shoe, right_thigh, right_shin, right_shoe, glasses",
        "woman":      "head, hair, torso, left_arm, right_arm, left_hand, right_hand, dress_skirt, left_leg, right_leg, left_shoe, right_shoe",
        "character":  "head, hair, torso, left_upper_arm, left_forearm, left_hand, right_upper_arm, right_forearm, right_hand, pelvis, left_thigh, left_shin, left_foot, right_thigh, right_shin, right_foot",
        "robot":      "head, visor, torso, left_shoulder, left_upper_arm, left_forearm, left_hand, right_shoulder, right_upper_arm, right_forearm, right_hand, pelvis, left_thigh, left_shin, left_foot, right_thigh, right_shin, right_foot, antenna",
        "wagon":      "water_tank, front_left_wheel, front_right_wheel, rear_left_wheel, rear_right_wheel, front_axle, rear_axle, tongue_pole, frame_rail, tank_band, wheel_hub",
        "water wagon":"water_tank, front_left_wheel, front_right_wheel, rear_left_wheel, rear_right_wheel, front_axle, rear_axle, tongue_pole, frame_rail, tank_band",
        "scissors":   "left_blade, right_blade, left_finger_ring, right_thumb_ring, pivot_screw",
        "watch":      "watch_case, dial_face, crown_button, strap_top, strap_bottom, buckle_clasp, bezel_ring",
        "bicycle":    "frame, front_wheel, rear_wheel, handlebar, seat, front_fork, chain_guard, pedal_left, pedal_right",
        "airplane":   "fuselage, left_wing, right_wing, vertical_tail, left_elevator, right_elevator, nose_cone, left_engine, right_engine",
        "lamp":       "base, pole, shade, bulb_socket, switch",
        "table":      "tabletop, front_left_leg, front_right_leg, rear_left_leg, rear_right_leg, apron",
        "sword":      "blade, crossguard, grip_handle, pommel, basket_guard, ricasso, fuller",
        "knife":      "blade, edge, spine, crossguard, grip_handle, pommel, bolster",
        "gun":        "barrel, slide, frame, grip, trigger, trigger_guard, magazine, sight",
        "pistol":     "barrel, slide, frame, grip, trigger, trigger_guard, magazine, rear_sight",
        "rifle":      "barrel, stock, receiver, trigger_group, magazine, handguard, scope_mount",
        "shield":     "face_plate, rim_border, boss_center, strap_mount, grip_handle",
        "helmet":     "dome, visor, chin_guard, ear_guard, neck_guard, crest, cheek_plate",
        "armor":      "chest_plate, back_plate, shoulder_left, shoulder_right, arm_left, arm_right, leg_left, leg_right",
        "car":        "hood, roof, trunk_lid, front_bumper, rear_bumper, left_front_door, right_front_door, windshield, front_left_wheel, rear_right_wheel",
        "truck":      "cab, cargo_bed, front_bumper, rear_bumper, front_left_wheel, front_right_wheel, rear_left_wheel, rear_right_wheel, exhaust_stack",
        "house":      "roof, front_wall, left_wall, right_wall, rear_wall, door, window_front, chimney, foundation, porch",
        "tree":       "trunk, root_base, branch_main_left, branch_main_right, canopy_top, canopy_mid",
        "fruit":      "body, stem, leaf, skin",
    }
    cat_low = category.lower()
    example_str = next((v for k,v in EXAMPLES.items() if k in cat_low or cat_low in k),
                       "main_body, upper_section, lower_section, left_part, right_part, front_part, rear_part")

    # Build spatial metadata
    lines = []
    for c in clusters:
        locs = [k for k in ("top","bottom","left","right","front","back") if c.get(f"is_{k}")]
        sz   = c["size"]
        lines.append(
            f"  Cluster {c['id']}: pos={'/'.join(locs) or 'center'}, "
            f"centroid=({c['centroid'][0]:.2f},{c['centroid'][1]:.2f},{c['centroid'][2]:.2f}), "
            f"size={sz[0]:.2f}x{sz[1]:.2f}x{sz[2]:.2f}, faces={c['face_count']}"
        )

    cluster_data_str = "\n".join(lines)

    # Map cluster IDs to colors so Gemma can match image colors to clusters
    PALETTE_HEX = [
        "#E63946","#2A9D8F","#E9C46A","#457B9D","#8338EC",
        "#06D6A0","#FF6B6B","#118AB2","#F72585","#3A86FF",
        "#FB5607","#8AC926","#FFD166","#4CC9F0","#7B2D8B",
        "#EF476F","#00B4D8","#90123F","#50C878","#FF8C00",
        "#6432C8","#149650","#C85014","#3264C8",
    ]
    color_map = "\n".join([
        f"  cluster_id {c['id']} = color {PALETTE_HEX[c['id'] % len(PALETTE_HEX)]}"
        for c in clusters
    ])

    # Build exact example with real cluster IDs
    example_json = json.dumps(
        [{"cluster_id": c["id"], "label": f"{category.split()[0]}_part_{c['id']}",
          "description": "describe what this specific part does"}
         for c in clusters],
        separators=(",", ":")
    )

    # Image order: FRONT, BACK, RIGHT, LEFT, TOP, BOTTOM, ISO_FL, ISO_FR
    img_guide = (
        "Image 1=FRONT  Image 2=BACK  Image 3=RIGHT-SIDE  Image 4=LEFT-SIDE  "
        "Image 5=TOP-DOWN  Image 6=BOTTOM-UP  Image 7=ISO-FRONT-LEFT  Image 8=ISO-FRONT-RIGHT"
    )

    prompt_text = (
        f"You are an expert 3D model part labeller. "
        f"Label every color-coded segment of this '{category}'.\n\n"
        f"IMAGES: 8 orthographic views of the COMPLETE model from all angles.\n"
        f"{img_guide}\n"
        f"Each colored region = one segment. Segment IDs are annotated in small text.\n\n"
        f"COLOR → CLUSTER ID MAP:\n{color_map}\n\n"
        f"SPATIAL METADATA (centroid x,y,z; y=up; face_count=size):\n"
        f"{cluster_data_str}\n\n"
        f"KNOWN PARTS OF A {category.upper()}: {example_str}\n\n"
        f"LABELLING RULES:\n"
        f"1. Cross-reference all 8 views to identify each colored region\n"
        f"2. Use the spatial metadata: large face_count = main structural part\n"
        f"   top(y>0.5), bottom(y<-0.5), left(x<-0.3), right(x>0.3)\n"
        f"3. FORBIDDEN WORDS (never use): part, body, section, component, "
        f"structure, piece, element, area, object, thing\n"
        f"   USE INSTEAD: specific mechanical/anatomical names from the known parts list\n"
        f"4. Symmetric pairs MUST have _left/_right or _front/_rear suffix\n"
        f"5. Zero duplicates — every label must be unique\n"
        f"6. Every cluster_id 0–{len(clusters)-1} must appear exactly once\n\n"
        f"OUTPUT: JSON array only. No markdown. No explanation. No preamble.\n"
        f"{example_json}"
    )

    print(f"Loading Gemma 3 4B from {gemma_dir}...")
    from transformers import AutoProcessor as AP3, AutoModelForImageTextToText
    import transformers
    print(f"Transformers version: {transformers.__version__}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}, CUDA available: {torch.cuda.is_available()}")

    processor = AP3.from_pretrained(gemma_dir)

    # Load model using AutoModelForImageTextToText (works for all transformers >=4.45)
    model = AutoModelForImageTextToText.from_pretrained(
        gemma_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",  # avoid SDPA alignment error with multi-image
    )
    print("Loaded via AutoModelForImageTextToText (eager attention)")

    model.eval()
    print("Gemma 3 4B loaded!")

    # Decode images
    pil_images = []
    for b64 in images_b64[:8]:
        try:
            raw = base64.b64decode(b64)
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            img = img.resize((448, 448))   # safe size for SDPA alignment
            pil_images.append(img)
        except Exception as e:
            print(f"Image decode error: {e}")
    print(f"Decoded {len(pil_images)} images")

    # Build messages for Gemma 3 vision
    content_list = []
    for img in pil_images:
        content_list.append({"type": "image", "image": img})
    content_list.append({"type": "text", "text": prompt_text})
    messages = [{"role": "user", "content": content_list}]

    # Apply chat template
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    # Move all tensors to device with correct dtypes
    model_device = next(model.parameters()).device
    processed_inputs = {}
    for k, v in inputs.items():
        if not hasattr(v, "to"):
            processed_inputs[k] = v
            continue
        if k == "pixel_values":
            processed_inputs[k] = v.to(model_device, dtype=torch.bfloat16)
        elif v.dtype in (torch.float32, torch.float16):
            processed_inputs[k] = v.to(model_device, dtype=torch.bfloat16)
        else:
            processed_inputs[k] = v.to(model_device)

    print(f"Input keys: {list(processed_inputs.keys())}")
    print(f"Input ids shape: {processed_inputs['input_ids'].shape}")

    with torch.inference_mode():
        output_ids = model.generate(
            **processed_inputs,
            max_new_tokens=2048,
            do_sample=False,           # greedy = deterministic, no hallucination
            temperature=1.0,           # must be 1.0 when do_sample=False
            top_p=None,                # unset to avoid warning
            top_k=None,                # unset to avoid warning
            repetition_penalty=1.15,   # prevent duplicate labels
            pad_token_id=processor.tokenizer.eos_token_id,
        )

    input_len = processed_inputs["input_ids"].shape[-1]
    text = processor.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()
    print(f"Gemma raw output ({len(text)} chars):")
    print(text[:800])
    print("---END OUTPUT---")

    # ── Robust JSON extraction ────────────────────────────────────────────────
    def extract_json(text):
        """Robust JSON extractor — handles all Gemma output formats."""
        import re as _re

        # 1. Strip markdown fences
        clean = _re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()

        # 2. Try whole text
        for candidate in [clean, text.strip()]:
            try: return json.loads(candidate)
            except: pass

        # 3. Find JSON array — try largest match first
        for pattern in [r'\[\s*\{.*?\}\s*\]', r'\[.*?\]']:
            for m in _re.finditer(pattern, clean, _re.DOTALL):
                try: return json.loads(m.group())
                except: pass

        # 4. Find JSON object with labels key
        for m in _re.finditer(r'\{[^{}]*"labels"[^{}]*\[.*?\][^{}]*\}', clean, _re.DOTALL):
            try:
                d = json.loads(m.group())
                return d.get("labels", d)
            except: pass

        # 5. Extract individual objects
        collected = []
        for m in _re.finditer(r'\{[^{}]+\}', clean):
            try:
                obj = json.loads(m.group())
                if "cluster_id" in obj or "label" in obj:
                    collected.append(obj)
            except: pass
        if collected: return collected

        # 6. Parse numbered/bulleted lists like "0: head" or "cluster 0 → head"
        collected = []
        for line in clean.split("\n"):
            line = line.strip()
            # Match: 0: head, cluster_id: 0 label: head, etc.
            m = _re.search(r'(?:cluster[_\s]?(?:id)?[:\s]+)?(\d+)[:\s\-→]+([a-zA-Z_][a-zA-Z_\s]*)', line)
            if m:
                cid  = int(m.group(1))
                name = m.group(2).strip().lower().replace(" ", "_")
                if 0 <= cid < 50 and len(name) > 1:
                    collected.append({"cluster_id": cid, "label": name, "description": name})
        if len(collected) >= len(clusters) // 2:
            return collected

        raise ValueError(f"Cannot extract JSON. Raw: {text[:300]}")

    try:
        raw_labels = extract_json(text)
        if isinstance(raw_labels, dict):
            raw_labels = raw_labels.get("labels", raw_labels.get("parts", [raw_labels]))
        if not isinstance(raw_labels, list) or len(raw_labels) == 0:
            raise ValueError(f"Empty or invalid labels list")

        labels = []
        seen_names = {}
        raw_labels.sort(key=lambda x: x.get("cluster_id", 0))
        for i, item in enumerate(raw_labels):
            cid  = int(item.get("cluster_id", i))
            name = str(item.get("label", f"part_{cid}")).lower().strip().replace(" ","_")
            # Deduplicate names
            if name in seen_names:
                seen_names[name] += 1
                c = next((x for x in clusters if x["id"] == cid), {})
                locs = [k for k in ("left","right","front","back","top","bottom") if c.get(f"is_{k}")]
                suffix = "_".join(locs) if locs else str(seen_names[name])
                name = f"{name}_{suffix}"
            else:
                seen_names[name] = 1

            labels.append(dict(
                cluster_id  = cid,
                label       = name,
                # MUST match frontend: PALETTE[cluster_id % len]
                color       = PALETTE[cid % len(PALETTE)],
                description = str(item.get("description", "")),
                confidence  = float(item.get("confidence", 0.85)),
            ))

        # Ensure cluster_ids are valid integers
        valid_ids = {c["id"] for c in clusters}
        labels = [l for l in labels if isinstance(l.get("cluster_id"), int)
                  and l["cluster_id"] in valid_ids]

        # If Gemma returned labels by position (0,1,2...) but real IDs differ,
        # remap positional IDs to actual cluster IDs
        if labels and len(labels) == len(clusters):
            returned_ids = {l["cluster_id"] for l in labels}
            actual_ids   = {c["id"] for c in clusters}
            if returned_ids != actual_ids and returned_ids == set(range(len(clusters))):
                sorted_clusters = sorted(clusters, key=lambda c: c["id"])
                for i, lbl in enumerate(sorted(labels, key=lambda l: l["cluster_id"])):
                    lbl["cluster_id"] = sorted_clusters[i]["id"]
                print("Remapped positional IDs to actual cluster IDs")

        # Backfill any missing cluster IDs
        labelled = {l["cluster_id"] for l in labels}
        for c in clusters:
            if c["id"] not in labelled:
                locs = [k for k in ("left","right","top","bottom","front","back")
                        if c.get(f"is_{k}")]
                fallback_name = f"{'_'.join(locs) or 'center'}_part_{c['id']}"
                labels.append(dict(
                    cluster_id  = c["id"],
                    label       = fallback_name,
                    color       = PALETTE[c["id"] % len(PALETTE)],
                    description = f"Part {c['id']} of the {category}.",
                    confidence  = 0.0,
                ))

        # Assign final colors — MUST use cluster_id % len for consistency with PLY
        labels.sort(key=lambda l: l["cluster_id"])
        for lbl in labels:
            lbl["color"] = PALETTE[lbl["cluster_id"] % len(PALETTE)]

        obj_desc = f"A {category} with {len(labels)} segmented parts."
        print(f"Gemma labelled {len(labels)} parts successfully!")
        return dict(labels=labels, object_description=obj_desc)

    except Exception as e:
        print(f"JSON parse failed: {e}\nRaw output:\n{text}\n")
        traceback.print_exc()
        def _geometric_labels_local(clusters, category):
            PALETTE = ["#E63946","#2A9D8F","#E9C46A","#457B9D","#F4A261",
                       "#A8DADC","#8338EC","#06D6A0","#FF6B6B","#FFD166",
                       "#118AB2","#EF476F","#06A77D","#D62246","#4CC9F0",
                       "#7B2D8B","#F72585","#3A86FF","#FB5607","#8AC926"]
            return [dict(cluster_id=c["id"], label=f"part_{c['id']}",
                         color=PALETTE[i%len(PALETTE)],
                         description="Geometric fallback.", confidence=0.0)
                    for i, c in enumerate(sorted(clusters, key=lambda x: x["id"]))]

        return dict(
            labels=_geometric_labels_local(clusters, category),
            object_description=f"A {category} with {len(clusters)} parts (geometric fallback).",
        )


def _parse_from_fl(model_path, FL_raw, n_parts, job_id):
    """Build cluster data from an existing FL (per-face label) array.
    Used when the model already has good submesh structure.
    """
    import trimesh, numpy as np
    from pathlib import Path

    # Load scene to get individual submeshes (same as _extract_submesh_labels)
    scene = trimesh.load(str(model_path), process=False)
    if hasattr(scene, "geometry") and scene.geometry:
        geoms = [g for g in scene.geometry.values()
                 if hasattr(g, "faces") and len(g.faces) > 0]
        mesh = trimesh.util.concatenate(geoms)
    else:
        mesh = scene

    V = np.array(mesh.vertices, dtype=np.float32)
    F = np.array(mesh.faces,    dtype=np.int32)
    print(f"_parse_from_fl: V={len(V)}, F={len(F)}, FL={len(FL_raw)}")

    # Normalise
    vmin, vmax = V.min(0), V.max(0)
    V_norm = (V - vmin) / (vmax - vmin + 1e-8) * 2 - 1

    FL = np.array(FL_raw, dtype=int)

    # Ensure FL length matches F
    if len(FL) != len(F):
        from sklearn.neighbors import KDTree
        fc = (V_norm[F[:,0]] + V_norm[F[:,1]] + V_norm[F[:,2]]) / 3.0
        src_pts = np.array([(V_norm[f[0]] + V_norm[f[1]] + V_norm[f[2]]) / 3.0
                            for f in F[:len(FL)]], dtype=np.float32) if len(FL) <= len(F) else V_norm[:len(FL)]
        _, idx = KDTree(src_pts).query(fc, k=1)
        FL = FL[idx.flatten() % len(FL)]

    # Remap to 0..N-1
    unique_ids = np.unique(FL)
    remap = {int(old): new for new, old in enumerate(unique_ids)}
    FL    = np.array([remap[v] for v in FL], dtype=int)

    PALETTE = [
        [230, 57, 70],  [42, 157, 143],  [233, 196, 106], [69, 123, 157],
        [131, 56, 236], [6, 214, 160],   [255, 107, 107], [17, 138, 178],
        [247, 37, 133], [58, 134, 255],  [251, 86, 7],    [138, 201, 38],
        [255, 209, 102],[76, 201, 240],  [123, 45, 139],  [239, 71, 111],
        [0, 180, 216],  [144, 12, 63],   [80, 200, 120],  [255, 140, 0],
        [100, 50, 200], [20, 150, 80],   [200, 80, 20],   [50, 100, 200],
    ]

    clusters = []
    for cid in sorted(np.unique(FL).tolist()):
        mask = FL == cid
        fi   = np.where(mask)[0]
        fv   = V_norm[F[fi]].reshape(-1, 3)
        cen  = fv.mean(0)
        bmin = fv.min(0); bmax = fv.max(0); sz = bmax - bmin
        clusters.append(dict(
            id=int(cid), face_count=int(mask.sum()),
            vertex_count=int(np.unique(F[fi]).shape[0]),
            centroid=[round(float(x),4) for x in cen],
            bbox_min=[round(float(x),4) for x in bmin],
            bbox_max=[round(float(x),4) for x in bmax],
            size=[round(float(x),4) for x in sz],
            volume=round(float(sz[0]*sz[1]*sz[2]),6),
            is_top=float(cen[1])>0.5, is_bottom=float(cen[1])<-0.5,
            is_left=float(cen[0])<-0.3, is_right=float(cen[0])>0.3,
            is_front=float(cen[2])>0.3, is_back=float(cen[2])<-0.3,
        ))

    n_verts = len(V_norm)
    vc_votes = {}
    for fi2, face in enumerate(F):
        cid = int(FL[fi2])
        for vi in face:
            vi = int(vi)
            if vi not in vc_votes: vc_votes[vi] = {}
            vc_votes[vi][cid] = vc_votes[vi].get(cid,0)+1

    vert_colors = np.zeros((n_verts,3), dtype=np.uint8)
    for vi in range(n_verts):
        best_cid = max(vc_votes[vi], key=vc_votes[vi].get) if vi in vc_votes else 0
        vert_colors[vi] = PALETTE[best_cid % len(PALETTE)]

    colored_out = VOL / "uploads" / f"{job_id}_colored.ply"
    _write_ply_binary(V_norm, F, vert_colors, colored_out)

    mesh_cache = VOL / "uploads" / f"{job_id}_mesh.npz"
    np.savez_compressed(str(mesh_cache),
                        V=V_norm.astype(np.float32),
                        F=F.astype(np.int32),
                        FL=FL.astype(np.int32))
    print(f"Submesh parse done: {len(clusters)} clusters, PLY+npz saved")

    return dict(
        clusters=clusters, FL=FL.tolist(),
        colored_model_path=str(colored_out), colored_model_ext="ply",
    )


def _extract_submesh_labels(model_path, min_parts=2):
    """Extract per-face cluster labels from GLB/OBJ submesh/material structure.
    Returns (FL, n_parts) where FL[i] = submesh index for face i.
    FL length matches the concatenated mesh face count exactly.
    """
    import trimesh, numpy as np
    try:
        scene = trimesh.load(str(model_path), process=False)
        if not hasattr(scene, "geometry") or not scene.geometry:
            print("No scene geometry — model is single mesh")
            return None, 0

        # Filter valid geometries with actual faces
        geoms = [(name, g) for name, g in scene.geometry.items()
                 if hasattr(g, "faces") and len(g.faces) > 0
                 and hasattr(g, "vertices") and len(g.vertices) > 0]

        print(f"Submesh extraction: found {len(geoms)} valid geometry groups")
        if len(geoms) < min_parts:
            print(f"Only {len(geoms)} geoms < min_parts={min_parts}")
            return None, 0

        # Build per-face labels — gid = submesh index
        all_FL = []
        for gid, (name, geom) in enumerate(geoms):
            nF = len(geom.faces)
            all_FL.append(np.full(nF, gid, dtype=np.int32))
            print(f"  Geom {gid} '{name}': {len(geom.vertices)} verts, {nF} faces")

        FL_all   = np.concatenate(all_FL)
        n_raw    = len(np.unique(FL_all))
        print(f"Raw submeshes: {n_raw} unique parts")

        # ── Merge tiny submeshes into nearest large neighbor ──────────────────
        # Count faces per submesh
        face_counts = {}
        for gid in range(len(geoms)):
            face_counts[gid] = int((FL_all == gid).sum())

        total_faces  = len(FL_all)
        # Only merge truly micro-geometry (< 0.1% of total faces)
        # 0.3% was too aggressive and merged real parts like basket/crossguard
        merge_thresh = max(5, total_faces * 0.001)
        print(f"Merge threshold: {merge_thresh:.0f} faces ({merge_thresh/total_faces*100:.2f}%)")

        # Build centroids per submesh
        all_V_stacked = []
        all_F_stacked = []
        v_off = 0
        for gid, (name, geom) in enumerate(geoms):
            V = np.array(geom.vertices, dtype=np.float32)
            F = np.array(geom.faces, dtype=np.int32)
            all_V_stacked.append(V)
            all_F_stacked.append(F + v_off)
            v_off += len(V)

        V_all2 = np.vstack(all_V_stacked) if all_V_stacked else np.zeros((1,3))
        F_all2 = np.vstack(all_F_stacked) if all_F_stacked else np.zeros((1,3),dtype=int)

        # Centroid per submesh
        centroids = {}
        for gid in range(len(geoms)):
            fi = np.where(FL_all == gid)[0]
            if len(fi) == 0:
                centroids[gid] = np.zeros(3)
                continue
            verts = V_all2[F_all2[fi]].reshape(-1, 3)
            centroids[gid] = verts.mean(0)

        # Build remap: tiny submesh → nearest large submesh
        large_ids = [g for g, c in face_counts.items() if c >= merge_thresh]
        remap = {g: g for g in range(len(geoms))}

        if large_ids:
            large_centroids = np.array([centroids[g] for g in large_ids])
            for gid, count in face_counts.items():
                if count < merge_thresh and gid not in large_ids:
                    # Find nearest large submesh centroid
                    dists = np.linalg.norm(large_centroids - centroids[gid], axis=1)
                    nearest = large_ids[np.argmin(dists)]
                    remap[gid] = nearest
                    print(f"  Merge submesh {gid} ({count} faces) → {nearest}")

        # Apply remap
        FL_merged = np.array([remap[x] for x in FL_all], dtype=np.int32)

        # Re-index to 0..N-1
        unique_merged = sorted(set(FL_merged.tolist()))
        reindex = {old: new for new, old in enumerate(unique_merged)}
        FL_final = np.array([reindex[x] for x in FL_merged], dtype=np.int32)
        n_final  = len(unique_merged)
        print(f"After merge: {n_final} parts (from {n_raw})")
        return FL_final, n_final

    except Exception as e:
        import traceback
        print(f"Submesh extraction failed: {e}")
        traceback.print_exc()
        return None, 0


# ── GPU: PartField inference + clustering ─────────────────────────────────────
@app.function(image=gpu_image, gpu="A10G", timeout=900, volumes={VOL: volume})
def run_partfield_gpu(job_id: str, model_path_str: str, n_parts: int):
    import sys, subprocess, numpy as np, os
    sys.path.insert(0, "/partfield")

    # Allow CUDA memory fragmentation fix
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    model_path = Path(model_path_str)
    ckpt_path  = VOL / "model" / "model_objaverse.ckpt"
    cfg_path   = "/partfield/configs/final/demo.yaml"
    feat_name  = f"partfield_features/{job_id}"
    dump_dir   = VOL / "clustering" / job_id
    dump_dir.mkdir(parents=True, exist_ok=True)
    (dump_dir / "cluster_out").mkdir(exist_ok=True)

    # ── CRITICAL: put ONLY this job's file in an isolated directory ──────────
    # PartField scans the entire data_dir — if we point it at VOL/uploads/
    # it processes every previous upload too, causing OOM and wrong results.
    import shutil as _shutil
    job_input_dir = Path(f"/tmp/partfield_input_{job_id}")
    job_input_dir.mkdir(parents=True, exist_ok=True)
    # Remove any stale files from a previous run with same job_id
    for f in job_input_dir.iterdir():
        f.unlink()

    # ── Auto-decimate if mesh is too large for A10G VRAM ─────────────────────
    # Decimate into the isolated dir so PartField only sees ONE file
    infer_path, orig_path, was_decimated = _maybe_decimate(model_path, job_id)

    if was_decimated:
        # Already written to /tmp by _maybe_decimate, move into job_input_dir
        isolated = job_input_dir / infer_path.name
        _shutil.copy2(str(infer_path), str(isolated))
        infer_path = isolated
    else:
        # Copy original into isolated dir — one file only
        isolated = job_input_dir / model_path.name
        _shutil.copy2(str(model_path), str(isolated))
        infer_path = isolated

    data_dir = job_input_dir   # PartField sees exactly ONE file

    file_ext = infer_path.suffix.lower().lstrip(".")
    is_pc    = file_ext == "ply"

    # ── Step 0: Try submesh-based segmentation first ─────────────────────────
    # If the GLB already has distinct material groups (like this robot),
    # use those directly — they're more accurate than PartField clustering
    print(f"[{job_id}] Checking for existing submesh structure...")
    FL_submesh, n_submesh = _extract_submesh_labels(model_path, min_parts=2)

    if FL_submesh is not None and n_submesh >= 2:
        # Only use submesh if count is reasonable (not 43 bolts/screws for a bicycle)
        # Always use submesh segmentation — merging already reduced to good count
        actual_parts = n_submesh
        print(f"[{job_id}] Using submesh segmentation: {actual_parts} parts (skipping PartField)")
        data = _parse_from_fl(model_path, FL_submesh, actual_parts, job_id)
        try:
            import shutil as _sh
            _sh.rmtree(str(job_input_dir), ignore_errors=True)
        except Exception:
            pass
        volume.commit()
        return data

    # ── PartField inference ──────────────────────────────────────────────────
    print(f"[{job_id}] No submesh structure — running PartField inference on A10G GPU...")
    env = {**__import__("os").environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    r1 = subprocess.run(
        [
            sys.executable, "/partfield/partfield_inference.py",
            "-c", cfg_path,
            "--opts",
            "continue_ckpt", str(ckpt_path),
            "result_name",   feat_name,
            "dataset.data_path", str(data_dir),
            *( ["is_pc", "True"] if is_pc else [] ),
        ],
        cwd="/partfield", capture_output=True, text=True, env=env,
    )
    print("INFERENCE STDOUT:", r1.stdout[-3000:])
    if r1.returncode != 0:
        err = r1.stderr[-2000:] + "\n" + r1.stdout[-2000:]
        raise RuntimeError(f"Inference failed: {err}")

    # ── Clustering ───────────────────────────────────────────────────────────
    print(f"[{job_id}] Running agglomerative clustering (option 1 + KNN)...")

    base_args = [
        sys.executable, "/partfield/run_part_clustering.py",
        "--root",             f"/partfield/exp_results/{feat_name}",
        "--dump_dir",         str(dump_dir),
        "--source_dir",       str(data_dir),
        "--max_num_clusters", str(n_parts),
        "--export_mesh",      "True",
    ]

    if is_pc:
        r2 = subprocess.run(base_args + ["--is_pc", "True"],
                            cwd="/partfield", capture_output=True, text=True)
    else:
        # option=1 + with_knn: MST-based, respects mesh topology (best for robots/chars)
        r2 = subprocess.run(
            base_args + ["--use_agglo", "True", "--option", "1", "--with_knn", "True"],
            cwd="/partfield", capture_output=True, text=True)
        print("CLUSTERING option1+knn:", r2.stdout[-1500:])

        if r2.returncode != 0:
            print("option 1+knn failed, trying option 0 (basic hierarchical)...")
            r2 = subprocess.run(
                base_args + ["--use_agglo", "True", "--option", "0"],
                cwd="/partfield", capture_output=True, text=True)
            print("CLUSTERING option0:", r2.stdout[-1500:])

        if r2.returncode != 0:
            err = r2.stderr[-2000:] + "\n" + r2.stdout[-2000:]
            raise RuntimeError(f"Clustering failed: {err}")

    # ── Parse + build colored PLY ──────────────────────────────────────────
    # infer_path = decimated/isolated mesh (what PartField ran on)
    # orig_path  = original full-res mesh (what we colour and serve)
    # If decimated, we map labels from decimated faces → original faces via KDTree
    data = _parse(infer_path, orig_path, dump_dir, n_parts, job_id)

    # Clean up isolated input dir to free /tmp space
    try:
        import shutil as _sh
        _sh.rmtree(str(job_input_dir), ignore_errors=True)
    except Exception:
        pass

    volume.commit()
    return data


def _parse(infer_path, orig_path, dump_dir, n_parts, job_id):
    """
    infer_path: path PartField actually ran on (may be decimated)
    orig_path:  original full-res mesh path (used for colouring output)
    """
    import trimesh, numpy as np

    # Load the inference mesh (used to read .npy labels)
    infer_mesh = trimesh.load(str(infer_path), force="mesh")
    V_infer    = np.array(infer_mesh.vertices, dtype=np.float32)
    F_infer    = np.array(infer_mesh.faces,    dtype=np.int32)

    # Load original full-res mesh (used for output / colouring)
    was_decimated = str(infer_path) != str(orig_path)
    if was_decimated:
        print(f"Remapping labels from decimated ({len(F_infer):,} faces) "
              f"to original mesh...")
        orig_mesh = trimesh.load(str(orig_path), force="mesh")
    else:
        orig_mesh = infer_mesh

    V    = np.array(orig_mesh.vertices, dtype=np.float32)
    F    = np.array(orig_mesh.faces,    dtype=np.int32)

    # Use inference mesh for normalisation scale (same as PartField saw)
    vmin, vmax = V_infer.min(0), V_infer.max(0)
    V_norm = (V - vmin) / (vmax - vmin + 1e-8) * 2 - 1

    # ── Find the .npy cluster label file ──────────────────────────────────
    uid      = infer_path.stem      # use infer_path stem (may be decimated)
    # Search cluster_out dir and also dump_dir root for .npy files
    npy_glob = (sorted((dump_dir / "cluster_out").glob("*.npy"))
                + sorted(dump_dir.glob("*.npy")))
    npy_path = None
    for f in npy_glob:
        if uid in f.name:
            npy_path = f; break
    if npy_path is None and npy_glob:
        npy_path = npy_glob[-1]

    # Also check for .txt label files (some PartField versions output these)
    if npy_path is None or not npy_path.exists():
        txt_glob = (sorted((dump_dir / "cluster_out").glob("*.txt"))
                    + sorted(dump_dir.glob("*.txt")))
        if txt_glob:
            # txt file: one int per line = per-face label
            txt_path = txt_glob[-1]
            print(f"Using txt label file: {txt_path}")
            FL_raw_txt = np.array([int(l.strip()) for l in
                                   open(txt_path).readlines() if l.strip()], dtype=int)
            npy_path = dump_dir / "cluster_out" / "_converted.npy"
            npy_path.parent.mkdir(exist_ok=True)
            np.save(str(npy_path), FL_raw_txt)

    if npy_path is None or not npy_path.exists():
        # Last resort: try to parse the colored mesh exported by --export_mesh True
        ply_glob = (sorted((dump_dir / "cluster_out").glob("*.ply"))
                    + sorted(dump_dir.glob("*.ply")))
        if ply_glob:
            import trimesh as _tr
            _pm = _tr.load(str(ply_glob[-1]))
            if hasattr(_pm, "visual") and hasattr(_pm.visual, "face_colors"):
                fc_colors = np.array(_pm.visual.face_colors)[:, :3]
                unique_colors = np.unique(fc_colors, axis=0)
                color_to_id   = {tuple(c): i for i, c in enumerate(unique_colors)}
                FL_raw_ply    = np.array([color_to_id[tuple(c)] for c in fc_colors], dtype=int)
                npy_path = dump_dir / "cluster_out" / "_from_ply.npy"
                npy_path.parent.mkdir(exist_ok=True)
                np.save(str(npy_path), FL_raw_ply)
                print(f"Extracted {len(unique_colors)} clusters from colored PLY mesh")

    if npy_path is None or not npy_path.exists():
        raise RuntimeError(
            "PartField produced no cluster label output (.npy/.txt/.ply). "
            "Check the clustering logs above."
        )

    print(f"Loading cluster labels from: {npy_path}")
    FL_raw = np.load(str(npy_path)).flatten().astype(int)
    n_unique = len(np.unique(FL_raw))
    print(f"Raw labels shape: {FL_raw.shape}, unique clusters: {n_unique}, "
          f"infer faces: {len(F_infer):,}, orig faces: {len(F):,}")

    # If only 1 cluster came out, use improved spectral KMeans fallback
    if n_unique < 2:
        print(f"WARNING: Only {n_unique} cluster(s). Using spectral KMeans fallback.")
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        import trimesh as _tr

        _tm = _tr.Trimesh(vertices=V_infer, faces=F_infer, process=False)
        V_n = (V_infer - V_infer.min(0)) / (V_infer.max(0) - V_infer.min(0) + 1e-8)

        # Face centroids (normalised)
        fc = (V_n[F_infer[:,0]] + V_n[F_infer[:,1]] + V_n[F_infer[:,2]]) / 3.0

        # Face normals — critical for thin parts like blades (they face different directions)
        fn = np.array(_tm.face_normals, dtype=np.float32)

        # Face area — helps distinguish large structural vs small detail parts
        fa = np.array(_tm.area_faces, dtype=np.float32)
        fa = (fa / (fa.max() + 1e-8)).reshape(-1, 1)

        # Combined feature: position(3) + normals(3) weighted higher + area(1)
        feat = np.hstack([fc * 1.0, fn * 0.8, fa * 0.3])
        feat = StandardScaler().fit_transform(feat)

        km = KMeans(
            n_clusters=min(n_parts, len(fc)),
            random_state=42, n_init=15,
            max_iter=500,
        )
        FL_raw = km.fit_predict(feat).astype(int)
        print(f"Spectral KMeans fallback: {len(np.unique(FL_raw))} clusters")

    # ── Step 1: get per-face labels on the INFERENCE mesh ─────────────────
    nF_infer = len(F_infer)
    nV_infer = len(V_infer)

    if len(FL_raw) == nF_infer:
        FL_infer = FL_raw
    elif len(FL_raw) == nV_infer:
        # Per-vertex → per-face majority vote on inference mesh
        FL_infer = np.zeros(nF_infer, dtype=int)
        for fi, face in enumerate(F_infer):
            vals = FL_raw[face]
            FL_infer[fi] = np.bincount(vals).argmax()
    else:
        from sklearn.neighbors import KDTree
        V_infer_norm = (V_infer - V_infer.min(0)) / (V_infer.max(0) - V_infer.min(0) + 1e-8) * 2 - 1
        fc_infer = (V_infer_norm[F_infer[:,0]] + V_infer_norm[F_infer[:,1]] + V_infer_norm[F_infer[:,2]]) / 3.0
        pts = V_infer_norm[:len(FL_raw)] if len(FL_raw) <= len(V_infer_norm) else V_infer_norm
        _, idx = KDTree(pts).query(fc_infer, k=1)
        FL_infer = FL_raw[np.clip(idx.flatten(), 0, len(FL_raw)-1)]

    # ── Step 2: if decimated, remap labels to original mesh faces via KDTree
    if was_decimated:
        from sklearn.neighbors import KDTree
        print("Remapping decimated labels → original mesh faces via face-centroid KDTree...")
        # Face centroids of decimated mesh (normalised)
        V_in = (V_infer - V_infer.min(0)) / (V_infer.max(0) - V_infer.min(0) + 1e-8) * 2 - 1
        fc_dec = (V_in[F_infer[:,0]] + V_in[F_infer[:,1]] + V_in[F_infer[:,2]]) / 3.0
        # Face centroids of original mesh (same normalisation)
        fc_orig = (V_norm[F[:,0]] + V_norm[F[:,1]] + V_norm[F[:,2]]) / 3.0
        tree = KDTree(fc_dec)
        _, nn_idx = tree.query(fc_orig, k=1)
        FL = FL_infer[nn_idx.flatten()]
        print(f"Remapped {len(FL):,} original faces from {len(FL_infer):,} decimated labels")
    else:
        FL = FL_infer

    # ── Step 3: remap cluster IDs to 0..N-1 ──────────────────────────────
    unique_ids = np.unique(FL)
    remap_ids  = {int(old_id): new_id for new_id, old_id in enumerate(unique_ids)}
    FL = np.array([remap_ids[v] for v in FL], dtype=int)
    print(f"Final: {len(np.unique(FL))} clusters on {len(FL):,} faces")

    PALETTE = [
        [230,  57,  70], [42,  157, 143], [233, 196, 106], [69,  123, 157],
        [131,  56, 236], [6,   214, 160], [255, 107, 107], [17,  138, 178],
        [247,  37, 133], [58,  134, 255], [251,  86,   7], [138, 201,  38],
        [255, 209, 102], [76,  201, 240], [123,  45, 139], [239,  71, 111],
        [0,   180, 216], [144,  12,  63], [80,  200, 120], [255, 140,   0],
        [100,  50, 200], [20,  150,  80], [200,  80,  20], [50,  100, 200],
    ]

    clusters = []
    for cid in sorted(np.unique(FL).tolist()):
        mask = FL == cid
        fi   = np.where(mask)[0]
        fv   = V_norm[F[fi]].reshape(-1, 3)
        cen  = fv.mean(0)
        bmin = fv.min(0)
        bmax = fv.max(0)
        sz   = bmax - bmin
        clusters.append(dict(
            id           = int(cid),
            face_count   = int(mask.sum()),
            vertex_count = int(np.unique(F[fi]).shape[0]),
            centroid     = [round(float(x), 4) for x in cen],
            bbox_min     = [round(float(x), 4) for x in bmin],
            bbox_max     = [round(float(x), 4) for x in bmax],
            size         = [round(float(x), 4) for x in sz],
            volume       = round(float(sz[0] * sz[1] * sz[2]), 6),
            is_top       = float(cen[1]) >  0.5,
            is_bottom    = float(cen[1]) < -0.5,
            is_left      = float(cen[0]) < -0.3,
            is_right     = float(cen[0]) >  0.3,
            is_front     = float(cen[2]) >  0.3,
            is_back      = float(cen[2]) < -0.3,
        ))

    # Map cluster_id → sequential palette index (guarantees unique colors)
    unique_cids   = sorted(np.unique(FL).tolist())
    cid_to_pidx   = {cid: i for i, cid in enumerate(unique_cids)}

    # Build per-vertex colors via majority-vote over faces
    n_verts  = len(V_norm)
    vc_votes = {}
    for fi, face in enumerate(F):
        cid = int(FL[fi])
        for vi in face:
            vi = int(vi)
            if vi not in vc_votes: vc_votes[vi] = {}
            vc_votes[vi][cid] = vc_votes[vi].get(cid, 0) + 1

    vert_colors = np.zeros((n_verts, 3), dtype=np.uint8)
    for vi in range(n_verts):
        if vi in vc_votes:
            best_cid = max(vc_votes[vi], key=vc_votes[vi].get)
        else:
            best_cid = unique_cids[0]
        rgb = PALETTE[cid_to_pidx[best_cid] % len(PALETTE)]
        vert_colors[vi] = rgb

    # Write the vertex-colored PLY directly here (we already have V, F, vert_colors in memory)
    colored_out = VOL / "uploads" / f"{job_id}_colored.ply"
    colored_out.parent.mkdir(parents=True, exist_ok=True)
    _write_ply_binary(V_norm, F, vert_colors, colored_out)
    print(f"Written colored PLY: {colored_out} ({colored_out.stat().st_size // 1024} KB)")

    # Save V/F/FL to volume for render_and_label to use (can't pass in payload — too large)
    mesh_cache = VOL / "uploads" / f"{job_id}_mesh.npz"
    np.savez_compressed(str(mesh_cache),
                        V=V_norm.astype(np.float32),
                        F=F.astype(np.int32),
                        FL=FL.astype(np.int32))
    print(f"Saved mesh cache: {mesh_cache} ({mesh_cache.stat().st_size//1024} KB)")

    return dict(
        clusters           = clusters,
        FL                 = FL.tolist(),
        colored_model_path = str(colored_out),
        colored_model_ext  = "ply",
    )


def _write_ply_binary(V, F, VC, out_path):
    """Write NON-INDEXED binary PLY: each face = 3 independent vertices.
    This guarantees nV_ply == nF*3, so FL[fi] maps to verts fi*3..fi*3+2
    with no index remapping needed in the viewer.
    """
    import numpy as np
    V  = np.asarray(V,  dtype=np.float32)
    F  = np.asarray(F,  dtype=np.int32)
    VC = np.asarray(VC, dtype=np.uint8)

    nF = len(F)

    # Expand to non-indexed: V_exp[fi*3+k] = V[F[fi,k]]
    V_exp  = V[F.reshape(-1)]          # shape (nF*3, 3)
    VC_exp = VC[F.reshape(-1)]         # shape (nF*3, 3)

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {nF * 3}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"          # NO face element — purely non-indexed
    ).encode("ascii")

    vdtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("r", "u1"),  ("g", "u1"),  ("b", "u1"),
    ])
    vdata = np.empty(nF * 3, dtype=vdtype)
    vdata["x"] = V_exp[:, 0]; vdata["y"] = V_exp[:, 1]; vdata["z"] = V_exp[:, 2]
    vdata["r"] = VC_exp[:, 0]; vdata["g"] = VC_exp[:, 1]; vdata["b"] = VC_exp[:, 2]

    with open(out_path, "wb") as fp:
        fp.write(header)
        fp.write(vdata.tobytes())
        # Non-indexed — no face list needed

    sz = Path(out_path).stat().st_size // 1024
    print(f"Written non-indexed PLY: {out_path} ({sz} KB, {nF} faces, {nF*3} verts)")


# ── CPU: renders + Gemini labelling ──────────────────────────────────────────
@app.function(image=web_image, timeout=600, volumes={VOL: volume},
              memory=4096)   # 4GB RAM for large mesh renders
def render_and_label(job_id: str, cluster_data: dict, category: str) -> dict:
    import numpy as np, base64
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    clusters = cluster_data["clusters"]
    FL       = np.array(cluster_data.get("FL", []), dtype=int)

    PALETTE = [
        "#E63946","#2A9D8F","#E9C46A","#457B9D","#8338EC",
        "#06D6A0","#FF6B6B","#118AB2","#F72585","#3A86FF",
        "#FB5607","#8AC926","#FFD166","#4CC9F0","#7B2D8B",
        "#EF476F","#00B4D8","#90123F","#50C878","#FF8C00",
        "#6432C8","#149650","#C85014","#3264C8",
    ]

    out_dir = VOL / "renders" / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    render_map = {}

    # Load V/F/FL from volume cache saved by run_partfield_gpu
    V, F = None, None
    mesh_cache = VOL / "uploads" / f"{job_id}_mesh.npz"
    try:
        volume.reload()
        if mesh_cache.exists():
            npz = np.load(str(mesh_cache))
            V  = npz["V"].astype(np.float32)
            F  = npz["F"].astype(np.int32)
            FL = npz["FL"].astype(int)   # override FL from npz (authoritative)
            print(f"Loaded mesh cache: V={len(V)}, F={len(F)}, FL={len(FL)}, unique={len(np.unique(FL))}")
        else:
            print(f"WARNING: mesh cache not found at {mesh_cache}")
    except Exception as e:
        print(f"Mesh cache load failed: {e}"); V = F = None

    def _hex_to_rgb01(h):
        h = h.lstrip("#")
        return tuple(int(h[i:i+2], 16)/255.0 for i in (0,2,4))

    # ── Production-grade render helpers ──────────────────────────────────────
    MAX_TRIS = 3000   # per cluster, safe for CPU container

    def _add_faces(ax, face_mask, color, alpha):
        if V is None or F is None or len(np.where(face_mask)[0]) == 0:
            return
        fi = np.where(face_mask)[0]
        if len(fi) > MAX_TRIS:
            fi = fi[np.random.choice(len(fi), MAX_TRIS, replace=False)]
        try:
            tris = V[F[fi]]
            poly = Poly3DCollection(tris, alpha=alpha, linewidth=0)
            poly.set_facecolor(color)
            ax.add_collection3d(poly)
        except Exception as e:
            print(f"Poly3D skip: {e}")

    def _style_ax(ax, verts=None):
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        for p in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            p.fill = False; p.set_edgecolor("none")
        ax.grid(False)
        if verts is not None and len(verts):
            mn, mx = verts.min(0), verts.max(0)
            pad = (mx-mn).max()*0.12 + 0.02
            ax.set_xlim(mn[0]-pad, mx[0]+pad)
            ax.set_ylim(mn[1]-pad, mx[1]+pad)
            ax.set_zlim(mn[2]-pad, mx[2]+pad)
            ax.set_box_aspect([max(0.1,mx[0]-mn[0]+pad*2),
                               max(0.1,mx[1]-mn[1]+pad*2),
                               max(0.1,mx[2]-mn[2]+pad*2)])

    GHOST_COLOR = (0.22, 0.28, 0.35)   # dark blue-grey for dimmed context mesh
    # ═══════════════════════════════════════════════════════════════════════════
    # ORTHOGRAPHIC RENDER STRATEGY: 8 full-mesh views at 896×896
    # Gemma sees the COMPLETE object from every angle with ALL colors.
    # Cluster IDs annotated at centroids so Gemma can map color→ID→name.
    # ═══════════════════════════════════════════════════════════════════════════
    ORTHO_VIEWS = [
        (  0,   0, "front"),   # straight front
        (  0, 180, "back"),    # straight back
        (  0,  90, "right"),   # right side
        (  0, 270, "left"),    # left side
        ( 89,   0, "top"),     # top-down
        (-89,   0, "bottom"),  # bottom-up
        ( 30, 135, "iso_fl"),  # isometric front-left
        ( 30, 315, "iso_fr"),  # isometric front-right
    ]

    print(f"Rendering {len(ORTHO_VIEWS)} orthographic views at 896x896...")
    for view_idx, (elev, azim, vname) in enumerate(ORTHO_VIEWS):
        try:
            fig = plt.figure(figsize=(8, 8), facecolor="#0d1117")
            ax  = fig.add_subplot(111, projection="3d", facecolor="#0d1117")
            ax.view_init(elev=elev, azim=azim)

            if V is not None and F is not None and len(FL) > 0:
                for c in clusters:
                    rgb = _hex_to_rgb01(PALETTE[c["id"] % len(PALETTE)])
                    _add_faces(ax, FL == c["id"], rgb, 0.95)
                _style_ax(ax, V)
                # Annotate cluster ID at centroid so Gemma can map color→ID
                for c in clusters:
                    cen = c["centroid"]
                    col = PALETTE[c["id"] % len(PALETTE)]
                    try:
                        ax.text(float(cen[0]), float(cen[1]), float(cen[2]),
                                str(c["id"]), color=col, fontsize=6,
                                fontweight="bold", ha="center", va="center",
                                bbox=dict(boxstyle="round,pad=0.1", fc="black",
                                          ec=col, alpha=0.7, lw=0.5))
                    except Exception:
                        pass
            else:
                for c in clusters:
                    cen = np.array(c["centroid"])
                    ax.scatter(*cen, color=PALETTE[c["id"]%len(PALETTE)], s=200)

            ax.set_title(
                f"{category}  ·  {vname.upper()}  ·  {len(clusters)} parts",
                color="#aed6f1", fontsize=11, pad=8, fontweight="bold"
            )
            plt.tight_layout(pad=0)
            ov = out_dir / f"view_{vname}.png"
            # 112 dpi × 8 inches = 896px — matches Gemma ViT patch grid
            fig.savefig(ov, dpi=112, bbox_inches="tight",
                        facecolor="#0d1117", edgecolor="none")
            plt.close(fig)
            render_map[-(view_idx+1)] = str(ov)
            print(f"  [{vname:10s}] saved")
        except Exception as e:
            print(f"View {vname} failed: {e}")
            import traceback; traceback.print_exc()

    print(f"All orthographic renders done: {len([k for k in render_map if k<0])} views")

    volume.commit()

    # ── Build images_b64: ALL 8 orthographic views (no per-cluster renders) ──
    # Send 6 most informative views (front/back/right/top + 2 isometric)
    # 8 views × 448×448 still safe; more = better context for Gemma
    imgs_for_gemma = []
    for k in sorted([k for k in render_map if k < 0], reverse=True):
        p = render_map.get(k)
        if p and Path(p).exists():
            with open(p, "rb") as fh:
                imgs_for_gemma.append(base64.b64encode(fh.read()).decode())
    print(f"Images for Gemma: {len(imgs_for_gemma)} orthographic views")

    # ── Try Gemma 3 4B locally first (GPU, fully local, no API limits) ───────
    labels, obj_desc = [], ""
    try:
        print(f"Calling Gemma 3 4B GPU inference with {len(imgs_for_gemma)} images...")
        gemma_result = run_gemma_labelling.remote(clusters, imgs_for_gemma, category)
        labels   = gemma_result.get("labels", [])
        obj_desc = gemma_result.get("object_description", "")
        if labels:
            print(f"✓ Gemma labelled {len(labels)} parts: {[l['label'] for l in labels[:4]]}...")
        else:
            print("Gemma returned empty labels — falling back")
    except Exception as e:
        print(f"Gemma GPU failed: {e}")
        import traceback; traceback.print_exc()

    # ── Fall back to Gemini API if Gemma local failed ─────────────────────────
    if not labels:
        print("Falling back to Gemini API...")
        try:
            result   = _call_gemini(clusters, render_map, category)
            labels   = result.get("labels", [])
            obj_desc = result.get("object_description", "")
        except Exception as e:
            print(f"All labelling failed: {e}")

    # Tag description with which LLM was used (for debugging)
    if labels and obj_desc and "geometric" not in obj_desc.lower():
        source_tag = ""  # LLM worked fine — no tag needed
    else:
        source_tag = " (geometric labels)"
        obj_desc = obj_desc or f"A {category} with {len(labels)} parts{source_tag}."

    return dict(
        labels             = labels,
        object_description = obj_desc,
        render_map         = {str(k): str(v) for k, v in render_map.items()},
    )


def _call_gemini(clusters, render_map, category):
    """Call Gemini / Ollama for semantic part labelling with real 2D renders."""
    import requests, json, base64, time, re

    OLLAMA_KEY = "55943d1e863a4f08b21d6e2e42bb9eae.J20gHHHGj_IcH1Ty4g8amVji"
    BASE = "https://generativelanguage.googleapis.com/v1beta/models"

    # Multiple Gemini keys — rotate on quota exhaustion
    GEMINI_KEYS = [
        "AIzaSyC4KbEW2d7px238zwD9JZgYN6OOm-GEyRQ",
        "AIzaSyDn8517GYmalxfoaLEs5fAH_9hbnpUV3ms",
        "AIzaSyDEba7EyXIz127OV8sRWhsII61tq5f9mA8",
        "AIzaSyAWAOOpQNfz-gEuAxsei3fcwY3O9msojd4",
    ]

    # Build endpoint list: try each key with each model
    ENDPOINTS = []
    for gkey in GEMINI_KEYS:
        ENDPOINTS.append(("gemini_flash", f"{BASE}/gemini-2.0-flash:generateContent", gkey))
        ENDPOINTS.append(("gemini_15",    f"{BASE}/gemini-1.5-flash:generateContent", gkey))
    ENDPOINTS.append(("openrouter", "https://openrouter.ai/api/v1/chat/completions", OLLAMA_KEY))

    PALETTE = [
        "#E63946","#2A9D8F","#E9C46A","#457B9D","#8338EC",
        "#06D6A0","#FF6B6B","#118AB2","#F72585","#3A86FF",
        "#FB5607","#8AC926","#FFD166","#4CC9F0","#7B2D8B",
        "#EF476F","#00B4D8","#90123F","#50C878","#FF8C00",
        "#6432C8","#149650","#C85014","#3264C8",
    ]

    # Build spatial metadata table
    lines = []
    for c in clusters:
        locs = [k for k in ("top","bottom","left","right","front","back") if c.get(f"is_{k}")]
        sz   = c["size"]
        lines.append(
            f"  Cluster {c['id']}: location={'/'.join(locs) or 'center'}, "
            f"centroid=({c['centroid'][0]:.2f},{c['centroid'][1]:.2f},{c['centroid'][2]:.2f}), "
            f"size={sz[0]:.2f}x{sz[1]:.2f}x{sz[2]:.2f}, "
            f"volume={c['volume']:.4f}, faces={c['face_count']}"
        )

    metadata_text = "\n".join(lines)

    # Load renders as base64
    images_b64 = []
    # All 3 overview angles first (front, side, top)
    for k in [-1, -2, -3]:
        p = render_map.get(k)
        if p and Path(p).exists():
            with open(p, "rb") as f:
                images_b64.append(base64.b64encode(f.read()).decode())
    # Per-cluster renders
    for c in clusters:
        p = render_map.get(c["id"])
        if p and Path(p).exists():
            with open(p, "rb") as f:
                images_b64.append(base64.b64encode(f.read()).decode())
    print(f"Total images for LLM: {len(images_b64)}")


    # Category-specific part name examples for accurate labelling
    EXAMPLES = {
        "chair":      "seat, backrest, left_armrest, right_armrest, front_left_leg, front_right_leg, rear_left_leg, rear_right_leg, seat_cushion, leg_stretcher",
        "car":        "hood, roof, trunk_lid, front_bumper, rear_bumper, left_front_door, right_front_door, windshield, front_left_wheel, front_right_wheel, rear_left_wheel, rear_right_wheel",
        "wagon":      "water_tank, front_left_wheel, front_right_wheel, rear_left_wheel, rear_right_wheel, front_axle, rear_axle, tongue_pole, frame_rail_left, frame_rail_right, tank_band, wheel_hub, driver_seat",
        "water wagon":"water_tank, front_left_wheel, front_right_wheel, rear_left_wheel, rear_right_wheel, front_axle, rear_axle, tongue_pole, frame_rail, tank_band, wheel_hub",
        "man":        "head, hair, torso_shirt, left_upper_arm, left_forearm, left_hand, right_upper_arm, right_forearm, right_hand, pants_pelvis, left_thigh, left_shin, left_shoe, right_thigh, right_shin, right_shoe, glasses, collar",
        "woman":      "head, hair, torso, left_upper_arm, left_forearm, left_hand, right_upper_arm, right_forearm, right_hand, skirt_dress, left_leg, right_leg, left_shoe, right_shoe",
        "character":  "head, hair, torso, left_upper_arm, left_forearm, left_hand, right_upper_arm, right_forearm, right_hand, pelvis, left_thigh, left_shin, left_foot, right_thigh, right_shin, right_foot, accessory",
        "robot":      "head, visor, torso, left_shoulder, left_upper_arm, left_forearm, left_hand, right_shoulder, right_upper_arm, right_forearm, right_hand, pelvis, left_thigh, left_shin, left_foot, right_thigh, right_shin, right_foot, antenna",
        "scissors":   "left_blade, right_blade, left_finger_ring, right_thumb_ring, pivot_screw",
        "watch":      "watch_case, dial_face, crown_button, strap_top, strap_bottom, buckle_clasp, bezel_ring, crystal_glass",
        "bicycle":    "frame, front_wheel, rear_wheel, handlebar, seat, front_fork, chain_guard, pedal_left, pedal_right",
        "airplane":   "fuselage, left_wing, right_wing, vertical_tail, left_elevator, right_elevator, nose_cone, left_engine, right_engine",
        "lamp":       "base, pole, shade, bulb_socket, switch",
        "table":      "tabletop, front_left_leg, front_right_leg, rear_left_leg, rear_right_leg, apron",
        "sword":      "blade, crossguard, grip_handle, pommel, basket_guard, ricasso",
        "knife":      "blade, crossguard, grip_handle, pommel, bolster",
        "gun":        "barrel, slide, frame, grip, trigger, trigger_guard, magazine",
        "helmet":     "dome, visor, chin_guard, ear_guard, neck_guard, crest",
        "car":        "hood, roof, trunk_lid, front_bumper, rear_bumper, windshield, front_left_wheel, rear_right_wheel",
        "house":      "roof, front_wall, left_wall, right_wall, door, window_front, chimney, foundation",
    }
    cat_low = category.lower()
    example_str = next((v for k,v in EXAMPLES.items() if k in cat_low or cat_low in k), None)
    if not example_str:
        example_str = "main_body, upper_section, lower_section, left_part, right_part, front_part, rear_part, detail_1, detail_2"

    prompt = (
        f"You are a world-class 3D semantic segmentation expert.\n\n"
        f"OBJECT TYPE: '{category}'\n"
        f"SEGMENTS: {len(clusters)} parts (cluster IDs: {[c['id'] for c in clusters]})\n\n"
        f"IMAGES PROVIDED:\n"
        f"- Images 1-3: Front view, Side view, Top view of ALL {len(clusters)} colored segments\n"
        f"- Images 4+: Each individual segment isolated and highlighted (others dimmed)\n\n"
        f"SPATIAL DATA (y=up axis, +x=right, +z=front):\n{metadata_text}\n\n"
        f"REFERENCE PART NAMES FOR '{category}':\n{example_str}\n\n"
        f"INSTRUCTIONS:\n"
        f"1. Look at each image to understand what body part/component is shown\n"
        f"2. Use the spatial data: top(y>0.5)=head/top, bottom(y<-0.5)=feet/base, "
        f"left(x<-0.3)=left side, right(x>0.3)=right side\n"
        f"3. Largest face_count cluster = main body (torso/tank/frame/hull)\n"
        f"4. NEVER use: part_0, part_1, component, structure, section, piece\n"
        f"5. NO duplicate names — use _left/_right/_front/_rear suffixes\n"
        f"6. ALL {len(clusters)} cluster IDs must appear exactly once\n\n"
        f"OUTPUT: ONLY valid JSON, no markdown, no text outside the JSON:\n"
        '{{"object_description":"<one precise sentence about what this model is>","labels":['
        '{{"cluster_id":<int>,"label":"<specific_name_like_left_forearm>","color":"#RRGGBB",'
        '"description":"<what this specific part is>","confidence":<0.0-1.0>}}'
        ']}}'
    )

    imgs_b64 = images_b64[:12]   # max 8 renders


    last_err = None
    for ep_name, ep_url, api_key in ENDPOINTS:
        for attempt in range(2):
            try:
                print(f"Trying {ep_name} attempt {attempt+1} ({len(imgs_b64)} images)...")

                if ep_name.startswith("gemini"):
                    # Gemini REST API: parts list with inline image data
                    parts_content = [{"text": prompt}]
                    for b64 in imgs_b64:
                        parts_content.append({
                            "inline_data": {"mime_type": "image/png", "data": b64}
                        })
                    payload = {
                        "contents": [{"parts": parts_content}],
                        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 4096}
                    }
                    resp = requests.post(
                        f"{ep_url}?key={api_key}",
                        json=payload,
                        headers={"Content-Type": "application/json"},
                        timeout=120,
                    )
                    print(f"  Gemini HTTP {resp.status_code}")
                    if resp.status_code != 200:
                        print(f"  Gemini error body: {resp.text[:500]}")
                    resp.raise_for_status()
                    rj   = resp.json()
                    cands = rj.get("candidates", [])
                    if not cands:
                        print(f"  Gemini no candidates: {rj}")
                        raise ValueError("No candidates in Gemini response")
                    text = cands[0]["content"]["parts"][0]["text"].strip()
                    print(f"  Gemini response ({len(text)} chars): {text[:200]}")

                else:  # openrouter — OpenAI-compatible
                    msgs = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
                    for b64 in imgs_b64:
                        msgs[0]["content"].append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"}
                        })
                    payload = {"model": "google/gemma-3-4b-it:free",
                               "messages": msgs, "temperature": 0.1, "max_tokens": 4096}
                    resp = requests.post(ep_url, json=payload,
                        headers={"Authorization": f"Bearer {api_key}",
                                 "Content-Type": "application/json",
                                 "HTTP-Referer": "https://partfield-studio.modal.run"},
                        timeout=120)
                    resp.raise_for_status()
                    text = resp.json()["choices"][0]["message"]["content"].strip()

                # Strip markdown fences
                if "```" in text:
                    for part in text.split("```"):
                        p2 = part.strip().lstrip("json").strip()
                        if p2.startswith("{"):
                            text = p2; break

                parsed   = json.loads(text)
                labels   = parsed.get("labels", [])
                obj_desc = parsed.get("object_description", "")
                print(f"{ep_name} success! {len(labels)} labels: {obj_desc[:80]}")

                # Ensure every cluster has a label + valid color
                labelled_ids = {l["cluster_id"] for l in labels}
                for c in clusters:
                    if c["id"] not in labelled_ids:
                        labels.append(dict(
                            cluster_id  = c["id"],
                            label       = f"part_{c['id']}",
                            color       = PALETTE[c["id"] % len(PALETTE)],
                            description = "Unlabelled segment.",
                            confidence  = 0.0,
                        ))
                # Always assign colors from palette by cluster_id — guarantees uniqueness
                # (LLM often returns same hex for multiple parts)
                for lbl in labels:
                    lbl["color"] = PALETTE[lbl["cluster_id"] % len(PALETTE)]

                # Deduplicate names: suffix with spatial position if same name used twice
                seen_names = {}
                for lbl in sorted(labels, key=lambda x: x["cluster_id"]):
                    name = lbl.get("label", "").lower().strip()
                    if name in seen_names:
                        seen_names[name] += 1
                        c = next((x for x in clusters if x["id"] == lbl["cluster_id"]), {})
                        locs = [k for k in ("left","right","front","back","top","bottom")
                                if c.get(f"is_{k}")]
                        suffix = "_".join(locs) if locs else str(seen_names[name])
                        lbl["label"] = f"{name}_{suffix}"
                    else:
                        seen_names[name] = 1

                return dict(labels=labels, object_description=obj_desc)

            except Exception as e:
                last_err = str(e)
                err_str = str(e).lower()
                print(f"{ep_name} attempt {attempt+1} failed: {e}")
                # Skip all remaining attempts for this key if quota/billing issue
                if any(x in err_str for x in ["quota", "429", "billing", "exhausted", "limit"]):
                    print(f"  Quota/billing issue on {ep_name} — skipping to next key")
                    break
                time.sleep(1.5 ** attempt)

    print(f"All LLM endpoints failed: {last_err}. Using geometric fallback.")
    return dict(
        labels             = _geometric_labels(clusters, category),
        object_description = f"A {category} with {len(clusters)} parts (geometric labels).",
    )


def _geometric_labels(clusters, category):
    """Spatial geometry labelling — always works, no API needed."""
    PALETTE = [
        "#E63946","#2A9D8F","#E9C46A","#457B9D","#8338EC",
        "#06D6A0","#FF6B6B","#118AB2","#F72585","#3A86FF",
        "#FB5607","#8AC926","#FFD166","#4CC9F0","#7B2D8B",
        "#EF476F","#00B4D8","#90123F","#50C878","#FF8C00",
        "#6432C8","#149650","#C85014","#3264C8",
    ]
    PARTS = {
        "chair":     ["seat","backrest","left_armrest","right_armrest",
                      "front_left_leg","front_right_leg","rear_left_leg","rear_right_leg",
                      "seat_base","leg_connector"],
        "car":       ["hood","roof","trunk","front_bumper","rear_bumper",
                      "left_front_door","right_front_door","windshield",
                      "left_rear_door","right_rear_door","chassis",
                      "left_front_wheel","right_front_wheel","left_rear_wheel","right_rear_wheel"],
        "table":     ["tabletop","front_left_leg","front_right_leg",
                      "rear_left_leg","rear_right_leg","apron","shelf"],
        "scissors":  ["left_blade","right_blade","left_finger_ring",
                      "right_thumb_ring","pivot_screw","blade_tip_left","blade_tip_right"],
        "watch":     ["watch_case","dial_face","crown_button","strap_top",
                      "strap_bottom","buckle_clasp","bezel_ring","crystal_glass",
                      "lug_top_left","lug_top_right","lug_bottom_left","lug_bottom_right","case_back"],
        "wagon":     ["water_tank","front_left_wheel","front_right_wheel",
                      "rear_left_wheel","rear_right_wheel","front_axle","rear_axle",
                      "tongue_pole","frame_rail_left","frame_rail_right",
                      "wheel_hub","spoke_set","seat_board"],
        "robot":     ["head","torso","left_upper_arm","left_forearm","left_hand",
                      "right_upper_arm","right_forearm","right_hand",
                      "left_thigh","left_shin","left_foot",
                      "right_thigh","right_shin","right_foot","antenna"],
        "ring":      ["band","setting","center_stone","side_stone_left",
                      "side_stone_right","prong_set","shank","shoulder_left","shoulder_right"],
        "character": ["head","hair","torso","left_upper_arm","left_forearm",
                      "left_hand","right_upper_arm","right_forearm","right_hand",
                      "pelvis","left_thigh","left_shin","left_foot",
                      "right_thigh","right_shin","right_foot"],
        "lamp":      ["base","pole","shade","bulb","switch","cord"],
        "airplane":  ["fuselage","left_wing","right_wing","tail_fin",
                      "left_elevator","right_elevator","nose_cone",
                      "left_engine","right_engine","landing_gear"],
        "bike":      ["frame","front_wheel","rear_wheel","handlebar","seat",
                      "front_fork","chain","pedal_left","pedal_right","brake_lever"],
    }

    part_names = PARTS.get(category.lower(), [])
    sorted_c   = sorted(clusters, key=lambda c: -c["volume"])
    labels     = []

    for rank, c in enumerate(sorted_c):
        cid  = c["id"]
        col  = PALETTE[cid % len(PALETTE)]   # cid is 0..N-1 so always unique
        locs = [k for k in ("top","bottom","left","right","front","back") if c.get(f"is_{k}")]
        loc  = "_".join(locs) if locs else "center"

        if rank < len(part_names):
            name = part_names[rank]
            desc = f"{name.replace('_',' ').title()} of the {category}."
            conf = 0.60
        elif rank == 0:
            name, desc, conf = "main_body", f"Primary structural body.", 0.75
        elif c.get("is_top"):
            name, desc, conf = "top_section", "Upper section.", 0.60
        elif c.get("is_bottom"):
            name, desc, conf = f"base_{rank}", "Lower base structure.", 0.60
        elif c.get("is_left"):
            name, desc, conf = f"left_part_{rank}", "Left-side component.", 0.55
        elif c.get("is_right"):
            name, desc, conf = f"right_part_{rank}", "Right-side component.", 0.55
        elif c.get("is_front"):
            name, desc, conf = f"front_{rank}", "Front-facing component.", 0.55
        elif c.get("is_back"):
            name, desc, conf = f"rear_{rank}", "Rear-facing component.", 0.55
        else:
            name, desc, conf = f"inner_part_{rank}", "Interior structural component.", 0.45

        labels.append(dict(
            cluster_id  = cid,
            label       = name,
            color       = col,
            description = desc,
            confidence  = conf,
        ))

    return labels


# ── Flask web endpoint ────────────────────────────────────────────────────────
@app.function(image=web_image, volumes={VOL: volume}, timeout=60)
@modal.concurrent(max_inputs=50)
@modal.wsgi_app()
def web():
    import uuid, threading
    from flask import Flask, request, jsonify, render_template, send_from_directory
    from flask_cors import CORS

    flask_app = Flask(
        __name__,
        template_folder="/app/templates",
        static_folder="/app/static",
    )
    CORS(flask_app)
    flask_app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024

    ALLOWED = {"obj", "glb", "gltf", "ply", "stl"}
    JOBS: dict = {}

    def get_ext(name):
        return name.rsplit(".", 1)[-1].lower() if "." in name else ""

    @flask_app.route("/")
    def index():
        return render_template("index.html")

    @flask_app.route("/static/<path:p>")
    def static_serve(p):
        return send_from_directory("/app/static", p)

    @flask_app.route("/renders/<path:p>")
    def render_serve(p):
        renders_root = str(VOL / "renders")
        return send_from_directory(renders_root, p)

    @flask_app.route("/model/<jid>")
    def serve_model(jid):
        j = JOBS.get(jid)
        if not j:
            return "Not found", 404
        fp = VOL / "uploads" / j["filename"]
        if not fp.exists():
            return f"File not found: {fp}", 404
        return send_from_directory(str(fp.parent), fp.name)

    @flask_app.route("/colored/<jid>")
    def serve_colored(jid):
        # Try job result first, then fall back to known disk path for this jid
        cp = None
        j  = JOBS.get(jid)
        if j and j.get("result"):
            cp = j["result"].get("colored_model_path", "")

        # Always also check the canonical disk path (written by GPU worker directly)
        canonical = VOL / "uploads" / f"{jid}_colored.ply"
        if not cp or not Path(cp).exists():
            if canonical.exists():
                cp = str(canonical)

        if not cp or not Path(cp).exists():
            return f"Colored PLY not found for job {jid}", 404

        volume.reload()  # ensure we see latest writes from GPU worker
        if not Path(cp).exists():
            return "Colored PLY not found after reload", 404

        return send_from_directory(str(Path(cp).parent), Path(cp).name)

    @flask_app.route("/api/upload", methods=["POST"])
    def upload():
        f        = request.files.get("model")
        category = request.form.get("category", "object").strip() or "object"
        n_parts  = max(4, min(24, int(request.form.get("n_parts", 12))))
        if not f or not f.filename:
            return jsonify(error="No file"), 400
        e = get_ext(f.filename)
        if e not in ALLOWED:
            return jsonify(error=f"Unsupported .{e}"), 400

        # ── Purge old completed/errored jobs to prevent state bleed ──────────
        stale = [k for k, v in list(JOBS.items())
                 if v.get("status") in ("done", "error")]
        for k in stale:
            old_job = JOBS.pop(k, None)
            if old_job:
                try:
                    for fname in [old_job.get("filename",""),
                                  f"{k}_colored.ply",
                                  f"{k}_mesh.npz"]:
                        p = VOL / "uploads" / fname
                        if fname and p.exists(): p.unlink()
                except Exception:
                    pass
        # ────────────────────────────────────────────────────────────────────

        jid   = str(uuid.uuid4())[:8]
        fname = f"{jid}.{e}"
        fpath = VOL / "uploads" / fname
        fpath.parent.mkdir(parents=True, exist_ok=True)
        f.save(str(fpath))
        volume.commit()

        JOBS[jid] = dict(
            id       = jid,
            status   = "queued",
            stage    = "Queued",
            progress = 0,
            category = category,
            n_parts  = n_parts,
            filename = fname,
            error    = None,
            result   = None,
        )

        threading.Thread(
            target=_run_pipeline,
            args=(jid, fpath, category, n_parts, JOBS),
            daemon=True,
        ).start()

        return jsonify(job_id=jid)

    @flask_app.route("/api/status/<jid>")
    def status(jid):
        j = JOBS.get(jid)
        if not j:
            return jsonify(error="Not found"), 404
        # Return job without FL (FL is large, served separately via /api/fl/<jid>)
        safe = {k: v for k, v in j.items() if k != "_FL"}
        return jsonify(safe)

    @flask_app.route("/api/fl/<jid>")
    def get_fl(jid):
        """Return the face→cluster label array for a completed job."""
        j = JOBS.get(jid)
        if not j:
            return jsonify(error="Not found"), 404
        fl = j.get("_FL", [])
        return jsonify(fl=fl, length=len(fl))

    @flask_app.route("/api/recommend", methods=["POST"])
    def recommend_parts():
        """AI-recommended part count — uses image preview + category."""
        import base64, requests as req, io
        f        = request.files.get("preview")
        category = request.form.get("category", "").strip()
        if not category:
            return jsonify(parts=12, reason="No category provided")

        img_b64 = None
        if f:
            raw = f.read()
            ext = (f.filename or "").rsplit(".", 1)[-1].lower()
            # Only use as image if it looks like an image (jpeg/png/webp)
            if ext in ("jpg", "jpeg", "png", "webp") and len(raw) > 1000:
                img_b64 = base64.b64encode(raw).decode()
            elif len(raw) > 1000:
                # Non-image file (GLB etc.) — skip vision, use text-only
                img_b64 = None

        # Ask Ollama/OpenRouter to recommend part count
        prompt = (
            f"You are a 3D segmentation expert. A user has uploaded a 3D model of a '{category}'. "
            f"I am showing you a preview image of this model. "
            f"Based on the visible geometry and the object type, recommend the EXACT number of "
            f"distinct parts this model should be segmented into for best semantic accuracy. "
            f"Consider: wheels count as individual parts, legs are separate, panels are separate. "
            f"Reply ONLY with a JSON object: {{\"parts\": <integer 4-24>, \"reason\": \"<one sentence>\"}}"
        )

        GEMINI_KEY = "AIzaSyC4KbEW2d7px238zwD9JZgYN6OOm-GEyRQ"
        import json, re

        # Try Gemini (vision if image available, text-only otherwise)
        for model in ["gemini-2.0-flash", "gemini-1.5-flash"]:
            try:
                parts_content = [{"text": prompt}]
                if img_b64:
                    parts_content.append({"inline_data": {"mime_type": "image/jpeg", "data": img_b64}})
                payload = {
                    "contents": [{"parts": parts_content}],
                    "generationConfig": {"temperature": 0.1, "maxOutputTokens": 256}
                }
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_KEY}"
                r = req.post(url, json=payload,
                             headers={"Content-Type": "application/json"}, timeout=30)
                r.raise_for_status()
                text = r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                m = re.search(r"\{[^{}]+\}", text)
                if m:
                    d = json.loads(m.group())
                    parts = max(4, min(24, int(d.get("parts", 12))))
                    return jsonify(parts=parts, reason=d.get("reason", "AI recommendation"))
            except Exception as e:
                print(f"Recommend {model} failed: {e}")

        # Fallback: rule-based recommendation
        rules = {
            "chair": 8, "car": 14, "truck": 12, "bicycle": 10, "bike": 10,
            "table": 6, "lamp": 6, "wagon": 13, "water wagon": 13,
            "airplane": 12, "plane": 12, "robot": 14, "character": 16,
            "scissors": 5, "watch": 10, "ring": 5, "bottle": 4, "cup": 4,
        }
        for k, v in rules.items():
            if k in category.lower():
                return jsonify(parts=v, reason=f"Rule-based recommendation for {category}.")
        return jsonify(parts=10, reason="Default recommendation.")

    def _run_pipeline(jid, fpath, category, n_parts, JOBS):
        def upd(**kw):
            JOBS[jid].update(kw)

        try:
            upd(status="running", stage="Submitting to PartField GPU...", progress=3)
            cluster_data = run_partfield_gpu.remote(jid, str(fpath), n_parts)
            upd(stage="GPU segmentation complete. Rendering + Gemma 3 4B labelling...", progress=62)

            # Labelling — non-fatal if it fails
            label_result = None
            try:
                print(f"[{jid}] Starting render_and_label (Gemma 3 4B)...")
                label_result = render_and_label.remote(jid, cluster_data, category)
                print(f"[{jid}] render_and_label completed")
            except Exception as le:
                print(f"[{jid}] Labelling failed: {le}")
                import traceback; traceback.print_exc()

            upd(stage="Finalising result...", progress=92)

            # Build renders list
            renders = []
            if label_result and label_result.get("render_map"):
                for cid_str, local_path in label_result["render_map"].items():
                    try:
                        cid = int(cid_str)
                        rel = Path(local_path).relative_to(VOL / "renders")
                        renders.append({"cluster_id": cid, "url": f"/renders/{rel}"})
                    except Exception:
                        pass

            # Labels — Gemini result or geometric fallback
            labels   = (label_result or {}).get("labels", []) or []
            obj_desc = (label_result or {}).get("object_description", "")

            if not labels:
                labels = _geometric_labels(cluster_data["clusters"], category)

            # Colored PLY paths
            colored_path = cluster_data.get("colored_model_path", "")
            colored_ext  = cluster_data.get("colored_model_ext", "")

            # Store FL separately — it can be 60k+ ints and bloats every poll response
            FL_data = cluster_data.get("FL", [])
            JOBS[jid]["_FL"] = FL_data   # stored outside result, served via /api/fl/<jid>

            result = dict(
                job_id               = jid,
                category             = category,
                model_ext            = fpath.suffix.lstrip("."),
                colored_model_path   = colored_path,
                colored_model_ext    = colored_ext,
                clusters             = cluster_data["clusters"],
                labels               = labels,
                object_description   = obj_desc,
                renders              = renders,
                has_fl               = len(FL_data) > 0,  # signal to frontend
            )

            upd(status="done", stage="Complete", progress=100, result=result)
            JOBS[jid]["result"] = result  # store for /colored/ route

        except Exception as e:
            import traceback; traceback.print_exc()
            upd(status="error", stage=f"Error: {e}", error=str(e), progress=100)

    return flask_app
