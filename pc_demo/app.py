"""Streamlit PC demo for BirdID using uploaded files."""
from __future__ import annotations

import base64
import io
import logging
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import streamlit as st

# Ensure repository root is on sys.path when running via Streamlit or directly
# so that `import birdid.*` works even if the package is not installed.
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from birdid.decision import decide
from birdid.embedding_baseline import compute_baseline_embedding
from birdid.engine import BirdIDConfig, Calibration
from birdid.inputs.file_adapter import build_inputs, load_rgb_image
from birdid.db import BirdIDDatabase, MatchResult
from birdid.segment import DepthSegmenter
from birdid.validate_3d import validate_tracklet
from birdid.firebase_logger import is_available as firebase_available, upload_artifacts


LOGGER = logging.getLogger(__name__)


@st.cache_resource
def _load_yolo_model():
    best_path = ROOT / "best.pt"
    if not best_path.exists():
        LOGGER.info("best.pt not found at %s; auto species disabled", best_path)
        return None
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        LOGGER.warning("ultralytics not installed; auto species disabled (%s)", exc)
        return None
    try:
        return YOLO(str(best_path))
    except Exception as exc:  # pragma: no cover - model load error
        LOGGER.warning("Failed to load best.pt: %s", exc)
        return None


def _auto_detect_species(rgb: np.ndarray):
    model = _load_yolo_model()
    if model is None:
        return None
    try:
        results = model.predict(rgb, verbose=False)
    except Exception as exc:  # pragma: no cover - inference error
        LOGGER.warning("Auto species prediction failed: %s", exc)
        return None
    if not results:
        return None
    res = results[0]
    boxes = getattr(res, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return None
    confs = boxes.conf.cpu().numpy()
    best_idx = int(np.argmax(confs))
    xyxy = boxes.xyxy[best_idx].cpu().numpy().tolist()
    cls_idx = int(boxes.cls[best_idx].item())
    names = getattr(model, "names", getattr(res, "names", {}))
    species = str(names.get(cls_idx, cls_idx))
    return {"species": species, "bbox": xyxy, "conf": float(confs[best_idx])}


def _get_engine_config() -> BirdIDConfig:
    cfg = BirdIDConfig()
    cfg.min_frames = 1
    cfg.aggregate_stride = 1
    cfg.min_frames_for_dispense = 1
    return cfg


def _overlay_bbox(rgb: np.ndarray, bbox) -> bytes:
    try:
        import cv2  # type: ignore
    except Exception:
        return _to_png(rgb)
    x1, y1, x2, y2 = map(int, bbox)
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    _, buf = cv2.imencode(".png", img)
    return bytes(buf)


def _render_depth_stats(depth: np.ndarray, mask: np.ndarray) -> str:
    if mask.sum() == 0:
        return "No valid depth"
    vals = depth[mask]
    return f"depth mean={vals.mean():.3f} m, std={vals.std():.3f}, min={vals.min():.3f}, max={vals.max():.3f}"


def _compute_embedding(embedder, depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
    emb = embedder(depth, mask)
    return emb / (np.linalg.norm(emb) + 1e-8)


def run_pipeline(
    cfg: BirdIDConfig,
    db: BirdIDDatabase,
    embedder,
    embedder_backend: str,
    species: str,
    detection,
    depth_frame,
    bbox,
    simulate_known: bool,
    force_enroll: bool,
    trusted_ply_input: bool,
) -> tuple[dict, Optional[np.ndarray]]:
    calib = Calibration()
    segmenter = DepthSegmenter(min_valid_ratio=cfg.min_valid_ratio)
    # For the PC demo we avoid depth cropping to reduce "size_out_of_bounds"
    # rejections when auto-detected boxes are too tight. Use the full depth
    # frame while still honoring the bbox for overlays/metadata.
    depth_crop = depth_frame.depth
    valid_crop = depth_frame.valid
    depth_clean, mask = segmenter.segment(depth_crop, valid_crop)
    if mask.sum() == 0 and valid_crop.any():
        LOGGER.info("Segmentation empty; using raw valid depth crop as fallback")
        depth_clean = depth_crop.astype(np.float32, copy=False)
        mask = valid_crop.astype(bool, copy=False)
    validation = validate_tracklet(
        [(depth_clean, mask)],
        cfg,
        point_cloud=depth_frame.point_cloud,
        trusted_ply_input=trusted_ply_input,
        point_units=cfg.pointcloud_units,
    )
    if mask.sum() == 0:
        embedding = np.zeros(1, dtype=np.float32)
    else:
        embedding = _compute_embedding(embedder, depth_clean, mask)

    # Optionally seed a known prototype using the uploaded embedding.
    if simulate_known and not db.get_prototypes(species):
        db.add_individual(species, embedding)

    prototypes = db.get_prototypes(species)
    from birdid.matching import topk_matches

    neighbors = topk_matches(embedding, prototypes, k=5) if prototypes else []
    match: MatchResult
    if force_enroll:
        match = MatchResult(None, float("inf"), 0.0, float("inf"), False, False)
    else:
        match = db.match(species, embedding, cfg.cosine_threshold)

    now_ts = time.time()
    if match.individual_id is not None:
        db.mark_seen(match.individual_id, now_ts)
    refresh_ok = (
        validation.is_valid
        and match.individual_id is not None
        and match.is_match
        and match.confidence >= cfg.refresh_confidence_threshold
        and validation.quality_score >= cfg.refresh_quality_min
        and db.is_refresh_stale(match.individual_id, now_ts, cfg.refresh_days)
    )
    if refresh_ok:
        db.update_prototype(match.individual_id, embedding, cfg.ema, cfg.max_prototypes, now_ts=now_ts)

    decision = decide(
        species=species,
        track_id=int(detection.get("track_id", 1)),
        embedding=embedding,
        match=match,
        validation=validation,
        frames_used=1,
        config=cfg,
        db=db,
        now_ts=now_ts,
    )

    result = {
        "species": species,
        "track_id": detection.get("track_id"),
        "individual_id": decision.individual_id,
        "decision": decision.decision,
        "deny_reason": decision.deny_reason,
        "min_dist": match.distance,
        "second_best_dist": match.second_best,
        "margin": decision.margin,
        "quality_score": validation.quality_score,
        "frames_used": 1,
        "cooldown_remaining": decision.cooldown_remaining,
        "roi_debug": {
            "depth_shape": depth_crop.shape[:2],
            "percent_valid_depth": float(valid_crop.mean()) if valid_crop.size else 0.0,
        },
        "diagnostics": validation.diagnostics,
        "neighbors": [n.__dict__ for n in neighbors],
        "embedding_norm": float(np.linalg.norm(embedding)) if embedding.size else 0.0,
        "embedding_checksum": float(embedding[:4].sum()) if embedding.size else 0.0,
        "embedder_backend": embedder_backend,
    }
    return result, depth_clean if mask.size else None


def _to_png(rgb: np.ndarray) -> bytes:
    try:
        from PIL import Image  # type: ignore
    except Exception:
        return b""
    img = Image.fromarray(rgb.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def main() -> None:
    st.title("BirdID PC Demo")
    st.write("Upload an RGB image and depth/point cloud to run the BirdID pipeline.")

    cfg = _get_engine_config()
    from birdid.embedding_tflite import load_embedder

    embedder, embedder_backend, model_path = load_embedder(
        model_dir=Path(cfg.tflite_model_path).parent,
        model_name=Path(cfg.tflite_model_path).name,
        input_size=cfg.tflite_input_size,
    )
    db_path = Path(st.session_state.get("birdid_db_path", ".pc_demo.sqlite"))
    st.session_state["birdid_db_path"] = str(db_path)
    db = BirdIDDatabase(db_path)

    st.info(f"Embedding backend: {embedder_backend} ({model_path})")

    species = st.text_input("Species", value="sparrow")
    auto_species = st.checkbox("Auto-detect species with best.pt (if available)", value=True)
    simulate_known = st.checkbox("Simulate known bird (seed DB if empty)", value=False)
    force_enroll = st.checkbox("Force enroll (ignore matches)", value=False)
    trusted_ply = st.checkbox("Trusted PLY input (skip planar spoof for uploads)", value=False)
    show_debug = st.checkbox("Show debug visuals", value=True)

    rgb_file = st.file_uploader("RGB image", type=["png", "jpg", "jpeg"])
    depth_file = st.file_uploader("Depth / point cloud", type=["png", "jpg", "jpeg", "npy", "ply", "pcd"])

    bbox_inputs = st.text_input("BBox x1,y1,x2,y2 (optional)", value="")
    bbox: Optional[list[float]] = None
    if bbox_inputs.strip():
        try:
            parts = [float(p) for p in bbox_inputs.split(",")]
            if len(parts) == 4:
                bbox = parts
        except Exception:
            st.warning("Invalid bbox format; using full image")

    if st.button("Run BirdID"):
        if rgb_file is None or depth_file is None:
            st.error("Please upload both RGB and depth/point cloud files.")
            return
        try:
            auto_detection = None
            if auto_species:
                auto_detection = _auto_detect_species(load_rgb_image(rgb_file))
                if auto_detection:
                    species = auto_detection["species"]
                    bbox = auto_detection["bbox"]
                    st.info(
                        f"Auto species: {species} (conf={auto_detection['conf']:.2f}) | bbox={bbox}"
                    )
                else:
                    st.info("Auto species unavailable; using manual inputs")
            conf = auto_detection["conf"] if auto_detection else 0.9
            inputs = build_inputs(
                rgb_file,
                depth_file,
                species=species,
                bbox_xyxy=bbox,
                conf=conf,
                point_units=cfg.pointcloud_units,
            )
        except Exception as exc:
            st.error(f"Failed to parse inputs: {exc}")
            return

        result, depth_clean = run_pipeline(
            cfg=cfg,
            db=db,
            embedder=embedder,
            embedder_backend=embedder_backend,
            species=species,
            detection=inputs.detection,
            depth_frame=inputs.depth_frame,
            bbox=inputs.bbox_xyxy,
            simulate_known=simulate_known,
            force_enroll=force_enroll,
            trusted_ply_input=trusted_ply,
        )

        st.subheader("Decision")
        st.json(result)
        st.write(
            f"Quality score: {result['quality_score']:.3f} | min_dist: {result['min_dist']:.3f} | margin: {result['margin']:.3f}"
        )
        with st.expander("Similarity Debug", expanded=True):
            st.write(f"Embedding backend: {result.get('embedder_backend', '?')}")
            st.write(f"Embedding norm: {result.get('embedding_norm', 0):.4f} | checksum: {result.get('embedding_checksum', 0):.4f}")
            neighbors = result.get("neighbors", [])
            if neighbors:
                st.table({"individual_id": [n["individual_id"] for n in neighbors], "distance": [n["distance"] for n in neighbors]})
            else:
                st.write("No stored prototypes for this species yet.")
        with st.expander("ROI Debug", expanded=False):
            roi_dbg = result.get("roi_debug", {})
            st.write(
                f"Depth shape: {roi_dbg.get('depth_shape', '?')} | percent valid depth: {roi_dbg.get('percent_valid_depth', 0):.3f}"
            )
            diag = result.get("diagnostics", {})
            if diag:
                st.write(
                    "Diagnostics: "
                    f"scattering={diag.get('scattering', 0):.5f}, planarity_metric={diag.get('planarity_metric', 0):.5f}, "
                    f"linearity={diag.get('linearity', 0):.5f}, thickness={diag.get('thickness', diag.get('planar_thickness', 0)):.4f}, "
                    f"points={diag.get('points', 0):.0f}"
                )

        if show_debug:
            st.subheader("Debug")
            st.image(inputs.rgb, caption="RGB input")
            overlay_bytes = _overlay_bbox(inputs.rgb, inputs.bbox_xyxy)
            if overlay_bytes:
                st.image(io.BytesIO(overlay_bytes), caption="RGB + bbox")
            if depth_clean is not None:
                st.write(_render_depth_stats(depth_clean, depth_clean > 0))
                st.image(depth_clean, caption="Depth crop", clamp=True)
            if inputs.point_cloud is not None:
                st.write(f"Point cloud points: {inputs.point_cloud.shape[0]}")

        if firebase_available():
            run_id = base64.urlsafe_b64encode(os.urandom(6)).decode("utf-8")
            metadata = {**result, "run_id": run_id, "timestamp": time.time()}
            image_bytes = _to_png(inputs.rgb)
            pc_bytes = None
            if inputs.point_cloud is not None:
                try:
                    import open3d as o3d  # type: ignore
                    import tempfile

                    pc = o3d.geometry.PointCloud()
                    pc.points = o3d.utility.Vector3dVector(inputs.point_cloud)
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".ply")
                    try:
                        o3d.io.write_point_cloud(tmp.name, pc)
                        tmp.seek(0)
                        pc_bytes = Path(tmp.name).read_bytes()
                    finally:
                        tmp.close()
                        Path(tmp.name).unlink(missing_ok=True)
                except Exception:
                    pc_bytes = None
            upload_artifacts(image_bytes, pc_bytes, metadata)
        else:
            st.info("Firebase credentials not detected; running in local-only mode.")

    st.caption("Run `streamlit run pc_demo/app.py` to launch this UI.")


if __name__ == "__main__":
    main()

