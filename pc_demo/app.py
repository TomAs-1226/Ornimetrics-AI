"""Streamlit PC demo for point-cloud re-id with identity debugging."""
from __future__ import annotations

import io
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
try:
    import streamlit as st
except ImportError:  # pragma: no cover - allow tests without streamlit installed
    class _Stub:
        def __getattr__(self, name):
            raise ImportError("streamlit is required to run the demo UI")

        def cache_resource(self, *args, **kwargs):
            def decorator(fn):
                return fn

            return decorator

    st = _Stub()

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.infer_stream import ensure_debug_dir, process_frame
from src.db import IdentityDB
from src.gallery import Gallery
from src.pc_preprocess import PreprocessConfig
from src.reid_embedder import PointReID
from src.tracker import Tracker
from src.yolo_detect import YOLODetector, default_yolo_model_path

LOGGER = logging.getLogger(__name__)


class HybridDetector:
    def __init__(
        self,
        base,
        fallback_bbox: Optional[List[float]],
        class_name: str,
        override_class: bool,
        allow_fallback: bool,
    ):
        self.base = base
        self.fallback_bbox = fallback_bbox
        self.class_name = class_name
        self.override_class = override_class
        self.allow_fallback = allow_fallback

    def detect(self, image) -> List[Dict[str, Any]]:
        detections: List[Dict[str, Any]] = []
        if self.base is not None:
            detections = self.base.detect(image)
            if self.override_class:
                for d in detections:
                    d["class_name"] = self.class_name
        if detections or not self.allow_fallback:
            return detections
        h, w = image.shape[:2]
        bbox = self.fallback_bbox or [0, 0, w, h]
        return [{"bbox": bbox, "class_name": self.class_name, "score": 0.99, "fallback": True}]


def _load_rgb(file) -> np.ndarray:
    import cv2  # type: ignore

    data = np.frombuffer(file.getvalue(), dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode RGB image")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _load_depth_or_cloud(file) -> Tuple[np.ndarray, str]:
    import cv2  # type: ignore

    suffix = Path(file.name).suffix.lower()
    buf = file.getvalue()
    if suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        arr = np.frombuffer(buf, dtype=np.uint8)
        depth = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise ValueError("Could not decode depth image")
        if depth.dtype == np.uint16:
            depth = depth.astype(np.float32) / 1000.0
        else:
            depth = depth.astype(np.float32)
        return depth, "depth"
    if suffix == ".npy":
        depth = np.load(io.BytesIO(buf))
        return depth.astype(np.float32), ("depth" if depth.ndim == 2 else "cloud")
    if suffix in {".ply", ".pcd"}:
        import open3d as o3d  # type: ignore

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(buf)
            tmp.flush()
            pc = o3d.io.read_point_cloud(tmp.name)
        points = np.asarray(pc.points, dtype=np.float32)
        return points, "cloud"
    raise ValueError(f"Unsupported depth/point-cloud extension: {suffix}")


def _draw_overlay(rgb: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
    import cv2  # type: ignore

    if rgb is None or rgb.size == 0:
        return rgb
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for det in detections:
        bbox = det.get("bbox", [0, 0, 0, 0])
        x1, y1, x2, y2 = [int(x) for x in bbox]
        cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cls = det.get("class_name", "")
        action = det.get("action") or det.get("decision_state")
        label_parts = [p for p in [cls, action] if p]
        label = " | ".join(label_parts) if label_parts else cls
        if label:
            cv2.putText(bgr, label, (x1, max(y1 - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


@st.cache_resource(show_spinner=False)
def _load_yolo(model_path: str):
    try:
        return YOLODetector(model_path)
    except Exception as exc:  # pragma: no cover - optional dependency
        LOGGER.warning("YOLO unavailable: %s", exc)
        return None


def _get_embedder(cfg: PreprocessConfig, backbone: str, emb_dims: int) -> PointReID:
    key = (
        cfg.plane_removal,
        cfg.plane_distance,
        cfg.voxel_size,
        cfg.fps_points,
        cfg.depth_gate_k,
        cfg.normalization,
        cfg.append_scale,
        backbone,
        emb_dims,
    )
    existing_key = st.session_state.get("embedder_key")
    if existing_key != key:
        st.session_state["embedder"] = PointReID(preprocess_config=cfg, model_name=backbone, emb_dims=emb_dims)
        st.session_state["embedder_key"] = key
    return st.session_state["embedder"]


def _get_gallery(default_threshold: float, margin_guard: float, geometry_guard: float) -> Gallery:
    cfg_key = (default_threshold, margin_guard, geometry_guard)
    gal = st.session_state.get("gallery")
    if st.session_state.get("gallery_key") != cfg_key or gal is None:
        gal = Gallery(
            default_threshold=default_threshold,
            margin_guard=margin_guard,
            geometry_guard=geometry_guard,
        )
        st.session_state["gallery"] = gal
        st.session_state["gallery_key"] = cfg_key
    return gal


def _get_tracker(smooth_window: int) -> Tracker:
    if st.session_state.get("tracker_window") != smooth_window or "tracker" not in st.session_state:
        st.session_state["tracker"] = Tracker(alpha=0.5, smooth_window=smooth_window)
        st.session_state["tracker_window"] = smooth_window
    return st.session_state["tracker"]


def _get_db() -> IdentityDB:
    if "identity_db" not in st.session_state:
        db_path = ROOT / "pc_demo_identity.sqlite"
        st.session_state["identity_db"] = IdentityDB(str(db_path))
    return st.session_state["identity_db"]


def _render_debug_blocks(blocks: List[Dict[str, Any]]):
    if not blocks:
        st.info("No detections to debug.")
        return
    for idx, block in enumerate(blocks):
        with st.expander(f"Detection debug #{idx+1}", expanded=True):
            st.json(block)


def _depth_summary(depth: np.ndarray, kind: str) -> str:
    if depth.size == 0:
        return "No depth provided"
    if kind == "cloud":
        return (
            f"point cloud points={depth.shape[0]}, depth min={float(depth[:,2].min()):.3f} m, "
            f"max={float(depth[:,2].max()):.3f} m"
        )
    return f"depth shape={depth.shape}, min={depth.min():.3f} m, max={depth.max():.3f} m, mean={depth.mean():.3f} m"


def main() -> None:
    st.title("Point-Cloud Identity Demo")
    st.caption("YOLO only for detection; identity comes from point-cloud embeddings and gallery thresholds.")

    species = st.text_input("Species label", value="sparrow")
    use_yolo = st.checkbox("Use YOLO detection if available", value=True)
    override_class = st.checkbox("Override YOLO class with species input", value=False)
    allow_fallback = st.checkbox("Use manual bbox fallback when no detections", value=False)
    bbox_inputs = st.text_input("Manual bbox x1,y1,x2,y2 (optional)", value="")
    model_path = st.text_input("YOLO model path", value=default_yolo_model_path())

    intrinsics_cols = st.columns(4)
    fx = intrinsics_cols[0].number_input("fx", value=525.0)
    fy = intrinsics_cols[1].number_input("fy", value=525.0)
    cx = intrinsics_cols[2].number_input("cx", value=319.5)
    cy = intrinsics_cols[3].number_input("cy", value=239.5)

    thr_col, margin_col, geom_col = st.columns(3)
    threshold = thr_col.number_input("Match threshold (cosine distance)", min_value=0.0, max_value=2.0, value=0.05, step=0.005)
    margin_guard = margin_col.number_input("Margin guard", min_value=0.0, max_value=1.0, value=0.02, step=0.005)
    geometry_guard = geom_col.number_input("Geometry guard (Chamfer)", min_value=0.0, max_value=1.0, value=0.02, step=0.005)

    smooth_window = st.number_input("Embedding smooth window", min_value=1, max_value=20, value=5, step=1)

    plane_removal = st.checkbox("Plane removal (RANSAC)", value=True)
    depth_gate_k = st.number_input("Depth gate k (MAD multiplier)", min_value=0.1, max_value=10.0, value=2.5, step=0.1)
    normalization = st.selectbox(
        "Normalization mode",
        options=["center_only", "center_and_scale", "center_and_scale_with_scale_feature"],
        index=0,
    )
    backbone = st.selectbox("Re-id backbone", options=["heavy", "light"], index=0, help="Heavy = wider DGCNN for harder identities")
    emb_dims = st.number_input("Embedding dims", min_value=64, max_value=1024, value=512 if backbone == "heavy" else 256, step=32)
    voxel_size = st.number_input("Voxel size (m)", min_value=0.001, max_value=0.1, value=0.01, step=0.001, format="%.3f")
    fps_points = st.number_input("Points for FPS", min_value=16, max_value=4096, value=2048, step=16)
    debug_identity = st.checkbox("Enable identity debug logging", value=True)

    rgb_file = st.file_uploader("RGB image", type=["png", "jpg", "jpeg"])
    depth_file = st.file_uploader("Depth / point cloud", type=["png", "jpg", "jpeg", "npy", "ply", "pcd"])

    cfg = PreprocessConfig(
        plane_removal=plane_removal,
        voxel_size=float(voxel_size),
        fps_points=int(fps_points),
        depth_gate_k=float(depth_gate_k),
        normalization=normalization,
        append_scale=(normalization == "center_and_scale_with_scale_feature"),
    )

    embedder = _get_embedder(cfg, backbone=backbone, emb_dims=int(emb_dims))
    gallery = _get_gallery(threshold, margin_guard, geometry_guard)
    tracker = _get_tracker(int(smooth_window))
    db = _get_db()

    debug_dir = ensure_debug_dir() if debug_identity else None

    manual_bbox: Optional[List[float]] = None
    if bbox_inputs.strip():
        try:
            vals = [float(x) for x in bbox_inputs.split(",")]
            if len(vals) == 4:
                manual_bbox = vals
        except Exception:
            st.warning("Could not parse bbox; using full frame")

    if st.button("Run re-id"):
        if rgb_file is None or depth_file is None:
            st.error("Please upload both RGB and depth/point cloud files.")
            return

        try:
            rgb = _load_rgb(rgb_file)
            depth, depth_kind = _load_depth_or_cloud(depth_file)
        except Exception as exc:
            st.error(f"Failed to parse inputs: {exc}")
            return

        model = _load_yolo(model_path) if use_yolo else None
        if model is None and use_yolo:
            st.info("YOLO unavailable; falling back to manual bbox.")

        detector = HybridDetector(model, manual_bbox, species, override_class, allow_fallback)

        intrinsics = {"fx": fx, "fy": fy, "cx": cx, "cy": cy}
        outputs, frame_debug = process_frame(
            rgb,
            depth,
            detector,
            embedder,
            gallery,
            tracker,
            db,
            intrinsics,
            threshold,
            debug_dir=debug_dir,
            smooth_window=int(smooth_window),
            debug=debug_identity,
            return_debug=True,
        )

        if not outputs:
            st.warning("No detections produced an identity result.")
        else:
            st.subheader("Detections")
            for det in outputs:
                st.json(det)
            st.subheader("RGB with detections")
            st.image(_draw_overlay(rgb, outputs), caption="Detection overlay")

        st.subheader("Identity debug")
        _render_debug_blocks(frame_debug)

        st.subheader("Inputs / stats")
        st.write(_depth_summary(depth, depth_kind))
        st.write(f"RGB shape: {rgb.shape}")

        if debug_identity and debug_dir is not None:
            st.info(f"Debug logs stored in {debug_dir}")

    st.caption("Run `streamlit run pc_demo/app.py` to launch this UI.")


if __name__ == "__main__":
    main()
