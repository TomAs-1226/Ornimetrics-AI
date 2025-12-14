"""Collect depth+mask crops from recorded tracklets for self-supervised training."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np


def save_tracklet_crops(tracklets: Iterable[dict], out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for idx, trk in enumerate(tracklets):
        depth = np.asarray(trk["depth"], dtype=np.float32)
        mask = np.asarray(trk["mask"], dtype=bool)
        species = trk.get("species", "unknown")
        track_id = trk.get("track_id", idx)
        np.save(out / f"depth_{idx}.npy", depth)
        np.save(out / f"mask_{idx}.npy", mask)
        manifest.append({"depth": f"depth_{idx}.npy", "mask": f"mask_{idx}.npy", "species": species, "track_id": track_id})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":  # pragma: no cover - utility script
    import argparse

    parser = argparse.ArgumentParser(description="Collect BirdID training crops from serialized tracklets")
    parser.add_argument("tracklets_json", help="Path to JSON with list of tracklets containing depth/mask arrays")
    parser.add_argument("output_dir", help="Directory to save crops and manifest")
    args = parser.parse_args()

    data = json.loads(Path(args.tracklets_json).read_text())
    save_tracklet_crops(data, args.output_dir)
