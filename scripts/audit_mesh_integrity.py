"""Audit MAMMAL mesh and fitting parameter integrity."""

from pathlib import Path
import hashlib
import json
import pickle

import numpy as np
from loguru import logger
from omegaconf import OmegaConf

from moremouse.geometry.obj import load_obj_mesh


def main() -> None:
    """Validate manifest assets, mesh topology, and fitting parameter files."""
    cfg = OmegaConf.load("configs/datasets/markerless_mammal.yaml").dataset
    output_path = Path(cfg.outputs.camera_projection_audit_dir) / "mesh_integrity.json"
    manifest = json.loads(Path(cfg.manifest).read_text(encoding="utf-8"))
    assets = manifest["assets"]
    if len(assets) != int(cfg.fitting.primary.frame_count):
        raise ValueError(f"Unexpected asset count: {len(assets)}")

    sampled_frames = [0, 2000, 6000, 12000, 17980]
    by_frame = {int(asset["frame_id"]): asset for asset in assets}
    full_stats = {
        "asset_count": len(assets),
        "primary_count": sum("primary" in asset["source"] for asset in assets),
        "override_count": sum("override" in asset["source"] for asset in assets),
        "frame_min": min(by_frame),
        "frame_max": max(by_frame),
        "frame_step_values": sorted(set(np.diff(sorted(by_frame)).tolist())),
    }
    samples = []
    topology_hashes = set()
    for frame_id in sampled_frames:
        asset = by_frame[frame_id]
        mesh = load_obj_mesh(Path(asset["obj_path"]))
        params = load_params(Path(asset["param_path"]))
        topology_hash = hash_faces(mesh.faces)
        topology_hashes.add(topology_hash)
        edge_lengths = mesh_edge_lengths(mesh.vertices, mesh.faces)
        bounds_min, bounds_max = mesh.bounds
        samples.append(
            {
                "frame_id": frame_id,
                "source": asset["source"],
                "vertices": int(mesh.vertices.shape[0]),
                "faces": int(mesh.faces.shape[0]),
                "topology_hash": topology_hash,
                "bounds_min": bounds_min.tolist(),
                "bounds_max": bounds_max.tolist(),
                "edge_length_mean": float(edge_lengths.mean()),
                "edge_length_p95": float(np.percentile(edge_lengths, 95)),
                "params": params,
            }
        )
    report = {
        "manifest": full_stats,
        "sampled_frames": samples,
        "same_topology_for_sampled_frames": len(topology_hashes) == 1,
        "mesh_usage_note": (
            "Meshes are currently used only for manifest, preview, projection, and integrity audits; "
            "they are not yet used in MoReMouse training."
        ),
    }
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Wrote mesh integrity report to {}", output_path)


def load_params(path: Path) -> dict[str, object]:
    """Load and summarize a MAMMAL fitting parameter pickle."""
    if not path.exists():
        raise FileNotFoundError(f"Parameter file missing: {path}")
    with path.open("rb") as file:
        payload = pickle.load(file)
    required = ["thetas", "trans", "scale", "rotation", "bone_lengths", "chest_deformer"]
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Missing parameter keys: {missing}")
    summary = {}
    for key in required:
        value = payload[key]
        array = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
        if not np.isfinite(array).all():
            raise ValueError(f"Non-finite parameter values in {path}: {key}")
        summary[key] = {
            "shape": list(array.shape),
            "min": float(array.min()),
            "max": float(array.max()),
        }
    return summary


def hash_faces(faces: np.ndarray) -> str:
    """Hash mesh face topology."""
    digest = hashlib.sha256(np.asarray(faces, dtype=np.int32).tobytes()).hexdigest()
    return digest[:16]


def mesh_edge_lengths(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute triangle edge lengths for mesh sanity statistics."""
    tri = vertices[faces[:, :3]]
    edges = np.concatenate(
        [
            np.linalg.norm(tri[:, 0] - tri[:, 1], axis=1),
            np.linalg.norm(tri[:, 1] - tri[:, 2], axis=1),
            np.linalg.norm(tri[:, 2] - tri[:, 0], axis=1),
        ]
    )
    if not np.isfinite(edges).all():
        raise ValueError("Non-finite mesh edge lengths")
    return edges


if __name__ == "__main__":
    main()
