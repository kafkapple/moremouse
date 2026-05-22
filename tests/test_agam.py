import numpy as np
import torch

from moremouse.data.agam import build_agam_template, build_target_avatar, move_torch_avatar, stack_torch_avatars, target_avatar_to_render_avatar
from moremouse.geometry.obj import ObjMesh
from moremouse.models.agam import AgamAvatarModel
from moremouse.training.agam_losses import gaussian_avatar_loss


def _mesh() -> ObjMesh:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [0, 1, 3]], dtype=np.int32)
    return ObjMesh(vertices=vertices, faces=faces)


def test_agam_template_and_model_contract() -> None:
    """Verify AGAM template construction and prediction tensor shapes."""
    mesh = _mesh()
    template = build_agam_template(mesh, anchor_count=3, geodesic_anchor_count=2)
    target = build_target_avatar(mesh, template)
    batch = stack_torch_avatars([target, target])
    device_batch = move_torch_avatar(batch, torch.device("cpu"))
    model = AgamAvatarModel(template, hidden_dim=32, use_dino=False, transformer_layers=1, transformer_heads=4)
    output = model(torch.zeros(2, 3, 64, 64))
    losses = gaussian_avatar_loss(output.avatar, device_batch)

    assert template.anchor_indices.shape == (3,)
    assert target.centers.shape == (3, 3)
    assert target_avatar_to_render_avatar(target).centers.shape == (3, 3)
    assert output.avatar.centers.shape == (2, 3, 3)
    assert output.avatar.rotations.shape == (2, 3, 4)
    assert torch.isfinite(losses["total"]).item()
