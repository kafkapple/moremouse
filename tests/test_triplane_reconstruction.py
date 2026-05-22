import torch

from moremouse.models.triplane_reconstruction import MoReMouseTriplane


def test_moremouse_triplane_shapes() -> None:
    """Verify transformer-triplane model output contracts."""
    model = MoReMouseTriplane(components=5, hidden_dim=32, plane_channels=4, plane_size=8, layers=1, heads=4)
    output = model(torch.zeros(2, 3, 64, 64))

    assert output["coeffs"].shape == (2, 5)
    assert output["triplanes"].shape == (2, 3, 4, 8, 8)
    assert torch.isfinite(output["coeffs"]).all()
