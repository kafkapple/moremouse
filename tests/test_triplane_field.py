import torch

from moremouse.models.triplane_field import TriplaneField, sample_triplanes


def test_sample_triplanes_and_field_shapes() -> None:
    """Query triplane features and NeRF field heads."""
    triplanes = torch.zeros(2, 3, 4, 8, 8)
    points = torch.zeros(2, 5, 3)

    features = sample_triplanes(triplanes, points)
    output = TriplaneField(channels=4, hidden_dim=16)(triplanes, points)

    assert features.shape == (2, 5, 12)
    assert output["density"].shape == (2, 5, 1)
    assert output["color"].shape == (2, 5, 3)
    assert output["deformation"].shape == (2, 5, 3)
