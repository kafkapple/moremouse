import torch

from moremouse.models.dino_encoder import DinoImageEncoder


def test_dino_encoder_offline_fallback_shape() -> None:
    """Use the offline fallback while preserving DINO-token contract."""
    encoder = DinoImageEncoder(output_dim=16, pretrained=False)
    tokens = encoder(torch.zeros(2, 3, 64, 64))

    assert tokens.shape == (2, 16, 16)
