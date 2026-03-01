import pytest
import torch

from clip.data import get_text_tokenizer
from clip.model import CLIP, TextEncoder, VisualEncoder


@pytest.fixture
def dummy_data():
    tokenizer = get_text_tokenizer()
    batch_size = 4
    sample_texts = ["a cat on the mat", "a dog runs fast"] * (batch_size // 2 + 1)
    sample_texts = sample_texts[:batch_size]
    tokenized = tokenizer(
        sample_texts,
        max_length=77,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    image_input = torch.randn(batch_size, 3, 224, 224)
    return image_input, tokenized


def test_text_encoder(dummy_data):
    embed_dim = 512
    _, text_dict = dummy_data
    input_ids = text_dict["input_ids"]
    attention_mask = text_dict["attention_mask"]

    model = TextEncoder(embed_dim=embed_dim)
    text_features = model(input_ids, attention_mask)
    assert text_features.shape == (input_ids.size(0), embed_dim)


def test_visual_encoder(dummy_data):
    embed_dim = 512
    image, _ = dummy_data
    batch_size = image.size(0)

    model = VisualEncoder(embed_dim=embed_dim)
    visual_features = model(image)
    assert visual_features.shape == (batch_size, embed_dim)


def test_clip(dummy_data):
    embed_dim = 512
    image, text_dict = dummy_data
    input_ids = text_dict["input_ids"]
    attention_mask = text_dict["attention_mask"]
    batch_size = image.size(0)

    text_encoder = TextEncoder(embed_dim=embed_dim)
    visual_encoder = VisualEncoder(embed_dim=embed_dim)
    model = CLIP(text_encoder, visual_encoder)

    visual_features_norm, text_features_norm = model(image, input_ids, attention_mask)

    assert visual_features_norm.shape == (batch_size, embed_dim)
    assert text_features_norm.shape == (batch_size, embed_dim)
    visual_norms = torch.norm(visual_features_norm, dim=1)
    text_norms = torch.norm(text_features_norm, dim=1)

    assert torch.allclose(
        visual_norms, torch.ones(batch_size, device=visual_norms.device), atol=1e-6
    ), f"Features not normalized, norms: {visual_norms}"

    assert torch.allclose(
        text_norms, torch.ones(batch_size, device=text_norms.device), atol=1e-6
    ), f"Features not normalized, norms: {text_norms}"
