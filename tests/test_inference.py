from pathlib import Path

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from PIL import Image

from clip.inference import InferenceEngine


class DummyTokenizer:
    def __call__(
        self,
        texts,
        max_length,
        padding,
        truncation,
        return_tensors,
    ):
        lengths = [max(1, len(text.split())) for text in texts]
        input_ids = torch.zeros((len(texts), max_length), dtype=torch.long)
        attention_mask = torch.zeros((len(texts), max_length), dtype=torch.long)
        for idx, length in enumerate(lengths):
            valid_len = min(length, max_length)
            input_ids[idx, :valid_len] = 1
            attention_mask[idx, :valid_len] = 1

        return {"input_ids": input_ids, "attention_mask": attention_mask}


class DummyTextEncoder(nn.Module):
    def forward(self, input_ids, attention_mask):
        lengths = attention_mask.sum(dim=1, keepdim=True).float()
        return torch.cat([lengths, 1.0 / (lengths + 1.0)], dim=1)


class DummyVisualEncoder(nn.Module):
    def forward(self, images):
        intensity = images[:, 0, 0, 0].unsqueeze(1)
        return torch.cat([intensity, 1.0 / (intensity + 1.0)], dim=1)


class DummyCLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_encoder = DummyTextEncoder()
        self.visual_encoder = DummyVisualEncoder()


def dummy_transform(image):
    pixel_value = image.getpixel((0, 0))[0] / 255.0
    return torch.full((3, 1, 1), fill_value=pixel_value, dtype=torch.float32)


def create_dummy_image(path: Path, value: int):
    image = Image.new("RGB", (2, 2), color=(value, value, value))
    image.save(path)


@pytest.fixture
def inference_engine(tmp_path):
    image_paths = [
        tmp_path / "img_0.jpg",
        tmp_path / "img_1.jpg",
        tmp_path / "img_2.jpg",
    ]
    for idx, image_path in enumerate(image_paths):
        create_dummy_image(image_path, (idx + 1) * 40)

    engine = InferenceEngine(
        model=DummyCLIP(),
        tokenizer=DummyTokenizer(),
        image_transform=dummy_transform,
        device=torch.device("cpu"),
        max_len=8,
    )
    return engine, [str(path) for path in image_paths]


def test_retrieve_text_to_image_topk(inference_engine):
    engine, image_paths = inference_engine
    texts = ["a", "a b c d e f"]

    results = engine.retrieve_text_to_image(
        texts=texts, image_paths=image_paths, top_k=2
    )

    assert len(results) == len(texts)
    for result in results:
        assert len(result["matches"]) == 2
        assert result["matches"][0]["score"] >= result["matches"][1]["score"]


def test_retrieve_image_to_text_topk(inference_engine):
    engine, image_paths = inference_engine
    texts = ["a", "a b", "a b c d e"]

    results = engine.retrieve_image_to_text(
        image_paths=image_paths, texts=texts, top_k=2
    )

    assert len(results) == len(image_paths)
    for result in results:
        assert len(result["matches"]) == 2
        assert result["matches"][0]["score"] >= result["matches"][1]["score"]


def test_topk_is_capped(inference_engine):
    engine, image_paths = inference_engine
    texts = ["short"]

    results = engine.retrieve_text_to_image(
        texts=texts, image_paths=image_paths, top_k=99
    )

    assert len(results[0]["matches"]) == len(image_paths)


def test_from_config_tokenizer_model_mismatch_raises():
    cfg = OmegaConf.create(
        {
            "model": {
                "embed_dim": 8,
                "init_temperature": 0.07,
                "text_encoder": {"model_name": "text-model", "pooler": "cls"},
                "visual_encoder": {"backbone_name": "resnet18", "pretrained": False},
            },
            "data": {
                "tokenizer_model_name": "tokenizer-model",
                "max_len": 8,
            },
        }
    )

    with pytest.raises(ValueError):
        InferenceEngine.from_config(cfg=cfg, checkpoint_path="dummy.pt")


def test_retrieve_text_to_image_include_errors_with_bad_inputs(inference_engine):
    engine, image_paths = inference_engine
    texts = ["valid text", "   ", "another text"]
    bad_image_path = str(Path(image_paths[0]).parent / "missing.jpg")
    mixed_image_paths = [image_paths[0], bad_image_path, image_paths[1]]

    payload = engine.retrieve_text_to_image(
        texts=texts,
        image_paths=mixed_image_paths,
        top_k=2,
        include_errors=True,
    )

    assert "results" in payload
    assert "errors" in payload
    assert "stats" in payload

    assert payload["stats"]["input_text_count"] == 3
    assert payload["stats"]["valid_text_count"] == 2
    assert payload["stats"]["input_image_count"] == 3
    assert payload["stats"]["valid_image_count"] == 2
    assert len(payload["errors"]) == 2

    error_types = {item["type"] for item in payload["errors"]}
    assert error_types == {"text", "image"}


def test_retrieve_image_to_text_include_errors_with_bad_inputs(inference_engine):
    engine, image_paths = inference_engine
    bad_image_path = str(Path(image_paths[0]).parent / "missing_2.jpg")
    mixed_image_paths = [bad_image_path, image_paths[0], image_paths[1]]
    texts = ["ok text", ""]

    payload = engine.retrieve_image_to_text(
        image_paths=mixed_image_paths,
        texts=texts,
        top_k=2,
        include_errors=True,
    )

    assert payload["stats"]["input_text_count"] == 2
    assert payload["stats"]["valid_text_count"] == 1
    assert payload["stats"]["input_image_count"] == 3
    assert payload["stats"]["valid_image_count"] == 2
    assert len(payload["errors"]) == 2


def test_encode_images_raises_when_any_image_invalid(inference_engine):
    engine, image_paths = inference_engine
    bad_image_path = str(Path(image_paths[0]).parent / "missing_3.jpg")

    with pytest.raises(ValueError):
        engine.encode_images([image_paths[0], bad_image_path])
