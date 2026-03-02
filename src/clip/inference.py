from __future__ import annotations

import os
from typing import Any, cast

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from PIL import Image
from torch import nn
from tqdm import tqdm

from clip.data import get_image_transform, get_text_tokenizer
from clip.model import CLIP, TextEncoder, VisualEncoder


class InferenceEngine:
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        image_transform,
        device: torch.device,
        max_len: int,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.device = device
        self.max_len = max_len

    @staticmethod
    def _iter_with_progress(items, desc: str, show_progress: bool):
        if show_progress:
            return tqdm(items, desc=desc, leave=False)
        return items

    def _encode_image_batch(self, image_tensor: torch.Tensor) -> torch.Tensor:
        visual_encoder = cast(Any, self.model).visual_encoder
        return visual_encoder(image_tensor)

    def _encode_text_batch(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        text_encoder = cast(Any, self.model).text_encoder
        return text_encoder(input_ids, attention_mask)

    @classmethod
    def from_config(
        cls,
        cfg: DictConfig,
        checkpoint_path: str,
        device: str | None = None,
    ) -> "InferenceEngine":
        tokenizer_model_name = cfg.data.tokenizer_model_name
        text_model_name = cfg.model.text_encoder.model_name

        if tokenizer_model_name != text_model_name:
            raise ValueError(
                "Tokenizer model and text encoder model must be consistent: "
                f"cfg.data.tokenizer_model_name={tokenizer_model_name}, "
                f"cfg.model.text_encoder.model_name={text_model_name}"
            )

        target_device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

        text_encoder = TextEncoder(
            embed_dim=cfg.model.embed_dim, **cfg.model.text_encoder
        )
        visual_encoder = VisualEncoder(
            embed_dim=cfg.model.embed_dim, **cfg.model.visual_encoder
        )
        model = CLIP(text_encoder, visual_encoder, cfg.model.init_temperature)

        checkpoint = torch.load(checkpoint_path, map_location=target_device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        model.load_state_dict(checkpoint)

        model.to(target_device)
        model.eval()

        tokenizer = get_text_tokenizer(tokenizer_model_name)
        image_transform = get_image_transform(train=False)

        return cls(
            model=model,
            tokenizer=tokenizer,
            image_transform=image_transform,
            device=target_device,
            max_len=cfg.data.max_len,
        )

    @torch.no_grad()
    def encode_images(
        self,
        image_paths: list[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> torch.Tensor:
        embeddings, _, _, errors = self.encode_images_with_paths(
            image_paths=image_paths,
            batch_size=batch_size,
            show_progress=show_progress,
        )
        if errors:
            raise ValueError(
                "Failed to encode all images. "
                "Use encode_images_with_paths/retrieval include_errors for fault-tolerant behavior."
            )
        return embeddings

    @torch.no_grad()
    def encode_images_with_paths(
        self,
        image_paths: list[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> tuple[torch.Tensor, list[str], list[int], list[dict[str, Any]]]:
        if len(image_paths) == 0:
            raise ValueError("image_paths cannot be empty")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        embeddings = []
        kept_paths: list[str] = []
        kept_indices: list[int] = []
        errors: list[dict[str, Any]] = []

        start_points = list(range(0, len(image_paths), batch_size))
        batch_iterator = self._iter_with_progress(
            start_points,
            desc="Encoding images",
            show_progress=show_progress,
        )
        for start in batch_iterator:
            batch_paths = image_paths[start : start + batch_size]
            images = []
            valid_batch_indices: list[int] = []
            valid_batch_paths: list[str] = []
            for local_idx, image_path in enumerate(batch_paths):
                global_idx = start + local_idx
                try:
                    with Image.open(image_path) as image:
                        image = image.convert("RGB")
                        images.append(self.image_transform(image))
                    valid_batch_indices.append(global_idx)
                    valid_batch_paths.append(image_path)
                except Exception as err:
                    errors.append(
                        {
                            "type": "image",
                            "index": global_idx,
                            "input": image_path,
                            "message": str(err),
                        }
                    )

            if len(images) == 0:
                continue

            image_tensor = torch.stack(images, dim=0).to(self.device)
            image_features = self._encode_image_batch(image_tensor)
            image_features = F.normalize(image_features, dim=-1)
            embeddings.append(image_features.cpu())

            kept_indices.extend(valid_batch_indices)
            kept_paths.extend(valid_batch_paths)

        if not embeddings:
            raise ValueError(
                f"No valid images were encoded. Total image errors: {len(errors)}"
            )

        return torch.cat(embeddings, dim=0), kept_paths, kept_indices, errors

    @torch.no_grad()
    def encode_texts(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> torch.Tensor:
        embeddings, _, _, errors = self.encode_texts_with_items(
            texts=texts,
            batch_size=batch_size,
            show_progress=show_progress,
        )
        if errors:
            raise ValueError(
                "Failed to encode all texts. "
                "Use encode_texts_with_items/retrieval include_errors for fault-tolerant behavior."
            )
        return embeddings

    @torch.no_grad()
    def encode_texts_with_items(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> tuple[torch.Tensor, list[str], list[int], list[dict[str, Any]]]:
        if len(texts) == 0:
            raise ValueError("texts cannot be empty")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        embeddings = []
        kept_texts: list[str] = []
        kept_indices: list[int] = []
        errors: list[dict[str, Any]] = []

        start_points = list(range(0, len(texts), batch_size))
        batch_iterator = self._iter_with_progress(
            start_points,
            desc="Encoding texts",
            show_progress=show_progress,
        )

        for start in batch_iterator:
            batch_texts = texts[start : start + batch_size]
            valid_batch_texts: list[str] = []
            valid_batch_indices: list[int] = []
            for local_idx, text in enumerate(batch_texts):
                global_idx = start + local_idx
                if not isinstance(text, str) or len(text.strip()) == 0:
                    errors.append(
                        {
                            "type": "text",
                            "index": global_idx,
                            "input": text,
                            "message": "Empty or non-string text query.",
                        }
                    )
                    continue
                valid_batch_texts.append(text)
                valid_batch_indices.append(global_idx)

            if len(valid_batch_texts) == 0:
                continue

            tokenized = self.tokenizer(
                valid_batch_texts,
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            input_ids = tokenized["input_ids"].to(self.device)
            attention_mask = tokenized["attention_mask"].to(self.device)

            text_features = self._encode_text_batch(input_ids, attention_mask)
            text_features = F.normalize(text_features, dim=-1)
            embeddings.append(text_features.cpu())

            kept_texts.extend(valid_batch_texts)
            kept_indices.extend(valid_batch_indices)

        if not embeddings:
            raise ValueError(
                f"No valid texts were encoded. Total text errors: {len(errors)}"
            )

        return torch.cat(embeddings, dim=0), kept_texts, kept_indices, errors

    def retrieve_text_to_image(
        self,
        texts: list[str],
        image_paths: list[str],
        top_k: int = 5,
        batch_size: int = 32,
        include_errors: bool = False,
        show_progress: bool = False,
    ) -> list[dict] | dict[str, Any]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        text_features, kept_texts, text_indices, text_errors = (
            self.encode_texts_with_items(
                texts=texts,
                batch_size=batch_size,
                show_progress=show_progress,
            )
        )
        (
            image_features,
            kept_image_paths,
            image_indices,
            image_errors,
        ) = self.encode_images_with_paths(
            image_paths=image_paths,
            batch_size=batch_size,
            show_progress=show_progress,
        )

        similarity = text_features @ image_features.T
        top_k = min(top_k, len(kept_image_paths))
        scores, indices = similarity.topk(k=top_k, dim=1)

        results = []
        for text_idx, text in enumerate(kept_texts):
            matches = []
            for rank in range(top_k):
                image_idx = int(indices[text_idx, rank].item())
                matches.append(
                    {
                        "image_path": kept_image_paths[image_idx],
                        "image_index": image_indices[image_idx],
                        "score": float(scores[text_idx, rank].item()),
                    }
                )
            results.append(
                {
                    "query_text": text,
                    "query_text_index": text_indices[text_idx],
                    "matches": matches,
                }
            )

        if include_errors:
            return {
                "results": results,
                "errors": [*text_errors, *image_errors],
                "stats": {
                    "input_text_count": len(texts),
                    "valid_text_count": len(kept_texts),
                    "input_image_count": len(image_paths),
                    "valid_image_count": len(kept_image_paths),
                },
            }

        return results

    def retrieve_image_to_text(
        self,
        image_paths: list[str],
        texts: list[str],
        top_k: int = 5,
        batch_size: int = 32,
        include_errors: bool = False,
        show_progress: bool = False,
    ) -> list[dict] | dict[str, Any]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        text_features, kept_texts, text_indices, text_errors = (
            self.encode_texts_with_items(
                texts=texts,
                batch_size=batch_size,
                show_progress=show_progress,
            )
        )
        (
            image_features,
            kept_image_paths,
            image_indices,
            image_errors,
        ) = self.encode_images_with_paths(
            image_paths=image_paths,
            batch_size=batch_size,
            show_progress=show_progress,
        )

        similarity = image_features @ text_features.T
        top_k = min(top_k, len(kept_texts))
        scores, indices = similarity.topk(k=top_k, dim=1)

        results = []
        for image_idx, image_path in enumerate(kept_image_paths):
            matches = []
            for rank in range(top_k):
                text_idx = int(indices[image_idx, rank].item())
                matches.append(
                    {
                        "text": kept_texts[text_idx],
                        "text_index": text_indices[text_idx],
                        "score": float(scores[image_idx, rank].item()),
                    }
                )
            results.append(
                {
                    "query_image": image_path,
                    "query_image_index": image_indices[image_idx],
                    "matches": matches,
                }
            )

        if include_errors:
            return {
                "results": results,
                "errors": [*text_errors, *image_errors],
                "stats": {
                    "input_text_count": len(texts),
                    "valid_text_count": len(kept_texts),
                    "input_image_count": len(image_paths),
                    "valid_image_count": len(kept_image_paths),
                },
            }

        return results
