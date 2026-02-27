import torch
import torch.nn as nn
import torch.nn.functional as F

from clip.model import TextEncoder, VisualEncoder


class CLIP(nn.Module):
    def __init__(
        self,
        text_encoder: TextEncoder,
        visual_encoder: VisualEncoder,
        init_temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.visual_encoder = visual_encoder
        self.text_encoder = text_encoder
        self.logit_scale = nn.Parameter(
            torch.ones([]) * torch.log(torch.tensor((1 / init_temperature)))
        )

    def forward(
        self, image: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ):
        visual_features = self.visual_encoder(image)  # [batch_size, embed_dim]
        text_features = self.text_encoder(
            input_ids, attention_mask
        )  # [batch_size, embed_dim]

        visual_features = F.normalize(visual_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        return visual_features, text_features
