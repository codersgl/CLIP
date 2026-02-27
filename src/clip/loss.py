import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.logit_scale = nn.Parameter(
            torch.ones([]) * torch.log(torch.tensor(1 / temperature))
        )

    def forward(self, image_features, text_features):
        """
        image_features: (batch, embed_dim) have normalized.
        text_features:  (batch, embed_dim) have normalized.
        """
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logits_per_image.T

        batch_size = logits_per_image.shape[0]
        labels = torch.arange(batch_size, device=logits_per_image.device)

        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)
        return (loss_i + loss_t) / 2
