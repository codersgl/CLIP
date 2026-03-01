import os

import hydra
import torch
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from clip import CLIPLoss, Trainer, set_seed
from clip.data import Flickr8kDataset, get_image_transform, get_text_tokenizer
from clip.model import CLIP, TextEncoder, VisualEncoder


@hydra.main(version_base=None, config_path="../configs/", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    original_cwd = hydra.utils.get_original_cwd()

    img_dir = os.path.join(original_cwd, cfg.data.img_dir)
    ann_file = os.path.join(original_cwd, cfg.data.ann_file)

    set_seed(cfg.training.seed)

    device = torch.device(
        cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    )

    text_encoder = TextEncoder(embed_dim=cfg.model.embed_dim, **cfg.model.text_encoder)
    visual_encoder = VisualEncoder(
        embed_dim=cfg.model.embed_dim, **cfg.model.visual_encoder
    )

    model = CLIP(text_encoder, visual_encoder, cfg.model.init_temperature)
    model.to(device)

    train_dataset, valid_dataset = Flickr8kDataset.split_train_val(
        img_dir=img_dir,
        ann_file=ann_file,
        train_transform=get_image_transform(train=True),
        val_transform=get_image_transform(train=False),
        tokenizer=get_text_tokenizer(cfg.data.tokenizer_model_name),
        seed=cfg.training.seed,
        train_ratio=cfg.data.train_ratio,
        max_len=cfg.data.max_len,
    )

    train_dataloader = DataLoader(
        train_dataset,
        cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )

    if cfg.optimizer.name == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg.optimizer.lr,
            betas=cfg.optimizer.betas,
            weight_decay=cfg.optimizer.weight_decay,
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg.optimizer.lr,
            betas=cfg.optimizer.betas,
            weight_decay=cfg.optimizer.weight_decay,
        )

    loss_func = CLIPLoss(cfg.model.init_temperature)

    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=valid_dataloader,
        loss_func=loss_func,
        optimizer=optimizer,
        device=device,
        epochs=cfg.training.epochs,
        use_early_stop=cfg.training.use_early_stop,
        patience=cfg.training.patience,
        save_path=cfg.training.save_path,
        grad_clip_max_norm=cfg.training.grad_clip_max_norm,
        log_dir=cfg.training.log_dir,
    )

    trainer.train()


if __name__ == "__main__":
    main()
