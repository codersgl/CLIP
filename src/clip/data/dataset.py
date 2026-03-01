import os
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import Compose
from transformers import AutoTokenizer, PreTrainedTokenizerBase


def get_image_transform(train: bool = True):
    if train:
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )

        return train_transform
    else:
        val_transform = transforms.Compose(
            [
                transforms.Resize(
                    224, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
        return val_transform


def get_text_tokenizer(model_name: str = "distilbert-base-uncased"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


class Flickr8kDataset(Dataset):
    def __init__(
        self,
        img_dir: str,
        ann_file: str,
        transform: Optional[Compose] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        max_len: int = 77,
        image_ids=None,
    ):
        """
        img_dir: The path of image dictionary.
        ann_file: The path of annotation file (Flickr8k.token.txt)
        max_len: The maximum length of the text.
        """

        if tokenizer is None:
            raise ValueError("A tokenizer must be provided for the dataset")
        self.img_dir = img_dir
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_len = max_len

        with open(ann_file, "r") as f:
            lines = f.readlines()

        # Data structure：{ Image name: [ann1, ann2, ...] }
        self.caption_dict: dict = {}
        for line in lines[1:]:
            image_name, caption = line.strip().split(",", maxsplit=1)

            if image_name not in self.caption_dict:
                self.caption_dict[image_name] = []
            self.caption_dict[image_name].append(caption)

        # Save all image name
        if image_ids is None:
            self.image_ids = list(self.caption_dict.keys())
        else:
            self.image_ids = image_ids

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_name = self.image_ids[idx]
        captions = self.caption_dict[img_name]

        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        caption = captions[torch.randint(0, len(captions), (1,)).item()]
        tokenized = self.tokenizer(
            caption,
            max_length=self.max_len,
            padding="max_length",  # padding to max_len
            truncation=True,
            return_tensors="pt",  # Return PyTorch Tensor
        )
        tokenized = {k: v.squeeze(0) for k, v in tokenized.items()}

        return image, tokenized

    @classmethod
    def split_train_val(
        cls,
        img_dir,
        ann_file,
        tokenizer,
        train_transform,
        val_transform,
        train_ratio=0.8,
        seed=42,
        max_len=77,
        **kwargs,
    ):
        with open(ann_file, "r") as f:
            lines = f.readlines()
        caption_dict = {}
        for line in lines[1:]:
            image_name, caption = line.strip().split(",", maxsplit=1)
            if image_name not in caption_dict:
                caption_dict[image_name] = []
            caption_dict[image_name].append(caption)
        all_ids = list(caption_dict.keys())

        torch.manual_seed(seed)
        indices = list(range(len(all_ids)))
        shuffled_indices = torch.randperm(len(indices)).tolist()
        train_size = int(train_ratio * len(indices))
        train_indices = shuffled_indices[:train_size]
        val_indices = shuffled_indices[train_size:]

        train_ids = [all_ids[i] for i in train_indices]
        val_ids = [all_ids[i] for i in val_indices]

        train_dataset = cls(
            img_dir=img_dir,
            ann_file=ann_file,
            transform=train_transform,
            tokenizer=tokenizer,
            max_len=max_len,
            image_ids=train_ids,
        )
        val_dataset = cls(
            img_dir=img_dir,
            ann_file=ann_file,
            transform=val_transform,
            tokenizer=tokenizer,
            max_len=max_len,
            image_ids=val_ids,
        )
        return train_dataset, val_dataset
