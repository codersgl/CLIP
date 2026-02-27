import os

import torch
from PIL import Image
from torch.utils.data import Dataset


class Flickr8kDataset(Dataset):
    def __init__(self, img_dir, ann_file, transform=None, tokenizer=None, max_len=77):
        """
        img_dir: The path of image dictionary.
        ann_file: The path of annotation file (Flickr8k.token.txt)
        max_len: The maximum length of the text.
        """
        self.img_dir = img_dir
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_len = max_len

        with open(ann_file, "r") as f:
            lines = f.readlines()

        # Data structure：{ Image name: [ann1, ann2, ...] }
        self.caption_dict = {}
        for line in lines[1:]:
            image_name, caption = line.strip().split(",")

            if image_name not in self.caption_dict:
                self.caption_dict[image_name] = []
            self.caption_dict[image_name].append(caption)

        # Save all image name
        self.image_ids = list(self.caption_dict.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_name = self.image_ids[idx]
        captions = self.caption_dict[img_name]

        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Tokenize (usually choose a annotaion randomly during the trainning.)
        if self.tokenizer:
            # TODO: need to implement.
            caption = captions[torch.randint(0, len(captions), (1,)).item()]
            tokenized = self.tokenizer(
                caption,
                max_length=self.max_len,
            )
            return image, tokenized
        else:
            return image, captions
