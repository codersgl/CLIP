import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from clip.data import Flickr8kDataset, get_image_transform, get_text_tokenizer


@pytest.fixture
def dummy_data_dir(tmp_path):
    """
    Create mock Flickr8k dataset：
    """
    img_dir = tmp_path / "Images"
    img_dir.mkdir()
    ann_file = tmp_path / "captions.txt"

    content = ""
    for i in range(4):
        img = Image.new("RGB", (300, 300), color=(i * 50, i * 50, i * 50))
        img_path = img_dir / f"{i}.jpg"
        img.save(img_path)
        content += f"{i}.jpg, this is a test text, just for testing\n"

    with open(ann_file, "w", encoding="utf-8") as f:
        f.write(content)

    return img_dir, ann_file


def test_dataset(dummy_data_dir):
    img_dir, ann_file = dummy_data_dir
    max_len = 77
    train = True

    transform = get_image_transform(train)
    tokenizer = get_text_tokenizer()

    dataset = Flickr8kDataset(
        img_dir=str(img_dir),
        ann_file=str(ann_file),
        transform=transform,
        tokenizer=tokenizer,
        max_len=max_len,
    )

    image, text = dataset[0]

    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 224, 224), "Image should be (3, 224, 224) after transform"

    assert isinstance(text, dict), "Text should be a dict from tokenizer"
    assert "input_ids" in text
    assert "attention_mask" in text

    input_ids = text["input_ids"]
    attention_mask = text["attention_mask"]
    assert isinstance(input_ids, torch.Tensor)
    assert isinstance(attention_mask, torch.Tensor)
    assert len(input_ids) == max_len, (
        f"input_ids length {len(input_ids)} exceeds max_len {max_len}"
    )


def test_dataset_split_train_val(dummy_data_dir):
    img_dir, ann_file = dummy_data_dir
    max_len = 77
    seed = 42

    train_transform = get_image_transform(train=True)
    valid_transform = get_image_transform(train=False)
    tokenizer = get_text_tokenizer()

    train_dataset, valid_dataset = Flickr8kDataset.split_train_val(
        img_dir=str(img_dir),
        ann_file=str(ann_file),
        train_transform=train_transform,
        val_transform=valid_transform,
        tokenizer=tokenizer,
        max_len=max_len,
        seed=seed,
    )
    for dataset in [train_dataset, valid_dataset]:
        image, text = dataset[0]

        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 224, 224), (
            "Image should be (3, 224, 224) after transform"
        )

        assert isinstance(text, dict), "Text should be a dict from tokenizer"
        assert "input_ids" in text
        assert "attention_mask" in text

        input_ids = text["input_ids"]
        attention_mask = text["attention_mask"]
        assert isinstance(input_ids, torch.Tensor)
        assert isinstance(attention_mask, torch.Tensor)
        assert len(input_ids) == max_len, (
            f"input_ids length {len(input_ids)} exceeds max_len {max_len}"
        )


def test_data_loader(dummy_data_dir):
    img_dir, ann_file = dummy_data_dir
    batch_size = 2
    shuffle = True
    num_workers = 0
    max_len = 77
    train = True

    transform = get_image_transform(train)
    tokenizer = get_text_tokenizer()

    dataset = Flickr8kDataset(
        img_dir=str(img_dir),
        ann_file=str(ann_file),
        transform=transform,
        tokenizer=tokenizer,
        max_len=max_len,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
    )

    batch = next(iter(dataloader))
    images, texts = batch

    assert isinstance(images, torch.Tensor)
    assert images.shape == (batch_size, 3, 224, 224)

    assert isinstance(texts, dict)
    assert "input_ids" in texts
    assert "attention_mask" in texts

    input_ids = texts["input_ids"]
    attention_mask = texts["attention_mask"]

    assert isinstance(input_ids, torch.Tensor)
    assert isinstance(attention_mask, torch.Tensor)

    assert input_ids.shape == (batch_size, max_len)
    assert attention_mask.shape == (batch_size, max_len)

    assert input_ids.dtype == torch.long
    assert attention_mask.dtype == torch.long
