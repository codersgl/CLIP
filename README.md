# Plan

## 🧩 第一步：组装 CLIP 模型

创建一个 `CLIP` 类，它包含两个编码器和一个可学习的温度参数 `logit_scale`（论文中初始化为 `ln(1/0.07)`）。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CLIP(nn.Module):
    def __init__(self, image_encoder: nn.Module, text_encoder: nn.Module, init_temperature: float = 0.07):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        # 可学习的温度参数（论文中是对数尺度）
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / init_temperature)))

    def forward(self, images, input_ids, attention_mask):
        # 获取图像和文本特征
        image_features = self.image_encoder(images)          # (batch, embed_dim)
        text_features = self.text_encoder(input_ids, attention_mask)  # (batch, embed_dim)

        # 特征归一化（重要！）
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        # 计算相似度矩阵并乘以温度
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logits_per_image.T

        return logits_per_image, logits_per_text
```

---

## 🔁 第二步：定义对比损失

使用对称的交叉熵损失，这是 CLIP 训练的核心。

```python
def clip_loss(logits_per_image, logits_per_text):
    # 创建标签：对角线为正样本
    batch_size = logits_per_image.shape[0]
    labels = torch.arange(batch_size, device=logits_per_image.device)

    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    return (loss_i + loss_t) / 2
```

---

## 📦 第三步：准备数据

你需要一个图文对数据集（如 Flickr30k、MSCOCO Captions 或 Conceptual Captions）。数据加载包含两个关键部分：

### 3.1 图像预处理

使用 `torchvision.transforms`，可参考 CLIP 论文的设置：

```python
from torchvision import transforms

image_transform = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                         std=[0.26862954, 0.26130258, 0.27577711])
])
```

### 3.2 文本分词

你需要一个分词器（tokenizer）。可以使用 HuggingFace 的 `AutoTokenizer`，或者自己实现一个简单的 BPE（但后者较复杂，建议先用现成的）。

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
# 设置最大长度，CLIP 原文为 77
max_length = 77

def tokenize(texts):
    return tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
```

### 3.3 自定义 Dataset

```python
from torch.utils.data import Dataset
from PIL import Image
import os

class ImageTextDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None, tokenizer=None):
        self.data = pd.read_csv(csv_file)  # 假设有 'image_name', 'caption' 列
        self.image_dir = image_dir
        self.transform = transform
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = os.path.join(self.image_dir, row['image_name'])
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        text = row['caption']
        encoding = self.tokenizer(text, padding='max_length', truncation=True, max_length=77, return_tensors='pt')
        return image, encoding['input_ids'].squeeze(0), encoding['attention_mask'].squeeze(0)
```

### 3.4 DataLoader

```python
from torch.utils.data import DataLoader

dataset = ImageTextDataset(csv_file='train.csv', image_dir='images/',
                           transform=image_transform, tokenizer=tokenize)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
```

---

## ⚙️ 第四步：训练循环

### 4.1 基本设置

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CLIP(image_encoder, text_encoder).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
scaler = torch.cuda.amp.GradScaler()  # 混合精度加速
```

### 4.2 训练循环

```python
for epoch in range(num_epochs):
    for batch in dataloader:
        images, input_ids, attention_mask = [x.to(device) for x in batch]

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():  # 混合精度
            logits_per_image, logits_per_text = model(images, input_ids, attention_mask)
            loss = clip_loss(logits_per_image, logits_per_text)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    scheduler.step()
    print(f"Epoch {epoch}: loss = {loss.item():.4f}")
```

---

## 📊 第五步：验证与评估

训练过程中，你可以定期评估模型的零样本性能。

### 零样本图像分类（以 CIFAR-100 为例）

```python
def zero_shot_eval(model, dataloader, class_names, template="a photo of a {}"):
    # 将类别名称转换为文本提示
    texts = [template.format(c) for c in class_names]
    tokenized = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)

    model.eval()
    with torch.no_grad():
        # 预计算文本特征
        text_features = model.text_encoder(tokenized['input_ids'], tokenized['attention_mask'])
        text_features = F.normalize(text_features, dim=-1)

        total, correct = 0, 0
        for images, labels in dataloader:
            images = images.to(device)
            image_features = model.image_encoder(images)
            image_features = F.normalize(image_features, dim=-1)

            # 计算相似度
            similarity = image_features @ text_features.T
            preds = similarity.argmax(dim=-1)
            correct += (preds == labels.to(device)).sum().item()
            total += labels.size(0)

    return correct / total
```

---

## 💡 额外建议

1. **大批量训练**：CLIP 依赖大量负样本，如果你的 GPU 显存有限，可以尝试**梯度累积**。
2. **使用预训练骨干**：你的图像和文本编码器可以加载预训练权重（如 `timm` 模型和 `transformers` 模型），这能大幅加速收敛。
3. **日志与可视化**：使用 TensorBoard 或 WandB 记录损失、温度参数、学习率等。
4. **保存模型**：定期保存 checkpoint，方便恢复训练或后续评估。

---

## 🚀 下一步：扩展与优化

- 尝试不同的池化策略（`cls` vs `mean`）或添加额外的投影层（如 LayerNorm）。
- 在更大数据集（如 CC3M、YFCC100M）上训练。
- 实现更多评估指标（图文检索 Recall@K）。
- 探索更先进的训练技巧，如 MoCo 或数据增强。
