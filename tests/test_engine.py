from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from clip.engine import EarlyStopping, Trainer
from clip.loss import CLIPLoss

# --- Mock Classes & Fixtures ---


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, image, input_ids, attention_mask):
        # Return two tensors: image_features, text_features
        # Assuming embed_dim=10 based on linear layer
        # batch_size is taken from image input
        batch_size = image.shape[0]
        return torch.randn(batch_size, 10), torch.randn(batch_size, 10)


@pytest.fixture
def mock_model():
    return SimpleModel()


@pytest.fixture
def mock_loss_func():
    return CLIPLoss()


@pytest.fixture
def mock_optimizer(mock_model):
    return torch.optim.SGD(mock_model.parameters(), lr=0.01)


@pytest.fixture
def mock_dataloader():
    # Create a dummy dataset list of batches
    # Batch format: (image, {"input_ids": ..., "attention_mask": ...})
    dataset = []
    for _ in range(5):
        image = torch.randn(2, 3, 224, 224)  # Batch size 2
        text_dict = {
            "input_ids": torch.randint(0, 100, (2, 10)),
            "attention_mask": torch.ones(2, 10),
        }
        dataset.append((image, text_dict))
    return dataset


# --- Tests for EarlyStopping ---


def test_early_stopping_initialization(tmp_path):
    ckpt_path = tmp_path / "checkpoint.pt"
    es = EarlyStopping(patience=5, delta=0.1, path=str(ckpt_path))
    assert es.patience == 5
    assert es.delta == 0.1
    assert es.counter == 0
    assert es.best_score is None
    assert es.early_stop is False


def test_early_stopping_improvement(tmp_path):
    ckpt_path = tmp_path / "checkpoint.pt"
    es = EarlyStopping(patience=3, path=str(ckpt_path))
    model = MagicMock()
    # Mock state_dict to return a simple dict
    model.state_dict.return_value = {"dummy": 1}

    with patch("torch.save") as mock_save:
        # First call: best_score becomes -1.0
        es(1.0, model)
        assert es.best_score == -1.0
        assert es.counter == 0
        mock_save.assert_called()

        # Second call: improvement (loss 0.5 -> score -0.5)
        es(0.5, model)
        assert es.best_score == -0.5
        assert es.counter == 0


def test_early_stopping_no_improvement(tmp_path):
    ckpt_path = tmp_path / "checkpoint.pt"
    es = EarlyStopping(patience=2, path=str(ckpt_path))
    model = MagicMock()
    model.state_dict.return_value = {"dummy": 1}

    with patch("torch.save"):
        # Initial best: -1.0
        es(1.0, model)

        # No improvement (loss 1.5 -> score -1.5 < -1.0)
        es(1.5, model)
        assert es.counter == 1
        assert not es.early_stop

        # Still no improvement
        es(1.5, model)
        assert es.counter == 2
        assert es.early_stop


# --- Tests for Trainer ---


def test_trainer_initialization(
    mock_model, mock_dataloader, mock_loss_func, mock_optimizer, tmp_path
):
    save_path = tmp_path / "model.pt"
    trainer = Trainer(
        model=mock_model,
        train_dataloader=mock_dataloader,
        val_dataloader=mock_dataloader,
        loss_func=mock_loss_func,
        optimizer=mock_optimizer,
        device=torch.device("cpu"),
        epochs=10,
        use_early_stop=True,
        patience=3,
        save_path=str(save_path),
    )
    assert trainer.epochs == 10
    assert trainer.early_stop is not None
    assert str(trainer.device) == "cpu"


def test_train_one_epoch(
    mock_model, mock_dataloader, mock_loss_func, mock_optimizer, tmp_path
):
    save_path = tmp_path / "model.pt"
    trainer = Trainer(
        model=mock_model,
        train_dataloader=mock_dataloader,
        val_dataloader=mock_dataloader,
        loss_func=mock_loss_func,
        optimizer=mock_optimizer,
        device=torch.device("cpu"),
        epochs=1,
        use_early_stop=False,
        patience=3,
        save_path=str(save_path),
        show_progress=False,
    )

    loss = trainer.train_one_epoch(1)
    assert isinstance(loss, float)
    assert loss > 0


def test_evaluate_one_epoch(
    mock_model, mock_dataloader, mock_loss_func, mock_optimizer, tmp_path
):
    save_path = tmp_path / "model.pt"
    trainer = Trainer(
        model=mock_model,
        train_dataloader=mock_dataloader,
        val_dataloader=mock_dataloader,
        loss_func=mock_loss_func,
        optimizer=mock_optimizer,
        device=torch.device("cpu"),
        epochs=1,
        use_early_stop=False,
        patience=3,
        save_path=str(save_path),
        show_progress=False,
    )

    loss = trainer.evaluate_one_epoch(1)
    assert isinstance(loss, float)


def test_trainer_run_with_mocks(
    mock_model, mock_dataloader, mock_loss_func, mock_optimizer, tmp_path
):
    """Test full loop with mocked epoch methods to ensure logic flow"""
    save_path = tmp_path / "final_model.pt"

    trainer = Trainer(
        model=mock_model,
        train_dataloader=mock_dataloader,
        val_dataloader=mock_dataloader,
        loss_func=mock_loss_func,
        optimizer=mock_optimizer,
        device=torch.device("cpu"),
        epochs=2,
        use_early_stop=False,
        patience=3,
        save_path=str(save_path),
        show_progress=False,
    )

    # Mocking instance methods
    with (
        patch.object(trainer, "train_one_epoch", return_value=0.5) as mock_train,
        patch.object(trainer, "evaluate_one_epoch", return_value=0.4) as mock_eval,
    ):
        trainer.train()

        assert mock_train.call_count == 2
        assert mock_eval.call_count == 2
        assert len(trainer.metrics["train_loss"]) == 2
        assert save_path.exists()


def test_trainer_early_stopping_trigger(
    mock_model, mock_dataloader, mock_loss_func, mock_optimizer, tmp_path
):
    save_path = tmp_path / "early_stop_model.pt"

    trainer = Trainer(
        model=mock_model,
        train_dataloader=mock_dataloader,
        val_dataloader=mock_dataloader,
        loss_func=mock_loss_func,
        optimizer=mock_optimizer,
        device=torch.device("cpu"),
        epochs=10,
        use_early_stop=True,
        patience=2,
        save_path=str(save_path),
        show_progress=False,
    )

    # Patience is 2.
    # Epoch 1: val_loss=0.5 (best)
    # Epoch 2: val_loss=0.6 (bad, count 1)
    # Epoch 3: val_loss=0.7 (bad, count 2) -> Trigger Stop

    val_losses = [0.5, 0.6, 0.7, 0.8]

    with (
        patch.object(trainer, "train_one_epoch", return_value=0.1),
        patch.object(trainer, "evaluate_one_epoch", side_effect=val_losses),
    ):
        trainer.train()

        # Should stop after epoch 3
        assert len(trainer.metrics["valid_loss"]) == 3
