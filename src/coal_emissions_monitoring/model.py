from typing import Any, Dict
from lightning import LightningModule
import torch
import torch.nn.functional as F
import torchmetrics
from sklearn.metrics import balanced_accuracy_score
import warnings

from coal_emissions_monitoring.constants import POSITIVE_THRESHOLD

# surpress balanced accuracy warning
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true*")


class SmallCNN(torch.nn.Module):
    def __init__(self, num_input_channels: int = 3, num_classes: int = 1):
        super().__init__()
        self.num_input_channels = num_input_channels
        self.num_classes = num_classes
        # build a simple model with EfficientNet-like blocks, global pooling
        # and a final linear layer, compatible with images of size 32x32
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=self.num_input_channels,
                out_channels=16,
                kernel_size=3,
                padding=1,
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1,
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                padding=1,
            ),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(output_size=1),
            torch.nn.Flatten(),
            torch.nn.Linear(128, self.num_classes),
        )

    def forward(self, x):
        return self.model(x)


class CoalEmissionsClassificationModel(LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        learning_rate: float = 1e-3,
        pos_weight: float = 1.0,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.pos_weight = pos_weight
        self.loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.pos_weight))
        self.weight_decay = 0.0

    def forward(self, x):
        preds = self.model(x).squeeze(-1)
        return preds

    def calculate_all_metrics(
        self, preds: torch.Tensor, targets: torch.Tensor
    ) -> Dict[str, float]:
        """
        Calculate metrics for a batch of predictions and targets.

        Args:
            preds (torch.Tensor): predictions
            targets (torch.Tensor): targets

        Returns:
            Dict[str, float]: metrics
        """
        metrics = dict()
        # calculate the cross entropy loss
        metrics["loss"] = self.loss(preds, targets)
        # apply sigmoid to the predictions to get a value between 0 and 1
        preds = torch.sigmoid(preds)
        # calculate emissions vs no-emissions accuracy
        metrics["accuracy"] = (
            ((preds > POSITIVE_THRESHOLD) == (targets > 0)).float().mean()
        )
        # calculate balanced accuracy, which accounts for class imbalance
        metrics["balanced_accuracy"] = balanced_accuracy_score(
            y_pred=(preds.cpu() > POSITIVE_THRESHOLD).int(),
            y_true=targets.cpu().int(),
        )
        # calculate recall and precision
        metrics["recall"] = torchmetrics.functional.recall(
            preds=preds,
            target=targets,
            average="macro",
            task="binary",
        )
        metrics["precision"] = torchmetrics.functional.precision(
            preds=preds,
            target=targets,
            average="macro",
            task="binary",
        )
        # average precision (area under precision-recall curve)
        metrics["average_precision"] = torchmetrics.functional.average_precision(
            preds=preds,
            target=targets.int(),
            task="binary",
        )
        return metrics

    def shared_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
        stage: str,
    ):
        if len(batch["image"].shape) == 0:
            # avoid iteration over a 0-d array error
            return dict()
        metrics = dict()
        x, y = batch["image"], batch["target"]
        x, y = x.float().to(self.device), y.float().to(self.device)
        # forward pass (calculate predictions)
        y_pred = self(x)
        # calculate metrics for the current batch
        metrics = self.calculate_all_metrics(preds=y_pred, targets=y)
        metrics = {
            (f"{stage}_{k}" if k != "loss" or stage != "train" else k): v
            for k, v in metrics.items()
        }
        # log metrics
        for k, v in metrics.items():
            if k == "loss":
                self.log(k, v, on_step=True, prog_bar=True)
            else:
                self.log(k, v, on_step=False, on_epoch=True, prog_bar=True)
        return metrics

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        return self.shared_step(batch, batch_idx, stage="train")

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        return self.shared_step(batch, batch_idx, stage="val")

    def test_step(self, batch: Dict[str, Any], batch_idx: int):
        return self.shared_step(batch, batch_idx, stage="test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.1, patience=5
            ),
            "monitor": "val_loss",
        }

# backward-compat alias for existing notebooks/scripts
CoalEmissionsModel = CoalEmissionsClassificationModel


class CoalEmissionsRegressionModel(LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss = torch.nn.HuberLoss(delta=0.1)

    def forward(self, x):
        preds = self.model(x).squeeze(-1)
        return preds.clamp(0, 1)

    def calculate_all_metrics(
        self, preds: torch.Tensor, targets: torch.Tensor
    ) -> Dict[str, float]:
        metrics = dict()
        clamped = preds.clamp(0, 1)
        metrics["loss"] = self.loss(clamped, targets)
        metrics["rmse"] = torch.sqrt(F.mse_loss(clamped, targets))
        metrics["mae"] = F.l1_loss(clamped, targets)
        ss_res = torch.sum((targets - clamped) ** 2)
        ss_tot = torch.sum((targets - targets.mean()) ** 2)
        if ss_tot > 0:
            metrics["r2"] = 1 - ss_res / ss_tot
        else:
            metrics["r2"] = torch.tensor(0.0, device=preds.device)
        return metrics

    def shared_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
        stage: str,
    ):
        if len(batch["image"].shape) == 0:
            return dict()
        x, y = batch["image"], batch["target"]
        x, y = x.float().to(self.device), y.float().to(self.device)
        y_pred = self(x)
        metrics = self.calculate_all_metrics(preds=y_pred, targets=y)
        metrics = {
            (f"{stage}_{k}" if k != "loss" or stage != "train" else k): v
            for k, v in metrics.items()
        }
        for k, v in metrics.items():
            if k == "loss":
                self.log(k, v, on_step=True, prog_bar=True)
            else:
                self.log(k, v, on_step=False, on_epoch=True, prog_bar=True)
        return metrics

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        return self.shared_step(batch, batch_idx, stage="train")

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        return self.shared_step(batch, batch_idx, stage="val")

    def test_step(self, batch: Dict[str, Any], batch_idx: int):
        return self.shared_step(batch, batch_idx, stage="test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.1, patience=5
            ),
            "monitor": "val_loss",
        }


class MultitaskCNN(torch.nn.Module):
    def __init__(self, num_input_channels: int = 3):
        super().__init__()
        self.backbone = torch.nn.Sequential(
            torch.nn.Conv2d(num_input_channels, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(output_size=1),
            torch.nn.Flatten(),
        )
        self.classification_head = torch.nn.Linear(128, 1)
        self.regression_head = torch.nn.Linear(128, 1)

    def forward(self, x):
        features = self.backbone(x)
        cls_logit = self.classification_head(features).squeeze(-1)
        reg_output = self.regression_head(features).squeeze(-1)
        return cls_logit, reg_output


class CoalEmissionsMultitaskModel(LightningModule):
    def __init__(
        self,
        model: MultitaskCNN,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        pos_weight: float = 1.0,
        cls_weight: float = 0.02,
        reg_weight: float = 0.98,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.cls_loss = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weight)
        )
        self.reg_loss = torch.nn.HuberLoss(delta=0.1)

    def forward(self, x):
        cls_logit, reg_output = self.model(x)
        return cls_logit, reg_output.clamp(0, 1)

    def shared_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
        stage: str,
    ):
        if len(batch["image"].shape) == 0:
            return dict()

        x = batch["image"].float().to(self.device)
        cls_target = batch["target"].float().to(self.device)

        # regression target from metadata; -1.0 means no label (non-US sample)
        reg_target_raw = batch["metadata"]["regression_target"]
        if isinstance(reg_target_raw, (list, tuple)):
            reg_target = torch.tensor(reg_target_raw, dtype=torch.float32).to(self.device)
        else:
            reg_target = reg_target_raw.float().to(self.device)

        cls_logit, reg_pred = self(x)

        # classification loss (all samples)
        loss_cls = self.cls_loss(cls_logit, cls_target)

        # regression loss (only samples with valid regression target)
        reg_mask = reg_target >= 0
        if reg_mask.any():
            loss_reg = self.reg_loss(reg_pred[reg_mask], reg_target[reg_mask])
        else:
            loss_reg = torch.tensor(0.0, device=x.device)

        loss = self.cls_weight * loss_cls + self.reg_weight * loss_reg

        # --- metrics ---
        metrics = {"loss": loss}

        # classification metrics (all samples)
        cls_probs = torch.sigmoid(cls_logit)
        metrics["cls_accuracy"] = (
            ((cls_probs > POSITIVE_THRESHOLD) == (cls_target > 0)).float().mean()
        )
        metrics["cls_balanced_accuracy"] = balanced_accuracy_score(
            y_pred=(cls_probs.cpu() > POSITIVE_THRESHOLD).int(),
            y_true=cls_target.cpu().int(),
        )
        metrics["cls_average_precision"] = torchmetrics.functional.average_precision(
            preds=cls_probs, target=cls_target.int(), task="binary",
        )

        # regression metrics (only valid samples)
        if reg_mask.any():
            clamped = reg_pred[reg_mask].clamp(0, 1)
            valid_targets = reg_target[reg_mask]
            metrics["reg_rmse"] = torch.sqrt(F.mse_loss(clamped, valid_targets))
            metrics["reg_mae"] = F.l1_loss(clamped, valid_targets)
            ss_res = torch.sum((valid_targets - clamped) ** 2)
            ss_tot = torch.sum((valid_targets - valid_targets.mean()) ** 2)
            metrics["reg_r2"] = (1 - ss_res / ss_tot) if ss_tot > 0 else torch.tensor(0.0, device=x.device)

        # prefix and log
        metrics = {
            (f"{stage}_{k}" if k != "loss" or stage != "train" else k): v
            for k, v in metrics.items()
        }
        for k, v in metrics.items():
            if k == "loss":
                self.log(k, v, on_step=True, prog_bar=True)
            else:
                self.log(k, v, on_step=False, on_epoch=True, prog_bar=True)
        return metrics

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        return self.shared_step(batch, batch_idx, stage="train")

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        return self.shared_step(batch, batch_idx, stage="val")

    def test_step(self, batch: Dict[str, Any], batch_idx: int):
        return self.shared_step(batch, batch_idx, stage="test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.1, patience=5
            ),
            "monitor": "val_loss",
        }
