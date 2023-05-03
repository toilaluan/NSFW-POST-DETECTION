import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
    Accuracy,
    Precision,
    Recall,
)


class LitClassification(pl.LightningModule):
    def __init__(self, model, configs=None):
        super().__init__()
        self.model = model
        self.configs = configs
        self.val_acc = Accuracy("multiclass", num_classes=2)
        self.val_p = Precision("multiclass", num_classes=2)
        self.val_r = Recall("multiclass", num_classes=2)

    def forward(self, x):
        return self.model(**x)

    def on_fit_start(self) -> None:
        return super().on_fit_start()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.configs.lr,
            weight_decay=self.configs.weight_decay,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, self.trainer.max_epochs, eta_min=1e-6
                ),
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        out = self.forward(x).logits
        loss = F.cross_entropy(out.float(), y.long())
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x).logits
        loss = F.cross_entropy(out.float(), y.long())
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.val_acc.update(out, y)
        self.val_p.update(out, y)
        self.val_r.update(out, y)

    def on_validation_epoch_end(self) -> None:
        self.log("val_acc", self.val_acc.compute(), prog_bar=True)
        self.log("val_p", self.val_p.compute(), prog_bar=True)
        self.log("val_r", self.val_r.compute(), prog_bar=True)
        self.val_acc.reset()
        self.val_p.reset()
        self.val_r.reset()
