import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
)


class LitClassification(pl.LightningModule):
    def __init__(self, model, tokenizer, configs=None):
        super().__init__()
        self.model = model
        self.configs = configs
        self.tokenizer = tokenizer

    def forward(self, x):
        x = self.tokenizer.batch_encode_plus(
            x,
            padding="max_length",
            truncation=True,
            max_length=self.configs.max_length,
            return_attention_mask=True,
            return_tensors="pt",
        )
        x = x["input_ids"].cuda()
        return self.model(x)

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
        out = self.forward(x)
        loss = F.binary_cross_entropy_with_logits(out.float(), y.float())
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.model(x)
        loss = F.binary_cross_entropy_with_logits(out.float(), y.float())
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {
            "pred": out,
            "target": y,
        }
