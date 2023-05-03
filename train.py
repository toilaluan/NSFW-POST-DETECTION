from lightning_module import LitClassification
from model import VanillaTransformer
from data import NSFWDataset
import argparse
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
import torch
import torch.nn as nn
from transformers import AutoTokenizer
import torch
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
)

seed_everything(42)


def get_args_parser():
    parser = argparse.ArgumentParser("NSFW Post Detector", add_help=False)
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--train_csv", type=str, default="./dataset/train.csv")
    parser.add_argument("--val_csv", type=str, default="./dataset/val.csv")
    parser.add_argument("--desc", type=str, default="")
    parser.add_argument("--accumulate", type=int, default=1)
    parser.add_argument("--model", default="vanilla")
    parser.add_argument("--max_length", default=128)
    return parser


def make_callbacks(args):
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"train_logs/{args.model}/checkpoints",
        monitor="train_loss",
        mode="min",
        filename=args.desc + "-" + args.model + "-{epoch:02d}-{val_F1:.3f}",
    )
    lr_callback = LearningRateMonitor("step")
    return [lr_callback, checkpoint_callback]


def collate_fn(batch, tokenizer, args):
    texts = []
    labels = []
    for item in batch:
        texts.append(item[0])
        labels.append(item[1])
    encoded_input = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt",
    )
    labels = torch.tensor(labels)
    return encoded_input, labels
    


def main(args):
    tokenizer = DistilBertTokenizer.from_pretrained("michellejieli/NSFW_text_classifier")
    train_dataset = NSFWDataset(
        csv_path=args.train_csv,
        max_length=args.max_length,
        mode="train",
    )
    val_dataset = NSFWDataset(
        csv_path=args.val_csv,
        max_length=args.max_length,
        mode="val",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=16,
        collate_fn=lambda x: collate_fn(x, tokenizer, args),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=16,
        collate_fn=lambda x: collate_fn(x, tokenizer, args),
    )

    model = DistilBertForSequenceClassification.from_pretrained(
        "michellejieli/NSFW_text_classifier"
    )

    L = LitClassification(model, args)

    callbacks = make_callbacks(args)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[args.device],
        max_epochs=args.epochs,
        check_val_every_n_epoch=1,
        callbacks=callbacks,
        log_every_n_steps=10,
        accumulate_grad_batches=args.accumulate,
    )
    print("Number of train samples:", len(train_dataset))
    print("Number of val samples:", len(val_dataset))
    trainer.fit(model=L, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("NSFW", parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
