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

seed_everything(42)


def get_args_parser():
    parser = argparse.ArgumentParser("NSFW Post Detector", add_help=False)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument(
        "--train_csv", type=str, default="./dataset/r_dataisbeautiful_posts.csv"
    )
    parser.add_argument("--val_csv", type=str, default="")
    parser.add_argument("--desc", type=str, default="")
    parser.add_argument("--accumulate", type=int, default=1)
    parser.add_argument("--model", default="vanilla")
    parser.add_argument("--max_length", default=64)
    return parser


def load_pretrain(model, path):
    print("Loading pretrain")
    device = torch.device("cpu")
    pretrained_dict = torch.load(path, map_location=device)
    model_dict = model.state_dict()
    print("Pretrain dict:", len(pretrained_dict))
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    print("Filtered dict", len(pretrained_dict))
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model


def make_callbacks(args):
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"train_logs/{args.model}/checkpoints",
        monitor="train_loss",
        mode="min",
        filename=args.desc + "-" + args.model + "-{epoch:02d}-{val_F1:.3f}",
    )
    lr_callback = LearningRateMonitor("step")
    return [lr_callback, checkpoint_callback]


def main(args):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    train_dataset = NSFWDataset(csv_path=args.train_csv, max_length=args.max_length)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16
    )

    model = VanillaTransformer(
        128, 256, 1, 1, 0.1, args.max_length, len(tokenizer.get_vocab())
    )
    print(model)
    L = LitClassification(model, tokenizer, args)

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
    trainer.fit(model=L, train_dataloaders=train_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("NSFW", parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
