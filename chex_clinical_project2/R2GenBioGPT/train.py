import os
from pprint import pprint
import torch
import matplotlib.pyplot as plt
import lightning.pytorch as pl
from lightning.pytorch import seed_everything, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, Callback
from lightning.pytorch.loggers import TensorBoardLogger
from configs.config import parser
from dataset.data_module import DataModule
from models.R2GenBioGPT import R2GenBioGPT


class TrainLossLogger(Callback):
    def __init__(self):
        self.train_losses = []
        self.val_losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        loss = trainer.callback_metrics.get("train_loss")
        if loss is not None:
            self.train_losses.append(loss.item())

    def on_validation_end(self, trainer, pl_module):
        val_loss = trainer.callback_metrics.get("val_loss")
        if val_loss is not None:
            self.val_losses.append(val_loss.item())

    def on_train_end(self, trainer, pl_module):
        plt.figure()
        plt.plot(self.train_losses, label="Train Loss")
        if self.val_losses:
            plt.plot(self.val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training and Validation Loss")
        save_path = os.path.join(pl_module.args.savedmodel_path, "loss_curve.png")
        plt.savefig(save_path)
        print(f"[INFO] Saved loss curve to {save_path}")


def train(args):
    seed_everything(42, workers=True)
    os.makedirs(args.savedmodel_path, exist_ok=True)

    #model = R2GenBioGPT(args)
    if args.ckpt_file:
        model = R2GenBioGPT.load_from_checkpoint(args.ckpt_file, strict=False)
    else:
        model = R2GenBioGPT(args)

    dm = DataModule(args)

    loss_logger = TrainLossLogger()
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.savedmodel_path,
        filename="mimic_cxr_biogpt_epoch={epoch}",
        save_top_k=1,
        monitor="val_loss",
        mode="min"
    )
    tb_logger = TensorBoardLogger(save_dir=args.savedmodel_path, name="tb_logs")

    ckpt_path = "auto" if getattr(args, "resume_training", False) else None
    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        strategy=args.strategy if hasattr(args, "strategy") else "auto",
        callbacks=[checkpoint_callback, loss_logger],
        logger=tb_logger,
        log_every_n_steps=10,
        val_check_interval=args.val_check_interval
    )

    if args.test:
        trainer.test(model, datamodule=dm)
    elif args.validate:
        trainer.validate(model, datamodule=dm)
    else:
        trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path)
        trainer.test(model, datamodule=dm)


def main():
    args = parser.parse_args()
    pprint(vars(args))
    train(args)


if __name__ == "__main__":
    main()
