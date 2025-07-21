import os
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks import ModelCheckpoint


from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

def add_callbacks(args):
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.savedmodel_path,
        filename=f"{args.dataset}_{args.biogpt_model.split('/')[-1]}_epoch{{epoch:02d}}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_weights_only=True
    )

    csv_logger = CSVLogger(save_dir=args.savedmodel_path, name="logs")

    return {
        "callbacks": [checkpoint_callback],
        "loggers": [csv_logger]
    }

    return to_returns
