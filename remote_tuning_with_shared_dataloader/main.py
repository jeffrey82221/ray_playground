"""
Q: How to send parameters for lightning module?
send `config` to the module via the train function
Q: How to add metric for tuning? 
Use self.log in validation_epoch_end:
self.log("ptl/val_loss", avg_loss)
self.log("ptl/val_accuracy", avg_acc)
Q: How to share large dataset between trail?
"""
from pl_module import LightningMNISTClassifier
from data_module import MNISTDataModule
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.pytorch_lightning import TuneReportCallback
import os
import math

def train_mnist(config):
    model = LightningMNISTClassifier(config)
    trainer = pl.Trainer(max_epochs=10, enable_progress_bar=False)

    trainer.fit(model)

def train_mnist_no_tune():
    config = {
        "layer_1_size": 128,
        "layer_2_size": 256,
        "lr": 1e-3,
        "batch_size": 64
    }
    train_mnist(config)
import logging
def train_mnist_tune(config, num_epochs=10, num_gpus=0, dm=None):
    assert dm is not None
    model = LightningMNISTClassifier(config)
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        # If fractional GPUs passed in, convert to int.
        gpus=math.ceil(num_gpus),
        logger=TensorBoardLogger(
            save_dir=tune.get_trial_dir(), name="", version="."),
        enable_progress_bar=True,
        callbacks=[
            TuneReportCallback(
                {
                    "loss": "ptl/val_loss",
                    "mean_accuracy": "ptl/val_accuracy"
                },
                on="validation_end")
        ])
    trainer.fit(model, dm)
    dm.close()

if __name__ == '__main__':
    
    config = {
        "layer_1_size": tune.choice([32, 64, 128]),
        "layer_2_size": tune.choice([64, 128, 256]),
        "lr": tune.loguniform(1e-4, 1e-1)
    }
    num_epochs = 10
    scheduler = ASHAScheduler(
        max_t=num_epochs,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        parameter_columns=["layer_1_size", "layer_2_size", "lr"],
        metric_columns=["loss", "mean_accuracy", "training_iteration"]
    )
    gpus_per_trial = 0
    data_dir = "~/data"
    data_dir = os.path.expanduser(data_dir)
    dm = MNISTDataModule(data_dir=data_dir, batch_size=32)
    dm.prepare_data()
    train_fn_with_parameters = tune.with_parameters(train_mnist_tune,
                                                    num_epochs=num_epochs,
                                                    num_gpus=gpus_per_trial,
                                                    dm=dm)
    resources_per_trial = {"cpu": 1, "gpu": gpus_per_trial}

    num_samples = 10
    analysis = tune.run(train_fn_with_parameters,
        resources_per_trial=resources_per_trial,
        metric="loss",
        mode="min",
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        name="tune_mnist_asha")

    print("Best hyperparameters found were: ", analysis.best_config)