"""
Q: How to send parameters for lightning module?
send `config` to the module via the train function
Q: How to add metric for tuning? 
Use self.log in validation_epoch_end:
self.log("ptl/val_loss", avg_loss)
self.log("ptl/val_accuracy", avg_acc)
Q: How to share large dataset between trail?
Using dataloader_adaptor
"""
from ray import tune
from ray.tune.schedulers import HyperBandScheduler
from ray.tune import CLIReporter
import os
from functools import partial
from train_function import train_cifar
from test_function import test_accuracy
import numpy as np
from data_module import CiFarModule
def main(num_samples=10, max_num_epochs=10, gpus_per_trial=2):
    config = {
        "l1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "l2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([2, 4, 8, 16])
    }
    scheduler = HyperBandScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        reduction_factor=2)
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "accuracy", "training_iteration"])
    data_dir = os.path.abspath("./data")
    dm = CiFarModule(data_dir=data_dir, batch_size=32)
    dm.prepare_data()
    # 用 with_parameters, DM 會被整包放進object store去被共享! 不需要特別去做dataloader的處理
    result = tune.run(
        tune.with_parameters(train_cifar, dm=dm),
        resources_per_trial={"cpu": 1, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))

    best_trained_model = Net(best_trial.config["l1"], best_trial.config["l2"])
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)

    test_acc = test_accuracy(best_trained_model, device)
    print("Best trial test set accuracy: {}".format(test_acc))


if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=10, max_num_epochs=10, gpus_per_trial=0)