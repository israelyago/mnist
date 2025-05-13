import mlflow
import lightning as L
import torch
import arguments
from dataset import CustomDataset
import logs
from lightning.pytorch.loggers import MLFlowLogger
from model import MNISTModel
from torch.utils.data import DataLoader
import pandas as pd
import sys
from pathlib import Path
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
)
from ray.train import RunConfig, ScalingConfig, CheckpointConfig
from ray.train.torch import TorchTrainer
from ray import tune
from ray.tune.schedulers import ASHAScheduler

args = arguments.get_args()
logger = logs.get_logger("train")

current_directory = Path.cwd()

MLFLOW_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "MNIST"
LOAD_MODEL_FROM_PATH = args.model
SAVE_MODEL_DIR = "artifacts"
DATA_DIR = current_directory.joinpath("dataset")

VALID_ARCH = ["lenet5", "udlbook"]
if args.arch not in VALID_ARCH:
    logger.error(f"Argument 'arch' must be one of {VALID_ARCH}, got '{args.arch}'")
    sys.exit(1)

# Hyperparameters
default_config = {
    "batch_size": 64,
    "lr": 1e-2,
    "epochs": 10,
}

search_space = {
    "lr": tune.loguniform(1e-2, 1e-1),
    "batch_size": tune.choice([32, 64, 128]),
    "epochs": tune.choice([5, 10]),
}

mlflow.set_tracking_uri(uri=MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

device = "cuda" if torch.cuda.is_available() else "cpu"

if LOAD_MODEL_FROM_PATH is not None:
    model_path_with_name = Path(LOAD_MODEL_FROM_PATH)
    if model_path_with_name.exists() and not model_path_with_name.is_file():
        logger.error(
            f"Environment variable LOAD_MODEL_FROM_PATH MUST be a file, got '{LOAD_MODEL_FROM_PATH}'"
        )
        sys.exit(1)


# Helper functions
def get_dataloader(dataset_file_path, params):
    df = pd.read_parquet(dataset_file_path, engine="pyarrow")
    if not args.release:
        df = df[0:1000]  # Faster iterations
        logger.info("🔍 DEBUG MODE ON: Loading only a fraction of dataset.")

    dataset = CustomDataset(dataframe=df, device=device)
    return DataLoader(dataset, batch_size=params["batch_size"], shuffle=False)


def train_func(config):

    dataloader = get_dataloader(
        DATA_DIR.joinpath("train.parquet"),
        params=config,
    )
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataloader.dataset, [0.8, 0.2]
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=False
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=config["batch_size"], shuffle=False
    )

    mlf_logger = MLFlowLogger(experiment_name="MNIST", tracking_uri=MLFLOW_URI)
    model = None
    if args.model is not None:
        model = MNISTModel.load_from_checkpoint(LOAD_MODEL_FROM_PATH)
    else:
        model = MNISTModel(arch=args.arch, params=config)

    trainer = L.Trainer(
        default_root_dir=Path(SAVE_MODEL_DIR),
        max_epochs=config["epochs"],
        logger=mlf_logger,
        strategy=RayDDPStrategy(),
        callbacks=[RayTrainReportCallback()],
        plugins=[RayLightningEnvironment()],
        enable_progress_bar=False,
    )
    trainer.fit(model, train_dataloader, val_dataloader)

    test_dataloader = get_dataloader(
        DATA_DIR.joinpath("test.parquet"),
        params=config,
    )
    trainer.test(model, test_dataloader)


def tune_mnist_asha(num_samples):

    # The maximum training epochs
    num_epochs = default_config["epochs"]

    scaling_config = ScalingConfig(
        num_workers=1, use_gpu=True, resources_per_worker={"CPU": 1, "GPU": 1}
    )

    run_config = RunConfig(
        checkpoint_config=CheckpointConfig(
            num_to_keep=2,
            checkpoint_score_attribute="val_f1",
            checkpoint_score_order="max",
        ),
    )

    scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)

    ray_trainer = TorchTrainer(
        train_func,
        scaling_config=scaling_config,
        run_config=run_config,
    )

    tuner = tune.Tuner(
        ray_trainer,
        param_space={"train_loop_config": search_space},
        tune_config=tune.TuneConfig(
            metric="val_f1",
            mode="max",
            num_samples=num_samples,
            scheduler=scheduler,
        ),
    )
    return tuner.fit()


results = tune_mnist_asha(num_samples=20)
results.get_best_result(metric="val_f1", mode="max")


logger.info("👋 Everything OK")
