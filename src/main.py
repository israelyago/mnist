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

args = arguments.get_args()
logger = logs.get_logger("train")

RUN_MODE = "release" if args.release else "debug"
MAX_MLFLOW_CONN_RETRIES = 10
MLFLOW_CONN_RETRY_INTERVAL_SECONDS = 1
MLFLOW_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "MNIST"
LOAD_MODEL_FROM_PATH = args.model
SAVE_MODEL_DIR = "artifacts"

VALID_ARCH = ["lenet5", "udlbook"]
if args.arch not in VALID_ARCH:
    logger.error(f"Argument 'arch' must be one of {VALID_ARCH}, got '{args.arch}'")
    sys.exit(1)

# Hyperparameters
params = {
    "batch_size": 64,
    "learning_rate": 1e-3,
    "epochs": 5,
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


dataloader = get_dataloader("dataset/train.parquet", params=params)
train_dataset, val_dataset = torch.utils.data.random_split(
    dataloader.dataset, [0.8, 0.2]
)

train_dataloader = DataLoader(
    train_dataset, batch_size=params["batch_size"], shuffle=False
)
val_dataloader = DataLoader(val_dataset, batch_size=params["batch_size"], shuffle=False)

mlf_logger = MLFlowLogger(experiment_name="MNIST", tracking_uri=MLFLOW_URI)
model = None
if args.model is not None:
    model = MNISTModel.load_from_checkpoint(LOAD_MODEL_FROM_PATH)
else:
    model = MNISTModel(arch=args.arch, params=params)

trainer = L.Trainer(
    default_root_dir=Path(SAVE_MODEL_DIR),
    max_epochs=params["epochs"],
    logger=mlf_logger,
)
trainer.fit(model, train_dataloader, val_dataloader)

test_dataloader = get_dataloader("dataset/test.parquet", params=params)
trainer.test(model, test_dataloader)

logger.info("👋 Everything OK")
