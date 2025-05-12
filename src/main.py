import mlflow
import numpy as np
import safetensors
import torch
import arguments
from torch import nn
from dataset import CustomDataset
from model import load_model, save_model
from mlflow.models import infer_signature
from PIL import Image
import logs
from torch.utils.data import DataLoader
from torchmetrics.classification import (
    MulticlassF1Score,
    MulticlassAccuracy,
    MulticlassPrecision,
)
import pandas as pd
import io
import time
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
    "batch_size": 4,
    "learning_rate": 1e-4,
    "epochs": 5,
    "log_every_x_batch": 10 if RUN_MODE == "debug" else 100,
    "val_every_x_batch": 100 if RUN_MODE == "debug" else 10000,
    "MODE": RUN_MODE,
}

training_state = {
    "train": {
        "iteration": 0,
        "epochs": 5,
        "loss": 0,
        "smoothed_loss": 0,
        "f1": 0,
        "acc": 0,
        "prec": 0,
    },
    "val": {
        "iteration": 0,
        "epochs": 1,
        "loss": 0,
        "smoothed_loss": 0,
        "f1": 0,
        "acc": 0,
        "prec": 0,
    },
    "test": {
        "iteration": 0,
        "epochs": 1,
        "loss": 0,
        "smoothed_loss": 0,
        "f1": 0,
        "acc": 0,
        "prec": 0,
    },
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

# Load model
logger.info("⏳ Initializing model...")
model = load_model(file_name=LOAD_MODEL_FROM_PATH, arch=args.arch, device=device)

optimizer = torch.optim.SGD(model.parameters(), lr=params["learning_rate"])
criterion = nn.CrossEntropyLoss()


# Helper functions
def get_image_tensor_from_row(row):
    img_field = row["image"]
    img = Image.open(io.BytesIO(img_field["bytes"]))
    pixels = list(img.getdata())
    pixels = np.array(pixels)
    grayscale_tensor = torch.from_numpy(pixels).to(device=device, dtype=torch.float32)

    return grayscale_tensor


def get_label_one_hotted_from_row(row):
    label = torch.tensor(row["label"], device=device)
    return nn.functional.one_hot(label, num_classes=10).float()


def is_mlflow_running() -> bool:
    for _ in range(1, MAX_MLFLOW_CONN_RETRIES + 1):
        if mlflow.active_run() is not None:
            return True
        time.sleep(MLFLOW_CONN_RETRY_INTERVAL_SECONDS)
    return False


def get_dataloader(dataset_file_path, params):
    df = pd.read_parquet(dataset_file_path, engine="pyarrow")
    if not args.release:
        df = df[0:1000]  # Faster iterations
        logger.info("🔍 DEBUG MODE ON: Loading only a fraction of dataset.")

    dataset = CustomDataset(dataframe=df, device=device)
    return DataLoader(dataset, batch_size=params["batch_size"], shuffle=False)


class ExponentialMovingAverage:
    def __init__(self, alpha):
        self.alpha = alpha
        self.smoothed_value = None

    def update(self, new_value):
        if self.smoothed_value is None:
            self.smoothed_value = new_value
        else:
            self.smoothed_value = (
                self.alpha * new_value + (1 - self.alpha) * self.smoothed_value
            )
        return self.smoothed_value

    def get_smoothed_value(self):
        return self.smoothed_value


def log_training_state(training_state, mode, step):
    icon = "⚙️ "
    if mode == "val":
        icon = "🧪 ⚙️ "
    if mode == "test":
        icon = "📐 ⚙️ "

    current_loss = training_state[mode]["loss"]
    current_smoothed_loss = training_state[mode]["smoothed_loss"]
    current_f1 = training_state[mode]["f1"]
    current_acc = training_state[mode]["acc"]
    current_prec = training_state[mode]["prec"]
    logger.info(
        f"{icon} {mode} metrics: Step {step}, smoothed loss: {current_smoothed_loss:.5f}, F1: {current_f1:.5f} (Acc: {current_acc:.5f}, Prec: {current_prec:.5f})"
    )
    metrics_prefix = f"{mode.title()}."
    mlflow.log_metric(
        f"{metrics_prefix}Loss",
        current_loss,
        step=step,
    )
    mlflow.log_metric(
        f"{metrics_prefix}Loss.EMA",
        current_smoothed_loss,
        step=step,
    )
    mlflow.log_metric(f"{metrics_prefix}F1", current_f1, step=step)
    mlflow.log_metric(
        f"{metrics_prefix}Accuracy",
        current_acc,
        step=step,
    )
    mlflow.log_metric(
        f"{metrics_prefix}Precision",
        current_prec,
        step=step,
    )


def run_model(model, dataloader, params, training_state, run_name, mode="train"):
    val_dataloader = None
    if mode == "train":
        # split dataloader into (dataloader, val_dataloader)
        # print(dataloader)
        new_dataset, val_dataset = torch.utils.data.random_split(
            dataloader.dataset, [0.8, 0.2]
        )

        dataloader = DataLoader(
            new_dataset, batch_size=params["batch_size"], shuffle=False
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=params["batch_size"], shuffle=False
        )

    smoothing_factor = 0.18  # Around 10 datapoints
    loss_smoother = ExponentialMovingAverage(smoothing_factor)
    f1_metric = MulticlassF1Score(num_classes=10, average="macro").to(device)
    accuracy_metric = MulticlassAccuracy(num_classes=10, average="macro").to(device)
    precision_metric = MulticlassPrecision(num_classes=10, average="macro").to(device)

    epochs = training_state[mode]["epochs"]
    for epoch in range(1, epochs + 1):
        for batch_index, (batch, labels) in enumerate(dataloader):

            for j, (img, label) in enumerate(zip(batch, labels)):
                # Zero the parameter gradients
                optimizer.zero_grad()

                img = img.reshape((1, 1, 28, 28))
                label = label.reshape((1, 10))

                # forward pass
                logits = model.forward(img)

                # logger.info("logits", logits)
                # logger.info("label", label)
                # sys.exit()

                loss = criterion(logits, label)
                loss_smoother.update(loss.item())
                f1_metric.update(logits, label)
                accuracy_metric.update(logits, label)
                precision_metric.update(logits, label)

                # backward pass
                if mode == "train":
                    loss.backward()

                    optimizer.step()

            if mode == "train" and batch_index % params["log_every_x_batch"] == 0:
                iteration = training_state[mode]["iteration"]

                training_state[mode].update(
                    {
                        "loss": loss.detach().cpu().item(),
                        "smoothed_loss": loss_smoother.get_smoothed_value(),
                        "f1": f1_metric.compute().cpu().item(),
                        "acc": accuracy_metric.compute().cpu().item(),
                        "prec": precision_metric.compute().cpu().item(),
                    }
                )
                log_training_state(
                    training_state=training_state, mode=mode, step=iteration
                )

            training_state[mode]["iteration"] += 1

        if val_dataloader is not None:
            logger.info("🧪 ⚙️  Validating model...")
            training_state = run_model(
                model,
                val_dataloader,
                params,
                run_name=run_name,
                training_state=training_state,
                mode="val",
            )
            log_training_state(training_state=training_state, mode="val", step=epoch)
            save_to = (
                Path(SAVE_MODEL_DIR)
                .joinpath(run_name)
                .joinpath(f"{epoch}-model.safetensors")
            )
            logger.info(f"💾 Saving model at '{save_to}'...")
            save_model(model, save_to)

    training_state[mode].update(
        {
            "loss": loss.detach().cpu().item(),
            "smoothed_loss": loss_smoother.get_smoothed_value(),
            "f1": f1_metric.compute().cpu().item(),
            "acc": accuracy_metric.compute().cpu().item(),
            "prec": precision_metric.compute().cpu().item(),
        }
    )
    return training_state


with mlflow.start_run() as run:

    if not is_mlflow_running():
        logger.error(
            f"❌ MLflow connection failed at '{MLFLOW_URI}', make sure it is possible to connect to the MLflow URI mentioned."
        )
        sys.exit(1)

    run = mlflow.active_run()
    run_name = run.info.run_name

    logger.info(f"🦾 Using device: {device}")
    mlflow.set_tag("device", device)
    mlflow.set_tag("mode", RUN_MODE)
    mlflow.log_params(params)

    logger.info("☁️  Loading dataset...")
    train_dataloader = get_dataloader("dataset/train.parquet", params=params)

    logger.info("⚙️  Training model...")
    run_model(
        model,
        train_dataloader,
        params,
        run_name=run_name,
        training_state=training_state,
    )
    logger.info("✅ Done training")

    save_to = (
        Path(SAVE_MODEL_DIR).joinpath(run_name).joinpath("final-model.safetensors")
    )
    logger.info(f"💾 Saving model at '{save_to}'...")
    save_model(model, save_to)

    logger.info("📐 Starting model test")
    logger.info("📐 ☁️  Loading test dataset...")
    test_dataloader = get_dataloader("dataset/test.parquet", params=params)

    logger.info("📐 ⚙️  Running model on test dataset")
    params["log_every_x_batch"] = 10
    with torch.no_grad():
        training_state = run_model(
            model,
            test_dataloader,
            params,
            run_name=run_name,
            training_state=training_state,
            mode="test",
        )
    current_smoothed_loss = training_state["test"]["smoothed_loss"]
    current_f1 = training_state["test"]["f1"]
    current_acc = training_state["test"]["acc"]
    current_prec = training_state["test"]["prec"]
    logger.info(
        f"📐 📈 Final metrics: smoothed loss: {current_smoothed_loss:.5f}, F1: {current_f1:.5f} (Acc: {current_acc:.5f}, Prec: {current_prec:.5f})"
    )
    logger.info("📐 ✅ Done testing model")

logger.info("👋 Everything OK")
