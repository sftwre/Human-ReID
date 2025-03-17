from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets
import torch
import torch.nn as nn
import argparse
from model import ReIDModel
from torchmetrics.functional.classification import multiclass_average_precision
from torch.utils.tensorboard import SummaryWriter
import os
import time
import mlflow
from datasets import transforms_minimal as transform


def train(
    model: nn.Module,
    dataloader: DataLoader,
    criterion,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """
    Trains model for one epoch.

    Args:
        model: model to train
        dataloader: training DataLoader
        criterion: loss function
        optimizer: optimization algorithm
        device: device to train on

    Returns:
        avg_loss: average loss for epoch
    """

    model.train()

    epoch_loss = 0

    for batch in dataloader:
        images, labels = batch

        images = images.to(device)
        labels = labels.to(device)

        # zero gradients from previous step
        optimizer.zero_grad()

        preds = model(images)

        # _, preds = torch.max(y_hat.data, 1)
        loss = criterion(preds, labels)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)

    return avg_loss


@torch.no_grad()
def val(
    model: nn.Module,
    dataloader: DataLoader,
    criterion,
    device: torch.device,
    eval=False,
) -> (float, float):
    """
    Evaluates model on validation set and returns mAP and average loss.
    The mAP is only computed when the eval flag is set to True.

    Args:
        model: model to evaluate
        dataloader: validation DataLoader
        criterion: loss function
        device: inference device
        eval: flag to compute mAP of classifier on validation set

    Returns:
        mAP: mean average precision on validation set. (-1 if eval=False)
        avg_loss: average loss for epoch
    """

    model.eval()

    n_batches = len(dataloader)
    n_classes = len(dataloader.dataset.classes)

    if eval:
        all_preds = torch.FloatTensor()
        all_labels = torch.LongTensor()

    eval_loss = 0
    for batch in dataloader:
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)

        preds = model(images)

        loss = criterion(preds, labels)

        eval_loss += loss.item()

        # track predictions and labels for evaluation
        if eval:
            preds = preds.to("cpu").detach()
            labels = labels.to("cpu")
            all_preds = torch.cat((all_preds, preds), 0)
            all_labels = torch.cat((all_labels, labels), 0)

    avg_loss = eval_loss / n_batches

    mAP = -1

    if eval:
        ap_cls_scores = multiclass_average_precision(
            all_preds, all_labels, num_classes=n_classes, average=None
        )
        mAP = ap_cls_scores.mean().item()

    return mAP, avg_loss


def save_model(model: nn.Module, exp_dir: str, filename: str):
    """

    Writes model weights to disk.
    Model weights are saved to ./models/<exp_dir>/<filename>.pth

    Args:
        model: trained model weights
        exp_dir: experiment directory
        filename: name of model file
    """

    model_dir = Path(f"./models/{exp_dir}")

    if not model_dir.exists():
        os.makedirs(model_dir)

    torch.save(model.state_dict(), model_dir / filename)

    print(f"Model weights saved to -> {model_dir / filename}")


if __name__ == "__main__":

    torch.manual_seed(123)

    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=2e-2)
    parser.add_argument("--exp_tag", type=str, default="baseline")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_data_loaders", type=int, default=4)
    parser.add_argument("--data_root", type=str, default="./data/market-1501-grouped")
    parser.add_argument(
        "--eval_freq", type=int, default=2, help="How many epochs to evaluate model"
    )

    args = parser.parse_args()

    dataset_base = Path(args.data_root)

    # load training and validation datasets
    train_dataset = datasets.ImageFolder(
        root=dataset_base / "train", transform=transform
    )

    val_dataset = datasets.ImageFolder(root=dataset_base / "val", transform=transform)

    # create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_data_loaders,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_data_loaders,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    n_classes = len(train_dataset.classes)
    model = ReIDModel(n_classes).to(device)

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    mlflow.set_experiment("Preliminary training")

    # start mlflow run
    try:

        start_time = time.time()

        mlflow.set_tracking_uri("http://127.0.0.1:8080")

        with mlflow.start_run() as run:

            eval_freq = args.eval_freq
            n_epochs = args.num_epochs

            exp_name = run.info.run_name
            writer = SummaryWriter(log_dir=f"./runs/{exp_name}")
            best_val_loss = float("inf")

            # train model
            for epoch in range(1, n_epochs + 1):

                eval_epoch = epoch % eval_freq == 0

                train_loss = train(model, train_loader, loss_func, optimizer, device)
                mAP, val_loss = val(
                    model, val_loader, loss_func, device, eval=eval_epoch
                )

                # log metrics to tensorboard
                writer.add_scalar("Train_loss", train_loss, epoch)
                writer.add_scalar("Val_loss", val_loss, epoch)

                if eval_epoch:
                    writer.add_scalar("mAP", mAP, epoch)

                log = f"Epoch [{epoch}]/[{n_epochs}] -> Train loss: {train_loss:.2f}, Val loss: {val_loss:.2f}"

                if eval_epoch:
                    log += f", mAP: {mAP*100:.2f}%"

                print(log)

                # save best performing model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_model(
                        model,
                        exp_dir=exp_name,
                        filename=f"best.pth",
                    )

            end_time = time.time()
            exec_time_minutes = (end_time - start_time) / 60

            # log experiment config in mlflow

            exp_params = {}
            for attr_name, attr_val in args.__dict__.items():
                exp_params[attr_name] = attr_val

            mlflow.log_params(exp_params)
            mlflow.set_tag("Training time", f"{exec_time_minutes:.2f} mins")
            mlflow.set_tag("Experiment", args.exp_tag)

    except Exception as e:
        mlflow.log_param("Exception", str(e))
        mlflow.end_run(status="FAILED")
