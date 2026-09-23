import argparse
import csv
import os
import random
import time

# Must be set before CUDA initializes for deterministic CUDA matmul behavior.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoProcessor

from dataset import CXRDeepfakeDataset
from model import MedSiglipDeepfakeClassifier


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    all_logits, all_labels = [], []
    total_loss = 0.0
    for pixel_values, input_ids, attention_mask, labels in loader:
        pixel_values = pixel_values.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        logits = model(pixel_values, input_ids, attention_mask)
        total_loss += criterion(logits, labels.to(device)).item() * pixel_values.size(0)
        all_logits.append(logits.cpu())
        all_labels.append(labels)
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs > 0.5).astype(int)
    return {
        "loss": total_loss / len(loader.dataset),
        "auroc": roc_auc_score(labels, probs),
        "auprc": average_precision_score(labels, probs),
        "f1": f1_score(labels, preds),
    }


def plot_history(history, stage, save_dir):
    epochs = [entry["epoch"] for entry in history]
    metrics = [("loss", "Loss"), ("auroc", "AUROC"), ("auprc", "AUPRC"), ("f1", "F1")]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))

    for axis, (metric, title) in zip(axes.flat, metrics):
        axis.plot(epochs, [entry[f"train_{metric}"] for entry in history], label="Train")
        axis.plot(epochs, [entry[f"val_{metric}"] for entry in history], label="Val")
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.grid(True, alpha=0.3)
        axis.legend()

    figure.suptitle(f"{stage} training curves")
    figure.tight_layout()
    figure.savefig(os.path.join(save_dir, f"{stage}_training_curves.png"), dpi=150)
    plt.close(figure)


def run_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0.0
    for pixel_values, input_ids, attention_mask, labels in loader:
        pixel_values = pixel_values.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        if scaler is not None:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(pixel_values, input_ids, attention_mask)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(pixel_values, input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * pixel_values.size(0)
    return total_loss / len(loader.dataset)


def main():

    args = {
        "csv": "",
        "backbone": "google/medsiglip-448",
        "epochs_probehead": 5,
        "epochs_finetune": 5,
        "unfreeze_blocks": 4,
        "batch_size": 32,
        "lr_head": 1e-3,
        "lr_backbone": 1e-5,
        "num_workers": 4,
        "max_text_length": 64,
        "model_save_dir": "",
        "seed": 1337,
    }

    set_seed(args["seed"])
    os.makedirs(args["model_save_dir"], exist_ok=True)

    log_file_path = os.path.join(args["model_save_dir"], "train_log.csv")
    log_fields = [
        "stage", "epoch", "epoch_time", "train_loss", "train_auroc", "train_auprc", "val_loss", "train_f1", "val_auroc", "val_auprc", "val_f1",
    ]
    with open(log_file_path, "w", newline="") as log_file:
        csv.DictWriter(log_file, fieldnames=log_fields).writeheader()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # AutoProcessor handles both image preprocessing and text tokenization for
    # MedSigLIP (unlike AutoImageProcessor, which only did images for RAD-DINO).
    processor = AutoProcessor.from_pretrained(args["backbone"], token = "")

    # Keep augmentation mild: no horizontal flip (laterality markers matter on CXR),
    # no heavy blur/noise (can erase the exact high-frequency artifacts you're detecting).
    train_augment = transforms.Compose(
        [
            transforms.RandomApply([transforms.ColorJitter(brightness=0.1, contrast=0.1)], p=0.5),
            transforms.RandomRotation(degrees=5),
        ]
    )

    train_ds = CXRDeepfakeDataset(args["csv"], "train", processor, augment=train_augment, max_length=args["max_text_length"])
    val_ds = CXRDeepfakeDataset(args["csv"], "val", processor, max_length=args["max_text_length"])

    train_generator = torch.Generator()
    train_generator.manual_seed(args["seed"])
    val_generator = torch.Generator()
    val_generator.manual_seed(args["seed"] + 1)

    train_loader = DataLoader(
        train_ds,
        batch_size=args["batch_size"],
        shuffle=True,
        num_workers=args["num_workers"],
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=train_generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args["batch_size"],
        shuffle=False,
        num_workers=args["num_workers"],
        worker_init_fn=seed_worker,
        generator=val_generator,
    )

    model = MedSiglipDeepfakeClassifier(args["backbone"], num_unfrozen_blocks=0).to(device)

    n_pos = int((train_ds.df["label"] == 1).sum())
    n_neg = int((train_ds.df["label"] == 0).sum())
    pos_weight = torch.tensor(n_neg / max(n_pos, 1), dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"train set: {n_neg} real / {n_pos} fake -> pos_weight={pos_weight.item():.2f}")

    # scaler = torch.amp.GradScaler() if device == "cuda" else None
    scaler = None
    best_auroc = 0.0

    # ---- Stage 1: linear probe on frozen backbone (image + text embeddings) ----
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=args["lr_head"])
    probe_history = []
    for epoch in range(args["epochs_probehead"]):
        epoch_start_time = time.time()
        loss = run_epoch(model, train_loader, optimizer, criterion, device, scaler)
        train_metrics = evaluate(model, train_loader, device, criterion)
        val_metrics = evaluate(model, val_loader, device, criterion)
        history_entry = {
            "epoch": epoch,
            "epoch_time": time.time() - epoch_start_time,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        probe_history.append(history_entry)
        with open(log_file_path, "a", newline="") as log_file:
            csv.DictWriter(log_file, fieldnames=log_fields).writerow({"stage": "probehead", **history_entry})
        print(
            f"[head {epoch}] loss={loss:.4f} "
            f"val_auroc={val_metrics['auroc']:.4f}"
        )
        if val_metrics["auroc"] > best_auroc:
            best_auroc = val_metrics["auroc"]
            torch.save(model.state_dict(), os.path.join(args["model_save_dir"], "best_model_probehead.pt"))
        torch.save(model.state_dict(), os.path.join(args["model_save_dir"], "latest_model_probehead.pt"))
    plot_history(probe_history, "probehead", args["model_save_dir"])

    # ---- Stage 2: unfreeze last N blocks of BOTH towers, fine-tune end-to-end ----
    model.unfreeze_last_blocks(args["unfreeze_blocks"])
    optimizer = torch.optim.AdamW(
        [
            {"params": model.head.parameters(), "lr": args["lr_head"]},
            {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": args["lr_backbone"]},
        ]
    )
    finetune_history = []
    for epoch in range(args["epochs_finetune"]):
        epoch_start_time = time.time()
        loss = run_epoch(model, train_loader, optimizer, criterion, device, scaler)
        train_metrics = evaluate(model, train_loader, device, criterion)
        val_metrics = evaluate(model, val_loader, device, criterion)
        history_entry = {
            "epoch": epoch,
            "epoch_time": time.time() - epoch_start_time,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        finetune_history.append(history_entry)
        with open(log_file_path, "a", newline="") as log_file:
            csv.DictWriter(log_file, fieldnames=log_fields).writerow({"stage": "finetune", **history_entry})
        print(
            f"[finetune {epoch}] loss={loss:.4f} "
            f"val_auroc={val_metrics['auroc']:.4f}"
        )
        if val_metrics["auroc"] > best_auroc:
            best_auroc = val_metrics["auroc"]
            torch.save(model.state_dict(), os.path.join(args["model_save_dir"], "best_model_finetune.pt"))
        torch.save(model.state_dict(), os.path.join(args["model_save_dir"], "latest_model_finetune.pt"))
    plot_history(finetune_history, "finetune", args["model_save_dir"])


if __name__ == "__main__":
    main()